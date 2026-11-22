"""The refit bootstrap of a fit: resample whole panels (or clusters) and refit.

`DPDResult.refit_bootstrap` calls `refit_bootstrap` here. The resampling, the
refits in worker processes, the failure ledger and the intervals are
hdfe_stream.core's (``core.refit``). This module owns what is particular to
dynamic panel GMM: the columns a fit reads, the resampling unit the fit's
clustering implies, and the recentering of the moments at the original
estimate (Hall and Horowitz 1996), which makes the moment conditions hold in
the bootstrap's population, so that studentized intervals and the Hansen test
can be bootstrapped. The refits keep the original fit's instrument columns
(`fixed_columns`), so the shift lines up with them, and skip every statistic a
draw does not need (`estimates_only`).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from hdfe_stream.core.partition import preprocess_budget_mb
from hdfe_stream.core.refit import (
    BootstrapFailureWarning,
    RefitBootstrapResult,
    refit_draws,
    refit_workers,
)
from hdfe_stream.core.utils import as_lazy
from stream_limits import MemoryBudget, thread_scope

from .estimator import DPD, WORKDIR_ENV

__all__ = ["BootstrapFailureWarning", "DPDBootstrapResult", "refit_bootstrap"]


@dataclass(kw_only=True)
class DPDBootstrapResult(RefitBootstrapResult):
    """The refit bootstrap of a `dpd_stream` fit (see `DPDResult.refit_bootstrap`).

    As `hdfe_stream.core.refit.RefitBootstrapResult` -- ``tidy()``,
    ``confint()``, ``standard_errors()``, ``draws()``, ``failures()`` -- with,
    when the draws were recentered, `hansen` (the bootstrap Hansen test).
    ``diagnostics["recentered"]`` says whether they were, and
    ``diagnostics["recenter_reason"]`` why not.
    """

    original_hansen: dict | None = None
    _hansen_draws: np.ndarray | None = field(default=None, repr=False)

    def hansen(self, *, allow_incomplete=False) -> dict:
        """The Hansen test of the overidentifying restrictions with a bootstrap
        p-value: the share of the recentered draws' statistics J* at least the
        fit's J, as (1 + #{J* >= J}) / (1 + B) over the B successful draws.
        Also the fit's own (asymptotic chi-squared) p-value.

        With many instruments relative to the panels, this test does not perform
        as well as the studentized intervals. It rejects even less often than the
        asymptotic test, both with valid instruments and with an invalid one, so
        it does not repair the Hansen test's weakness there. With few instruments
        the two tests agree. See the user guide's "Bootstrap inference"."""
        if self._hansen_draws is None:
            raise ValueError(
                "no bootstrap Hansen test: it needs recentered draws of an overidentified "
                "model with a Hansen statistic (see diagnostics['recenter_reason'])")
        if self._failures and not allow_incomplete:
            raise ValueError("some refits failed; inspect failures() and explicitly set "
                             "allow_incomplete=True for inference conditional on success")
        jstar = self._hansen_draws[np.isfinite(self._hansen_draws)]
        if len(jstar) < 2:
            raise ValueError("at least two successful refits are required for bootstrap "
                             "inference")
        h = self.original_hansen
        return dict(statistic=h["statistic"], df=h["df"], p=h["p"],
                    bootstrap_p=float((1 + (jstar >= h["statistic"]).sum()) / (1 + len(jstar))),
                    successful_reps=len(jstar))

    def summary_dict(self):
        out = super().summary_dict()
        if self._hansen_draws is not None:
            try:
                out["hansen"] = self.hansen(allow_incomplete=True)
            except ValueError:
                out["hansen"] = None
        return out


def _refit_task(task):
    """One draw's refit (core.refit calls it, in this process or a worker)."""
    if task["dates"] != task["expected_dates"]:
        raise ValueError("the resample lacks some of the original fit's periods, which "
                         "changes the instruments")
    args = dict(task["args"])
    if task["cluster"] is not None:
        # each copy of a cluster is a cluster of its own
        args["vcov"] = {"CRV1": task["group_column"]}
    elif task["panel_cluster"]:
        # clustered by panel: by each copy's own ID
        args["vcov"] = {"CRV1": task["unit_column"]}
    est = DPD(task["fml"], task["path"], (task["unit_column"], task["time"]), **args,
              tdelta=task["tdelta"], workdir=task["workdir"], n_threads=task["threads"],
              max_memory_gb=task["memory_gb"], verbose=False, estimates_only=True,
              draw_se=task["draw_se"], moment_shift=task["moment_shift"],
              fixed_columns=task["live_mask"], fixed_xsh=task["xsh"])
    f = est.fit()
    if f.info["keep_names"] != task["keep_names"]:
        raise ValueError("the resample drops a different set of regressors as collinear")
    idx = task["index"]
    values = [f.b[idx]]
    if task["draw_se"]:
        values.append(np.sqrt(np.diag(f.V))[idx])
    if task["hansen"]:
        values.append([np.nan if f.info["hansen"] is None else f.info["hansen"]])
    return np.concatenate(values)


def _recenter_reasons(result, args):
    """Why the draws cannot be recentered (empty if they can)."""
    m = result._model
    reasons = []
    if args.get("weights"):
        reasons.append("weights (their normalization changes in every draw)")
    if result.absorbed and result.absorbed.get("method") == "cells":
        reasons.append(f"an absorbed effect ('| {result.absorbed['term']}')")
    if args.get("nl"):
        reasons.append("nl()")
    if args.get("pca"):
        reasons.append("pca (the components would be re-estimated in every draw)")
    if not reasons and m.get("moment_sum") is None:
        reasons.append("this model")
    return reasons


def refit_bootstrap(result, *, reps=199, seed=None, coefficients=None, recenter="auto",
                    draw_standard_errors=True, n_workers=1, n_threads=None,
                    max_memory_gb=None, memory_reserve=None, workdir=None,
                    preprocess_mb=None) -> DPDBootstrapResult:
    """See `DPDResult.refit_bootstrap`."""
    m = result._model
    if not m or m.get("args") is None:
        raise ValueError("this result cannot be bootstrapped: call refit_bootstrap on the "
                         "result dpd_stream returned (not on its onestep result)")
    refit_workers(reps, n_workers, n_threads)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if recenter not in ("auto", True, False):
        raise ValueError("recenter must be 'auto', True or False")
    args = dict(m["args"])
    panel = tuple(m["panel"])

    # the resampling unit: panels, or the clusters of a clustering by a variable
    # that is constant within panels
    cluster, panel_cluster = None, False
    clus = list(m["clusterings"])
    if len(clus) > 1:
        raise ValueError("multi-way clustering has no resampling scheme here: bootstrap "
                         "with a one-way clustering")
    if clus:
        vars_, nested, by_obs, is_panel = clus[0]
        panel_cluster = is_panel
        if not is_panel:
            if by_obs or not nested:
                raise ValueError(f"clustering by {'+'.join(vars_)}, which varies within "
                                 "panels: whole panels cannot be resampled by it")
            cluster = vars_[0]

    # recentering
    reasons = _recenter_reasons(result, args)
    if recenter is True and reasons:
        raise ValueError("recenter=True is not available with " + "; ".join(reasons))
    recentered = recenter is not False and not reasons
    hansen = recentered and result.hansen is not None and result.hansen["df"] > 0

    # the coefficients
    names = list(result.coefnames)
    if coefficients is None:
        wanted = names
    else:
        if isinstance(coefficients, str):
            raise ValueError("coefficients must be a sequence of exact coefficient names")
        wanted = list(coefficients)
        if not wanted or len(set(wanted)) != len(wanted) or not set(wanted) <= set(names):
            raise ValueError("requested coefficient names are missing, duplicated or empty")
    index = np.array([names.index(n) for n in wanted], np.int64)
    V0 = result.all_vcovs[m["vkey"]][0] if m["vkey"] in result.all_vcovs else result.vcov
    beta = np.asarray(result.beta, float)[index]
    se0 = np.sqrt(np.diag(V0))[index]
    if draw_standard_errors and not np.isfinite(se0).all():
        raise ValueError("the fit's standard errors of the requested coefficients are not "
                         "finite: use draw_standard_errors=False")

    budget = MemoryBudget(max_memory_gb, memory_reserve)
    refit_preprocess_mb = preprocess_mb
    preprocess_mb = preprocess_budget_mb(64 if preprocess_mb is None else preprocess_mb, budget)
    if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
        raise MemoryError("bootstrap preprocessing buffers exceed job headroom")
    base = workdir or os.environ.get(WORKDIR_ENV)
    if base is not None:
        Path(base).mkdir(parents=True, exist_ok=True)
    width = len(index) * (2 if draw_standard_errors else 1) + (1 if hansen else 0)
    with thread_scope(n_threads), TemporaryDirectory(prefix="dpd-stream-refit-",
                                                     dir=base) as run:
        run = Path(run)
        snapshot = run / "source.parquet"
        src = as_lazy(m["source"])
        missing = set(m["columns"]) - set(src.collect_schema().names())
        if missing:
            raise ValueError(f"the source no longer has the columns {sorted(missing)}")
        (src.select(m["columns"])
            .filter(pl.col(panel[0]).is_not_null() & pl.col(panel[1]).is_not_null())
            .sink_parquet(snapshot, engine="streaming", row_group_size=32768))
        dates = (pl.scan_parquet(snapshot).select(pl.col(panel[1]).unique().sort())
                 .collect(engine="streaming")[panel[1]].to_list())
        n_shift = None
        if recentered:
            if cluster is None:
                n_shift = m["n_panels"]
            else:
                n_shift = (pl.scan_parquet(snapshot).select(pl.col(cluster).n_unique())
                           .collect(engine="streaming").item())
        task = dict(
            fml=m["fml"], time=panel[1], args=args, tdelta=m["tdelta"],
            expected_dates=dates, cluster=cluster, panel_cluster=panel_cluster,
            draw_se=bool(draw_standard_errors),
            moment_shift=(m["moment_sum"], n_shift) if recentered else None,
            live_mask=m["live_mask"], xsh=m["xsh"], keep_names=m["keep_names"],
            index=index, hansen=hansen)
        out = refit_draws(
            snapshot, run, fit=_refit_task, task=task, width=width, reps=reps, seed=seed,
            workers=n_workers, threads=n_threads, memory_gb=max_memory_gb, budget=budget,
            preprocess_mb=preprocess_mb,
            key=cluster if cluster is not None else panel[0],
            within=panel[0] if cluster is not None else None,
            sort_columns=[cluster, *panel] if cluster is not None else list(panel),
            time_column=panel[1], expected_groups=m["n_panels"], expected_rows=m["n_rows"],
            worker_minimum_gb=0.35 + (refit_preprocess_mb or 64) / 1000 + 0.096,
            preload=("dpd_stream",))
    k = len(index)
    draws = out.draws[:, :k]
    se_draws = out.draws[:, k:2 * k] if draw_standard_errors else None
    j_draws = out.draws[:, -1] if hansen else None
    diagnostics = dict(
        out.diagnostics,
        sampling=(f"whole clusters of {cluster} with replacement; each copy a cluster of "
                  "its own, its panels with IDs of their own" if cluster is not None
                  else "whole panels with replacement; each copy with an ID of its own"),
        recentered=recentered,
        recenter_reason=None if recentered else ("recenter=False" if recenter is False
                                                 else "not available with "
                                                 + "; ".join(reasons)),
        moment_shift_units=n_shift,
        analytic_vcov_type=m["vkey"],
        panels=m["n_panels"], source_rows=m["n_rows"], periods=dates,
        fit_options={kk: v for kk, v in args.items()},
    )
    boot = DPDBootstrapResult(
        fml=m["fml"], estimator="dpd_stream", coefnames=wanted, beta=beta,
        diagnostics=diagnostics, _draws=draws, _failures=out.failures,
        _warnings=out.warnings,
        original_se=se0 if draw_standard_errors else None, _se_draws=se_draws,
        original_hansen=dict(result.hansen) if hansen else None, _hansen_draws=j_draws)
    if out.failures:
        warnings.warn(
            f"{len(out.failures)} of {reps} refits failed. Inspect failures(); incomplete-run "
            "inference requires allow_incomplete=True and is conditional on successful fits.",
            BootstrapFailureWarning, stacklevel=3)
    return boot
