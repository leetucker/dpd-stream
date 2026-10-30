"""`DPDResult`: a fitted dynamic panel GMM model, and `DPDMulti`.

A `hdfe_stream.core.results_base.RegressionResult` subclass, so the
coefficient table, confidence intervals, Wald tests, the ``summary*``
reports, `etable` and the multiple-testing corrections are hdfe-stream's,
with normal inference (z and chi-squared) unless the fit was ``small``
(t and F, as xtabond2's ``small``). The specification tests --
Arellano-Bond, Sargan, Hansen, difference-in-Hansen -- are attributes,
reported in `summary()` as xtabond2 reports them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from hdfe_stream.core.results_base import RegressionMulti, RegressionResult
from scipy import stats


@dataclass(kw_only=True)
class DPDResult(RegressionResult):
    """A fitted difference or system GMM model.

    Attributes
    ----------
    fml, depvar, coefnames, beta, vcov, vcov_type, df_t, n_obs, all_vcovs, \
        n_clusters, collin_vars, diagnostics, weights, weights_type
        As in `hdfe_stream.HDFEResult`. `n_obs` counts the observations of
        the levels equation in system GMM, of the transformed equation in
        difference GMM (as xtabond2). `vcov_type` is ``"iid"`` or
        ``"robust"`` (``"CRV1:<vars>"`` when clustered); for a two-step fit,
        robust means Windmeijer-corrected, and ``"iid"`` the uncorrected
        two-step variance.
    estimator : str
        ``"difference"`` or ``"system"``.
    steps : int or str
        1, 2 or ``"iterated"``.
    transform : str
        ``"fd"`` or ``"fod"``.
    n_groups : int
        Panels with observations.
    obs_per_group : dict
        ``min``, ``avg`` and ``max``.
    n_instruments : int
        Instruments, less any the generalized inverse drops (xtabond2's
        count).
    ar : list of dict
        Arellano-Bond tests for AR(1), AR(2), ... in the first-differenced
        residuals: ``order``, ``z``, ``p``.
    sargan : dict
        Sargan test of the overidentifying restrictions (``statistic``,
        ``df``, ``p``): not robust, from the one-step residuals.
    hansen : dict or None
        Hansen's J test (robust); None for a one-step non-robust fit.
    diff_hansen : list of dict
        Difference-in-Hansen (or -Sargan) tests of each instrument group's
        exogeneity: ``group``, the test of the model without it
        (``excluded``, ``excluded_df``, ``excluded_p``) and of the
        difference (``difference``, ``difference_df``, ``difference_p``).
    model_test : dict or None
        Wald test that all coefficients are zero: chi2, or F with ``small``.
    sigma2 : float
        xtabond2's sigma^2 (e(sig2)).
    mmsc : dict or None
        Andrews and Lu's (2001) model and moment selection criteria
        (``bic``, ``aic``, ``hqic``; lower is better), from Hansen's J.
    onestep : DPDResult or None
        For two-step and iterated fits, the one-step estimates (free).
    idtests : dict or None
        Identification tests (Windmeijer 2025; Stata's underid ``kp sw``),
        robust to the fit's clustering, with the regressors that are their
        own IV-style instruments (and any absorbed effect) partialled out:
        ``kp_lm``, Kleibergen and Paap's rk LM test of underidentification
        (rank of the first stage K - 1; rejecting means identified);
        ``kp_wald``, its Wald form, with ``F`` the weak-identification
        statistic with ivreg2's small-sample factor (no Stock-Yogo critical
        values apply with robust variances and many instruments);
        ``sanderson_windmeijer``, the conditional underidentification test
        of each endogenous regressor; ``partial_r2``, each one's first-stage
        partial R^2 and Shea's partial R^2; ``effective_f``, Montiel Olea and
        Pflueger's effective F with its 2SLS critical value (one endogenous
        regressor; else None); ``anderson_rubin``, the weak-instrument-robust
        test that the endogenous regressors' coefficients are 0 (from the
        moment conditions, with the reduced form of y), and with one of them
        its 95% confidence set (``confidence_set["intervals"]``). No
        endogeneity test: the lagged dependent variable is endogenous in
        the transformed equation by construction, and the other regressors'
        exogeneity is what the difference-in-Hansen test of their IV-style
        group checks. None with no endogenous regressor, or
        when skipped (see ``idtests=``). The algebra is
        `hdfe_stream.core.ivtests`.
    nl : dict or None
        With Ahn-Schmidt moments: their ``kind`` (``"noserial"`` or
        ``"iid"``), number (``moments``), the first step's ``weight`` on
        them, and whether the nonlinear step ``converged``.
    absorbed : dict or None
        The absorbed fixed effect: its ``term``, the number of ``effects``
        estimated (cells with observations), and the ``method``:
        ``"dummies"`` for time effects (``| year``), estimated as dummies and
        not reported, so every statistic is exact; ``"cells"`` for a
        period effect by group (``| g^year``, difference GMM), partialled out
        of the moments (see the user guide).
    pca : dict or None
        With ``pca=True``: the number of ``components``, the
        ``eigenvalues``, ``kmo`` and ``r2`` (the share of the variance kept).
    """

    estimator: str = "difference"
    steps: object = 2
    transform: str = "fd"
    n_groups: int = 0
    obs_per_group: dict = field(default_factory=dict)
    n_instruments: int = 0
    ar: list = field(default_factory=list)
    sargan: dict = field(default_factory=dict)
    hansen: dict | None = None
    diff_hansen: list = field(default_factory=list)
    model_test: dict | None = None
    sigma2: float = float("nan")
    mmsc: dict | None = None
    onestep: DPDResult | None = None
    pca: dict | None = None
    small: bool = False
    instrument_list: list = field(default_factory=list)
    time_effects: str | None = None
    absorbed: dict | None = None
    idtests: dict | None = None
    nl: dict | None = None
    _model: dict = field(default_factory=dict, repr=False)

    @property
    def _normal_inference(self):
        return not self.small

    def with_vcov(self, vcov) -> DPDResult:
        """A copy with the other variance computed at fit time:
        ``"iid"`` (for two-step: uncorrected) or ``"robust"`` (the fit's
        clustering, Windmeijer-corrected for two-step)."""
        key = str(vcov).lower() if not isinstance(vcov, dict) else None
        if key in ("uncorrected", "unadjusted"):
            key = "iid"
        if key in ("robust", "hetero", "hc1"):
            key = next((k for k in self.all_vcovs if k != "iid"), "robust")
        if isinstance(vcov, dict):
            (kind, var), = vcov.items()
            key = f"{kind.upper()}:{var.replace(' ', '')}"
        if key not in self.all_vcovs:
            raise KeyError(f"{vcov!r} was not computed; available: {list(self.all_vcovs)}. "
                           "The clustering changes a two-step estimate itself, so refit with "
                           "vcov= for another.")
        out = copy.copy(self)
        out.vcov, out.df_t = self.all_vcovs[key]
        out.vcov_type = key
        return out

    def ar_test(self, order: int = 2) -> dict:
        """The Arellano-Bond test for AR(`order`) in first differences."""
        for a in self.ar:
            if a["order"] == order:
                return a
        raise KeyError(f"AR({order}) was not computed; refit with artests={order}")

    def long_run(self, var: str, ldv: str | None = None) -> dict:
        """The long-run effect of `var`: the sum of its coefficients (the
        variable and its lags) over one minus the sum of the lagged
        dependent variable's, with a delta-method standard error.

        Parameters
        ----------
        var : str
            A regressor's base name: ``"x"`` collects ``x``, ``l(x, 1)``,
            ``l(x, 2)``, ...
        ldv : str, optional
            The dependent variable's name in its lags (default the outcome),
            collecting ``l(<ldv>, k)``.
        """
        ldv = ldv or self.depvar
        names = self.coefnames
        xi = [i for i, n in enumerate(names) if n == var or _is_lag_of(n, var)]
        yi = [i for i, n in enumerate(names) if _is_lag_of(n, ldv)]
        if not xi:
            raise KeyError(f"no coefficient on {var!r} or its lags")
        if not yi:
            raise KeyError(f"no lag of {ldv!r} among the regressors")
        b = self.beta
        num = b[xi].sum()
        den = 1 - b[yi].sum()
        est = num / den
        g = np.zeros(len(b))
        g[xi] = 1 / den
        g[yi] = num / den ** 2
        se = float(np.sqrt(g @ self.vcov @ g))
        z = est / se
        p = (2 * stats.norm.sf(abs(z)) if not self.small
             else 2 * stats.t.sf(abs(z), self.df_t))
        return dict(variable=var, estimate=float(est), se=se, z=float(z), p=float(p),
                    terms=[names[i] for i in xi], ldv_terms=[names[i] for i in yi])

    def predict(self, newdata=None) -> pl.LazyFrame:
        """The linear prediction x_it'b in levels (without the panel
        effect), as a lazy frame of the panel, time and ``fitted``, over the
        fit's data or `newdata` (which must have the columns the formula
        uses). Lags are taken by time value, as in the fit."""
        lf, fitted, _ = self._levels(newdata)
        m = self._model
        return lf.select(pl.col(m["panel"][0]), pl.col(m["panel"][1]), fitted.alias("fitted"))

    def resid(self, transform: str = "levels", newdata=None) -> pl.LazyFrame:
        """Residuals as a lazy frame of the panel, time and ``resid``:
        ``"levels"``, y - x'b (which includes the panel effect), or
        ``"fd"``, their first differences (the transformed equation's
        residuals in difference GMM)."""
        if transform not in ("levels", "fd"):
            raise ValueError("transform must be 'levels' or 'fd'")
        m = self._model
        pid, tv = m["panel"]
        lf, fitted, y = self._levels(newdata)
        out = lf.select(pl.col(pid), pl.col(tv), (y - fitted).alias("resid"))
        if transform == "fd":
            prev = out.select(pl.col(pid), (pl.col(tv) + m["tdelta"]).alias(tv),
                              pl.col("resid").alias("__prev"))
            out = (out.join(prev, on=[pid, tv], how="left", maintain_order="left")
                   .select(pl.col(pid), pl.col(tv), (pl.col("resid") - pl.col("__prev"))
                           .alias("resid")))
        return out

    def sample(self) -> pl.LazyFrame:
        """The estimation sample, as a lazy frame of the panel, time and
        whether the observation enters the ``transformed`` equation (its
        first difference, or its forward orthogonal deviation) and, in
        system GMM, the ``levels`` equation; one row per observation of the
        fit's data that enters either. Computed from the data as the fit
        selects its rows: the outcome, the regressors (with the lags they
        take by time) and the weight present, and the IV-style instruments
        (unless ``mz``) in the equation they instrument."""
        m = self._model
        pid, tv = m["panel"]
        terms = [m["yterm"], *m["terms"].values(), *[c[0] for c in m["iv_cols"]],
                 *m["complete_terms"]]
        lf, term = self._frame(None, terms)
        yx = [m["yterm"], *m["terms"].values()]

        def valid(e):
            return e.is_not_null() & e.is_not_nan()

        def present(ts):
            return pl.all_horizontal([valid(term(t)) for t in ts] or [pl.lit(True)])
        ok = present(yx)
        if m["weights"]:
            ok = ok & pl.col(m["weights"]).is_not_null() & (pl.col(m["weights"]) > 0)
        complete = present(yx + list(m["complete_terms"]))
        div = [(t, passthru) for t, eq, passthru, mz in m["iv_cols"] if eq in (1, 2) and not mz]
        lev = [t for t, eq, passthru, mz in m["iv_cols"] if eq in (0, 2) and not mz]
        cur = {f"__s{i}": valid(term(t)) for i, t in enumerate(yx)}
        cur.update({f"__d{i}": valid(term(t)) for i, (t, _) in enumerate(div)})
        base = lf.select(pl.col(pid), pl.col(tv), ok.alias("__ok"),
                         complete.alias("__complete"),
                         present(lev).alias("__lev"), **cur)
        step = m["tdelta"]
        if not m["orth"]:
            # first differences: the period before has the outcome, the
            # regressors and the differenced instruments
            prev = base.select(pl.col(pid), (pl.col(tv) + step).alias(tv),
                               *[pl.col(c).alias(c + "_p") for c in cur])
            f = base.join(prev, on=[pid, tv], how="left", maintain_order="left")
            need = [pl.col(f"__s{i}_p").fill_null(False) for i in range(len(yx))]
            for i, (_, passthru) in enumerate(div):
                need.append(pl.col(f"__d{i}") if passthru
                            else pl.col(f"__d{i}") & pl.col(f"__d{i}_p").fill_null(False))
            trans = pl.col("__ok") & pl.all_horizontal(need or [pl.lit(True)])
        else:
            # forward orthogonal deviations: a later complete observation;
            # a passed-through instrument is taken at the next period
            f = base.sort([pid, tv]).with_columns(
                (pl.col("__complete").cast(pl.Int64).reverse().cum_sum().reverse()
                 - pl.col("__complete").cast(pl.Int64)).over(pid).alias("__later"))
            nxt = base.select(pl.col(pid), (pl.col(tv) - step).alias(tv),
                              *[pl.col(f"__d{i}").alias(f"__d{i}_n")
                                for i, (_, p) in enumerate(div) if p])
            f = f.join(nxt, on=[pid, tv], how="left", maintain_order="left")
            need = [pl.col(f"__d{i}_n").fill_null(False) if passthru else pl.col(f"__d{i}")
                    for i, (_, passthru) in enumerate(div)]
            trans = (pl.col("__ok") & (pl.col("__later") > 0)
                     & pl.all_horizontal(need or [pl.lit(True)]))
        f = f.with_columns(trans.alias("transformed"))
        cols = [pl.col(pid), pl.col(tv), pl.col("transformed")]
        keep = pl.col("transformed")
        if m["system"]:
            levels = pl.col("__ok") & pl.col("__lev")
            cols.append(levels.alias("levels"))
            keep = keep | levels
        return f.filter(keep).select(cols)

    def _levels(self, newdata):
        m = self._model
        lf, term = self._frame(newdata, [m["yterm"], *m["terms"].values()])
        fitted = pl.lit(0.0)
        # every coefficient, the absorbed time effects' among them
        coefs = m.get("coef_all") or dict(zip(self.coefnames, self.beta))
        for name, b in coefs.items():
            fitted = fitted + term(m["terms"][name]) * float(b)
        return lf, fitted, term(m["yterm"])

    def _frame(self, newdata, terms):
        """The data (the fit's, or `newdata`) with the base columns of
        `terms` and their lags by time, and a function from a term to its
        expression."""
        if not self._model:
            raise ValueError("this result has no model information (built by hand?)")
        from hdfe_stream.core.utils import as_lazy
        m = self._model
        pid, tv = m["panel"]
        lf = as_lazy(m["source"] if newdata is None else newdata)
        lf = lf.filter(pl.col(pid).is_not_null() & pl.col(tv).is_not_null())
        names = {i: f"__dpd_b{i}" for i in range(len(m["bases"]))}
        lf = lf.with_columns([expr.cast(pl.Float64).alias(names[i])
                              for i, (_, expr) in enumerate(m["bases"])])
        needed = {(t.base, t.k + d) for t in terms if t.base >= 0
                  for d in ((0, 1) if t.diff else (0,))}
        cols = {}
        keep = [pid, tv] + list(names.values())
        base = lf.select(keep)
        for b, k in sorted(needed):
            if k == 0:
                cols[(b, k)] = pl.col(names[b])
                continue
            nm = f"__dpd_b{b}_l{k}"
            src = base.select(pl.col(pid), (pl.col(tv) + k * m["tdelta"]).alias(tv),
                              pl.col(names[b]).alias(nm))
            lf = lf.join(src, on=[pid, tv], how="left", maintain_order="left")
            cols[(b, k)] = pl.col(nm)

        def term(t):
            if t.base < 0:
                return pl.lit(1.0)
            if t.diff:
                return cols[(t.base, t.k)] - cols[(t.base, t.k + 1)]
            return cols[(t.base, t.k)]
        return lf, term

    def instruments(self):
        """The instrument columns (a list of names), as xtabond2 lists them."""
        return list(self.instrument_list)

    # ------------------------------------------------------------------ reporting
    def _summary_lines(self) -> list[str]:
        est = {"difference": "difference GMM", "system": "system GMM"}[self.estimator]
        step = {1: "one-step", 2: "two-step", "iterated": "iterated"}[self.steps]
        lines = [f"Dynamic panel-data estimation, {step} {est} (dpd_stream)   "
                 f"vcov: {self.vcov_type}"
                 + (" (Windmeijer-corrected)" if self.steps == 2 and self.vcov_type != "iid"
                    else "")]
        og = self.obs_per_group
        lines.append(f"observations: {self.n_obs:,}   groups: {self.n_groups:,}   "
                     f"obs per group: min {og.get('min')}, avg {og.get('avg', 0):.2f}, "
                     f"max {og.get('max')}   instruments: {self.n_instruments}")
        if self.transform == "fod":
            lines.append("transformation: forward orthogonal deviations")
        if self.nl:
            lines.append(f"Ahn-Schmidt nonlinear moments ({self.nl['kind']}): "
                         f"{self.nl['moments']}" + ("" if self.nl["converged"] else
                                                    "  (the nonlinear step did NOT converge)"))
        if self.absorbed:
            a = self.absorbed
            lines.append(f"absorbed fixed effect: {a['term']} ({a['effects']:,} effects"
                         + (", as dummies)" if a["method"] == "dummies" else ")"))
        if self.model_test:
            m = self.model_test
            lines.append("Wald test of all coefficients: "
                         + (f"F({m['df1']}, {m['df2']}) = {m['statistic']:.2f}"
                            if m["distribution"] == "F"
                            else f"chi2({m['df1']}) = {m['statistic']:.2f}")
                         + f", p = {m['pvalue']:.4f}")
        for a in self.ar:
            lines.append(f"Arellano-Bond test for AR({a['order']}) in first differences: "
                         f"z = {a['z']:.2f}, p = {a['p']:.4f}")
        s = self.sargan
        lines.append(f"Sargan test of overid. restrictions: chi2({s['df']}) = "
                     f"{s['statistic']:.2f}, p = {s['p']:.4f}  (not robust)")
        if self.hansen is not None:
            hh = self.hansen
            lines.append(f"Hansen test of overid. restrictions: chi2({hh['df']}) = "
                         f"{hh['statistic']:.2f}, p = {hh['p']:.4f}  (robust)")
        if self.diff_hansen:
            lines.append("Difference-in-" + ("Hansen" if self.hansen is not None else "Sargan")
                         + " tests of exogeneity of instrument subsets:")
            for d in self.diff_hansen:
                lines.append(f"  {d['group']}")
                lines.append(f"    excluding group: chi2({d['excluded_df']}) = "
                             f"{d['excluded']:.2f}, p = {d['excluded_p']:.4f}   "
                             f"difference: chi2({d['difference_df']}) = "
                             f"{d['difference']:.2f}, p = {d['difference_p']:.4f}")
        if self.idtests:
            it = self.idtests
            kp = it["kp_lm"]
            lines.append(f"Underidentification (Kleibergen-Paap rk LM, robust): chi2({kp['df']}) "
                         f"= {kp['statistic']:.2f}, p = {kp['p']:.4f}   "
                         f"weak identification (KP rk Wald F): {it['kp_wald']['F']:.2f}")
            for sw in it["sanderson_windmeijer"]:
                lines.append(f"  Sanderson-Windmeijer, {sw['variable']}: chi2({sw['df']}) = "
                             f"{sw['statistic']:.2f}, p = {sw['p']:.4f}")
            if it.get("partial_r2"):
                lines.append("  partial R2 / Shea partial R2: " + ", ".join(
                    f"{k}: {v['partial']:.4f} / {v['shea']:.4f}"
                    for k, v in it["partial_r2"].items()))
            ef = it.get("effective_f")
            if ef:
                lines.append(f"  effective F (Montiel Olea-Pflueger): {ef['F']:.2f}; 2SLS "
                             f"critical value (10% worst-case bias) {ef['critical_value']:.2f}")
            ar = it.get("anderson_rubin")
            if ar:
                line = (f"Anderson-Rubin test, endogenous coefficients = 0: chi2({ar['df']}) = "
                        f"{ar['statistic']:.2f}, p = {ar['p']:.4f}")
                cs = ar.get("confidence_set")
                if cs is not None:
                    ivs = cs["intervals"]
                    line += (f"; {cs['level']:.0%} set: "
                             + (" U ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in ivs)
                                if ivs else "empty"))
                lines.append(line)
        if self.pca:
            lines.append(f"GMM-style instruments replaced by {self.pca['components']} principal "
                         f"components (share of variance {self.pca['r2']:.3f}, KMO "
                         f"{self.pca['kmo']:.3f})")
        if self.weights:
            lines.append(f"weights: {self.weights} ({self.weights_type})")
        if self.collin_vars:
            lines.append(f"dropped as collinear: {self.collin_vars}")
        return lines

    def _summary_fields(self) -> dict:
        return {
            "formula": self.fml,
            "depvar": self.depvar,
            "estimator": self.estimator,
            "steps": self.steps,
            "transform": self.transform,
            "vcov": self.vcov_type,
            "small": self.small,
            "n_obs": self.n_obs,
            "n_groups": self.n_groups,
            "obs_per_group": dict(self.obs_per_group),
            "n_instruments": self.n_instruments,
            "n_clusters": dict(self.n_clusters),
            "model_test": self.model_test,
            "ar": list(self.ar),
            "sargan": self.sargan,
            "hansen": self.hansen,
            "diff_hansen": list(self.diff_hansen),
            "sigma2": self.sigma2,
            "mmsc": self.mmsc,
            "absorbed": self.absorbed,
            "nl": self.nl,
            "idtests": self.idtests,
            "pca": ({k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in self.pca.items()} if self.pca else None),
            "weights": self.weights,
            "weights_type": self.weights_type if self.weights else None,
            "collinear_dropped": list(self.collin_vars),
            "seconds": self.diagnostics.get("seconds"),
        }

    def _pyfixest_attrs(self) -> dict:
        return {
            "_fixef": None, "_has_fixef": False,
            "_r2": np.nan, "_adj_r2": np.nan, "_r2_adj": np.nan,
            "_r2_within": np.nan, "_adj_r2_within": np.nan, "_rmse": np.nan,
            "_method": "feols", "_is_iv": False, "_f_stat_1st_stage": None,
        }


def _is_lag_of(name, var):
    if not (name.startswith("l(") and name.endswith(")")):
        return False
    inner = name[2:-1]
    base, _, k = inner.rpartition(",")
    return base.strip() == var and k.strip().lstrip("-").isdigit()


class DPDMulti(RegressionMulti):
    """Several `DPDResult`s, e.g. from a loop over specifications; the
    reporting methods of `hdfe_stream.HDFEMulti`."""
