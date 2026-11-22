"""DPDResult.refit_bootstrap: the draws, the resampling unit, recentering, the
refused cases and the failure ledger. (Exactness of a recentered refit against
the dense reference is in test_recenter.py.)"""

import warnings

import numpy as np
import polars as pl
import pytest

from dpd_stream import BootstrapFailureWarning, DPDBootstrapResult, dpd_stream
from dpd_stream.bootstrap import _refit_task
from dpd_stream.estimator import DPD
from dpd_stream.simulate import simulate_dynamic

FML = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"
PANEL = ("id", "year")


@pytest.fixture(scope="module")
def data():
    return simulate_dynamic(120, 6, n_clusters=30, weights=True, time_effects=False, seed=3)


def _fit(df, **kw):
    return dpd_stream(FML, df, panel=PANEL, verbose=False, **kw)


def test_draws_are_reproducible_and_the_same_in_workers(data):
    fit = _fit(data, system=True)
    a = fit.refit_bootstrap(reps=6, seed=4)
    b = fit.refit_bootstrap(reps=6, seed=4)
    c = fit.refit_bootstrap(reps=6, seed=4, n_workers=2, n_threads=2)
    assert isinstance(a, DPDBootstrapResult)
    np.testing.assert_allclose(b._draws, a._draws, rtol=1e-13)
    np.testing.assert_allclose(c._draws, a._draws, rtol=1e-12)
    np.testing.assert_allclose(c._se_draws, a._se_draws, rtol=1e-10)
    np.testing.assert_allclose(c._hansen_draws, a._hansen_draws, rtol=1e-10)
    assert a.diagnostics["recentered"] and a.diagnostics["moment_shift_units"] == 120
    assert a.coefnames == fit.coefnames
    np.testing.assert_allclose(a.beta, fit.beta)
    np.testing.assert_allclose(a.original_se, np.sqrt(np.diag(fit.vcov)))
    # the draws scatter around the estimate
    assert np.all(np.abs(a._draws.mean(0) - fit.beta) < 4 * a._draws.std(0) + 1e-12)
    h = a.hansen()
    assert h["statistic"] == fit.hansen["statistic"] and 0 < h["bootstrap_p"] <= 1
    for method in ("percentile", "basic", "normal", "studentized", "symmetric"):
        tab = a.tidy(method=method)
        assert (tab["Lower"] < tab["Upper"]).all()
    assert a.summary_dict()["hansen"]["successful_reps"] == 6


def test_one_refit_is_a_plain_fit_of_the_draw(data, tmp_path):
    """A refit is DPD on the written draw, with the shift, the original's
    columns and each copy's IDs (and clusters) of its own."""
    fit = _fit(data, system=True, vcov={"CRV1": "cluster"})
    m = fit._model
    # a draw of clusters as core writes it: the copy's panels renumbered, the
    # copy in its own column
    rng = np.random.default_rng(0)
    clusters = np.sort(data["cluster"].unique().to_numpy())
    parts = []
    for copy, g in enumerate(rng.integers(0, len(clusters), len(clusters))):
        part = data.filter(pl.col("cluster") == clusters[g])
        parts.append(part.with_columns((pl.col("id") + copy * 10_000).alias("unit"),
                                       pl.lit(copy).alias("grp")))
    draw = pl.concat(parts)
    path = tmp_path / "draw.parquet"
    draw.write_parquet(path)
    shift = (m["moment_sum"], len(clusters))
    task = dict(fml=m["fml"], time="year", args=m["args"], tdelta=m["tdelta"],
                expected_dates=sorted(draw["year"].unique().to_list()),
                dates=sorted(draw["year"].unique().to_list()), cluster="cluster",
                panel_cluster=False, draw_se=True, moment_shift=shift,
                live_mask=m["live_mask"], xsh=m["xsh"], keep_names=m["keep_names"],
                index=np.arange(len(fit.coefnames)), hansen=True, path=path,
                unit_column="unit", group_column="grp", workdir=tmp_path / "fit",
                threads=1, memory_gb=None)
    values = _refit_task(task)
    est = DPD(FML, draw, ("unit", "year"), system=True, vcov={"CRV1": "grp"},
              verbose=False, moment_shift=shift, fixed_columns=m["live_mask"],
              fixed_xsh=m["xsh"])
    f = est.fit()
    k = len(fit.coefnames)
    np.testing.assert_allclose(values[:k], f.b, rtol=1e-10)
    np.testing.assert_allclose(values[k:2 * k], np.sqrt(np.diag(f.V)), rtol=1e-8)
    np.testing.assert_allclose(values[-1], f.info["hansen"], rtol=1e-10)
    # missing a period: a failed draw
    with pytest.raises(ValueError, match="periods"):
        _refit_task(dict(task, dates=task["dates"][1:]))


def test_clusters_are_resampled_whole(data):
    fit = _fit(data, vcov={"CRV1": "cluster"})
    boot = fit.refit_bootstrap(reps=4, seed=1)
    assert boot.diagnostics["resampled"] == "clusters (cluster)"
    assert boot.diagnostics["clusters"] == 30 and boot.diagnostics["moment_shift_units"] == 30
    assert np.isfinite(boot._draws).all()
    # clustered by panel: panels are drawn, each copy its own cluster
    by_panel = _fit(data, vcov={"CRV1": "id"}).refit_bootstrap(reps=4, seed=1)
    assert by_panel.diagnostics["resampled"] == "units"


def test_subset_and_options(data):
    fit = _fit(data, system=True, steps="iterated")
    boot = fit.refit_bootstrap(reps=4, seed=2, coefficients=["x", "l(y, 1)"],
                               draw_standard_errors=False)
    assert boot.coefnames == ["x", "l(y, 1)"] and boot._draws.shape == (4, 2)
    assert boot._se_draws is None
    with pytest.raises(ValueError, match="standard errors"):
        boot.confint(method="studentized")
    boot.confint(method="percentile")
    for bad in (["nope"], ["x", "x"], [], "x"):
        with pytest.raises(ValueError, match="coefficient"):
            fit.refit_bootstrap(reps=4, coefficients=bad)
    with pytest.raises(ValueError, match="recenter"):
        fit.refit_bootstrap(reps=4, recenter="yes")
    plain = fit.refit_bootstrap(reps=4, seed=2, recenter=False)
    assert not plain.diagnostics["recentered"]
    assert plain.diagnostics["recenter_reason"] == "recenter=False"
    with pytest.raises(ValueError, match="Hansen"):
        plain.hansen()


def test_refused(data):
    with pytest.raises(ValueError, match="multi-way"):
        _fit(data, vcov={"CRV1": "id+year"}).refit_bootstrap(reps=4)
    with pytest.raises(ValueError, match="varies within"):
        _fit(data, vcov={"CRV1": "year"}).refit_bootstrap(reps=4)
    with pytest.raises(ValueError, match="onestep"):
        _fit(data).onestep.refit_bootstrap(reps=4)
    weighted = _fit(data, weights="wt")
    with pytest.raises(ValueError, match="weights"):
        weighted.refit_bootstrap(reps=4, recenter=True)
    boot = weighted.refit_bootstrap(reps=4, seed=0)
    assert not boot.diagnostics["recentered"]
    assert "weights" in boot.diagnostics["recenter_reason"]
    assert boot._hansen_draws is None and boot._se_draws is not None


def test_failed_draws_are_recorded_not_redrawn(data):
    # one panel has a period no other has: most draws lose it
    extra = data.filter(pl.col("id") == data["id"][0]).head(1).with_columns(
        pl.lit(1999, dtype=data["year"].dtype).alias("year"))
    df = pl.concat([extra, data])
    fit = _fit(df)
    with pytest.warns(BootstrapFailureWarning):
        boot = fit.refit_bootstrap(reps=6, seed=0)
    fails = boot.failures()
    assert fails.height >= 1 and fails.height + boot.successful_reps == 6
    assert all("periods" in r for r in fails["reason"])
    with pytest.raises(ValueError, match="allow_incomplete"):
        boot.tidy()
    if boot.successful_reps >= 2:
        boot.tidy(allow_incomplete=True)


def test_sources(data, tmp_path):
    """A Parquet file (or glob) gives the draws an in-memory frame gives, and a
    source changed since the fit is refused."""
    for i, part in enumerate(data.partition_by("cluster")):
        part.write_parquet(tmp_path / f"part{i:02d}.parquet")
    glob = str(tmp_path / "part*.parquet")
    a = _fit(data).refit_bootstrap(reps=4, seed=9)
    b = _fit(glob).refit_bootstrap(reps=4, seed=9)
    np.testing.assert_allclose(b.beta, a.beta, rtol=1e-12)
    np.testing.assert_allclose(b._draws, a._draws, rtol=1e-10)
    path = tmp_path / "one.parquet"
    data.write_parquet(path)
    fit = _fit(path)
    data.filter(pl.col("id") != data["id"][0]).write_parquet(path)
    with pytest.raises(RuntimeError, match="differs"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit.refit_bootstrap(reps=4)
