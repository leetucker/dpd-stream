"""Options beyond the oracle's: time effects, iterated GMM, long-run
effects, MMSC, residuals and predictions, reporting."""

import json

import numpy as np
import polars as pl
import pytest
from _oracle import GMM, IV, xtabond2
from conftest import lag_by_time

from dpd_stream import DPDResult, dpd_stream, etable
from dpd_stream.simulate import simulate_dynamic, truth_dynamic

FML = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"


def test_time_effects_system(panel_data):
    """'| year' in system GMM: absorbed, as dummies for every year, each an
    IV-style instrument (xtabond2's yr* in the regressors and iv(yr*)), that
    are not reported."""
    df = panel_data
    years = sorted(df["year"].unique())
    d2 = df.with_columns([(pl.col("year") == y).cast(pl.Float64).alias(f"yr{y}") for y in years])
    yr = " + ".join(f"yr{y}" for y in years)
    a = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + iv(w)", df,
                   panel=("id", "year"), system=True, verbose=False)
    b = dpd_stream(f"y ~ l(y, 1) + x + w + {yr} | gmm(y, 2:4) + iv(w) + iv({yr})", d2,
                   panel=("id", "year"), system=True, verbose=False)
    keep = [b.coefnames.index(n) for n in a.coefnames]
    assert a.coefnames == ["l(y, 1)", "x", "w", "Intercept"]
    np.testing.assert_allclose(a.beta, b.beta[keep], rtol=1e-10)
    np.testing.assert_allclose(a.se, b.se[keep], rtol=1e-10)
    assert a.absorbed["method"] == "dummies"


def test_fweights(panel_data):
    df = panel_data.with_columns((pl.col("wt") * 4).round(0).alias("fw"))
    data = {c: df[c].to_numpy().astype(float) for c in df.columns}
    data["l(y, 1)"] = lag_by_time(df, "y", 1)
    for system in (False, True):
        for steps, okw in [(2, dict(twostep=True, robust=True)), (1, {})]:
            r = dpd_stream(FML, df, panel=("id", "year"), system=system, steps=steps,
                           vcov="robust" if steps == 2 else "iid", weights="fw",
                           weights_type="fweights", verbose=False)
            o = xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w"],
                         [GMM(["y"], lags=(2, 4)), GMM(["x"], lags=(1, 3), collapse=True)],
                         [IV(["w"])], system=system, weights="fw", wtype="f", **okw)
            np.testing.assert_allclose(r.beta, o.b, rtol=1e-8)
            np.testing.assert_allclose(r.vcov, o.V, rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(r.sargan["statistic"], o.sargan, rtol=1e-7)
            assert r.n_obs == o.n_obs


def test_fweights_must_be_constant(panel_data):
    df = panel_data.with_columns(pl.int_range(pl.len()).alias("fw"))
    with pytest.raises(ValueError, match="constant within each panel"):
        dpd_stream(FML, df, panel=("id", "year"), weights="fw", weights_type="fweights",
                   verbose=False)


def test_iterated(panel_data):
    two = dpd_stream(FML, panel_data, panel=("id", "year"), verbose=False)
    it0 = dpd_stream(FML, panel_data, panel=("id", "year"), steps="iterated", maxiter=0,
                     verbose=False)
    np.testing.assert_allclose(it0.beta, two.beta, rtol=1e-12)
    it = dpd_stream(FML, panel_data, panel=("id", "year"), steps="iterated", verbose=False)
    assert it.diagnostics["iterations"] > 2
    assert np.all(np.abs(it.beta - two.beta) < 3 * two.se)
    # at convergence one more step changes nothing
    more = dpd_stream(FML, panel_data, panel=("id", "year"), steps="iterated", tol=1e-13,
                      verbose=False)
    np.testing.assert_allclose(more.beta, it.beta, atol=1e-7)


def test_long_run(panel_data):
    r = dpd_stream("y ~ l(y, 1:2) + x + l(x, 1) + w | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)",
                   panel_data, panel=("id", "year"), verbose=False)
    lr = r.long_run("x")
    b = dict(zip(r.coefnames, r.beta))
    est = (b["x"] + b["l(x, 1)"]) / (1 - b["l(y, 1)"] - b["l(y, 2)"])
    assert lr["estimate"] == pytest.approx(est)
    assert lr["terms"] == ["x", "l(x, 1)"] and lr["ldv_terms"] == ["l(y, 1)", "l(y, 2)"]
    # delta method against a numerical gradient
    def f(beta):
        d = dict(zip(r.coefnames, beta))
        return (d["x"] + d["l(x, 1)"]) / (1 - d["l(y, 1)"] - d["l(y, 2)"])
    g = np.array([(f(r.beta + h) - f(r.beta - h)) / 2e-6
                  for h in np.eye(len(r.beta)) * 1e-6])
    assert lr["se"] == pytest.approx(np.sqrt(g @ r.vcov @ g), rel=1e-6)


def test_mmsc(panel_data):
    r = dpd_stream(FML, panel_data, panel=("id", "year"), verbose=False)
    over = r.n_instruments - len(r.beta)
    J = r.hansen["statistic"]
    assert r.mmsc["bic"] == pytest.approx(J - over * np.log(r.n_obs))
    assert r.mmsc["aic"] == pytest.approx(J - 2 * over)


def test_resid_predict(panel_data):
    r = dpd_stream(FML, panel_data, panel=("id", "year"), verbose=False)
    ly = lag_by_time(panel_data, "y", 1)
    b = dict(zip(r.coefnames, r.beta))
    col = {c: panel_data[c].to_numpy() for c in ("y", "x", "w")}
    want = col["y"] - (b["l(y, 1)"] * ly + b["x"] * col["x"] + b["w"] * col["w"])
    got = r.resid().collect()
    assert got.columns == ["id", "year", "resid"]
    np.testing.assert_allclose(got["resid"].to_numpy(), want, rtol=1e-12)
    fd = r.resid("fd").collect()
    lagged = lag_by_time(got.rename({"resid": "e"}), "e", 1)
    np.testing.assert_allclose(fd["resid"].to_numpy(), got["resid"].to_numpy() - lagged,
                               rtol=1e-12)
    pr = r.predict().collect()
    np.testing.assert_allclose(pr["fitted"].to_numpy() + got["resid"].to_numpy(),
                               np.where(np.isnan(want), np.nan, col["y"]), rtol=1e-12)


def test_reporting(panel_data, tmp_path):
    r = dpd_stream(FML, panel_data, panel=("id", "year"), system=True, verbose=False)
    assert isinstance(r, DPDResult)
    txt = r.summary_text()
    for s in ("two-step system GMM", "Arellano-Bond test for AR(2)", "Hansen test",
              "Difference-in-Hansen", "GMM instruments for levels", "Windmeijer"):
        assert s in txt, s
    d = json.loads(r.summary_json(tmp_path / "s.json"))
    assert d["estimator"] == "system" and len(d["ar"]) == 2
    assert r.tidy().height == len(r.beta)
    assert r.confint().height == len(r.beta)
    w = r.wald_test(keep="x")
    assert w["distribution"] == "chi2"
    t = etable([r, r.onestep])
    assert t is not None
    u = r.with_vcov("iid")
    assert u.vcov_type == "iid" and not np.allclose(u.se, r.se)
    assert r.ar_test(2)["order"] == 2
    assert len(r.instruments()) == r.diagnostics["j_columns"] or r.instruments()


def test_onestep_attached(panel_data):
    r = dpd_stream(FML, panel_data, panel=("id", "year"), verbose=False)
    one = dpd_stream(FML, panel_data, panel=("id", "year"), steps=1, verbose=False)
    np.testing.assert_allclose(r.onestep.beta, one.beta, rtol=1e-12)
    np.testing.assert_allclose(r.onestep.se, one.se, rtol=1e-10)


def test_recovers_truth():
    """A large sample recovers the simulation's coefficients (w is correlated
    with the fixed effect, so it instruments the transformed equation only)."""
    df = simulate_dynamic(20_000, 7, rho=(0.6,), seed=3)
    r = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4, collapse=True) "
                   "+ gmm(x, 1:3, collapse=True) + iv(w, eq='diff')", df, panel=("id", "year"),
                   system=True, verbose=False)
    truth = truth_dynamic(rho=(0.6,))
    b = dict(zip(r.coefnames, r.beta))
    se = dict(zip(r.coefnames, r.se))
    for k, v in truth.items():
        assert abs(b[k] - v) < 4 * se[k], (k, b[k], v, se[k])


def test_levels_instrument_moves_to_deeper_lag(panel_data):
    """xtabond2's search: a collapsed levels instrument (the difference at lag
    a - 1) that is zero throughout the sample moves to deeper lags. xs changes
    only between the first two years, so its difference at lag 0 is zero in
    every levels row with two lags of y, and the instrument becomes the
    difference at lag 1 -- the same model as that difference given as an
    IV-style levels instrument."""
    t0 = panel_data["year"].min()
    df = panel_data.with_columns(
        xs=pl.when(pl.col("year") > t0).then(pl.col("id") % 3 + 1.0).otherwise(0.0))
    df = df.with_columns(dxs1=lag_by_time(df, "xs", 1) - lag_by_time(df, "xs", 2))
    a = dpd_stream("y ~ l(y, 1:2) + x + w | gmm(y, 2:4) + gmm(xs, 1:3, collapse=True) + iv(w)",
                   df, panel=("id", "year"), system=True, verbose=False)
    b = dpd_stream("y ~ l(y, 1:2) + x + w | gmm(y, 2:4) + gmm(xs, 1:3, collapse=True, eq='diff') "
                   "+ iv(w) + iv(dxs1, eq='level', mz=True)", df.with_columns(
                       pl.col("dxs1").fill_nan(None)), panel=("id", "year"), system=True,
                   verbose=False)
    assert any("deeper lags" in n for n in a.diagnostics["notes"])
    assert a.n_instruments == b.n_instruments
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9)
    np.testing.assert_allclose(a.hansen["statistic"], b.hansen["statistic"], rtol=1e-9)


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("transform", ["fd", "fod"])
def test_sample(panel_data, system, transform):
    """sample(): its rows in each equation are the fit's observations."""
    df = panel_data.with_columns(
        z=pl.when(pl.col("id") % 5 == 0).then(None).otherwise(pl.col("w") ** 2))
    fml = ("y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w) "
           "+ iv(l(x, 2), mz=True) + iv(z, eq='level')" if system else
           "y ~ l(y, 1) + x + w | gmm(y, 2:4) + iv(w) + iv(l(x, 1), passthru=True) + iv(z)")
    r = dpd_stream(fml, df, panel=("id", "year"), system=system, transform=transform,
                   weights="wt", verbose=False)
    s = r.sample().collect()
    if system:
        assert s["levels"].sum() == r.n_obs
        assert s["transformed"].sum() == r.diagnostics["wttot"]
    else:
        assert s["transformed"].sum() == r.n_obs
    assert s.select("id", "year").is_unique().all()
