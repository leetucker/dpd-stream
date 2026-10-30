"""An absorbed period-specific fixed effect in difference GMM against the
same model with its cell dummies as regressors and IV-style instruments
(the oracle): the estimates, their variances (with Windmeijer's
correction), and the Sargan, Hansen, difference-in-Hansen and
Arellano-Bond tests are the same, with first differences or orthogonal
deviations, weights and any clustering."""

import _oracle
import numpy as np
import polars as pl
import pytest
from _oracle import GMM, IV, xtabond2
from conftest import lag_by_time

from dpd_stream import dpd_stream, inference
from dpd_stream.simulate import simulate_dynamic


def _cells(panel_data):
    g = (pl.col("id") % 4).alias("g")
    # a clustering that crosses the panels
    df = panel_data.with_columns(g, ((pl.col("id") * 7 + pl.col("year")) % 40).alias("cl2"))
    df = df.with_columns((pl.col("g").cast(pl.Utf8) + "_" + pl.col("year").cast(pl.Utf8))
                         .alias("cell"))
    cells = sorted(df["cell"].unique())
    return df.with_columns([(pl.col("cell") == c).cast(pl.Float64).alias(f"D{c}")
                            for c in cells]), [f"D{c}" for c in cells]


@pytest.fixture(autouse=True)
def exact_inverse(monkeypatch):
    """The cells and the dummies are the same model with an exact inverse of
    the moment covariance. Stata's invsym() drops negative pivots, which
    (with a covariance that is not positive semidefinite, as two-way
    clustering's) makes the models' generalized inverses differ."""
    monkeypatch.setattr(inference, "DROP_NEGATIVE_PIVOTS", False)
    monkeypatch.setattr(_oracle, "DROP_NEGATIVE_PIVOTS", False)


@pytest.fixture(scope="module")
def cell_data(panel_data):
    return _cells(panel_data)


@pytest.fixture(scope="module")
def cell_data_nogaps():
    """As `cell_data`, without gaps: with orthogonal deviations, the rows
    at a gap have no cluster here (see `kernels.panel._cluster_ok`) and do in the
    oracle, so clusterings by variables compare only without them."""
    return _cells(simulate_dynamic(150, 7, rho=(0.5,), unbalanced=True, p_gap=0.0,
                                   n_clusters=60, weights=True, heteroskedastic=True, seed=11))


CASES = {
    "1step iid": (dict(steps=1, vcov="iid"), dict()),
    "1step robust": (dict(steps=1), dict(robust=True)),
    "2step iid": (dict(vcov="iid"), dict(twostep=True)),
    "2step robust": (dict(), dict(twostep=True, robust=True)),
    "2step small": (dict(small=True), dict(twostep=True, robust=True, small=True)),
    "fod 1step iid": (dict(steps=1, vcov="iid", transform="fod"), dict(orthogonal=True)),
    "fod 1step robust": (dict(steps=1, transform="fod"), dict(robust=True, orthogonal=True)),
    "fod 2step": (dict(transform="fod"), dict(twostep=True, robust=True, orthogonal=True)),
    "aweights 1step iid": (dict(steps=1, vcov="iid", weights="wt"), dict(weights="wt")),
    "aweights 2step": (dict(weights="wt"), dict(twostep=True, robust=True, weights="wt")),
    "fod aweights 2step": (dict(weights="wt", transform="fod"),
                           dict(twostep=True, robust=True, weights="wt", orthogonal=True)),
    "nested 2step": (dict(vcov={"CRV1": "cluster"}),
                     dict(twostep=True, robust=True, cluster=["cluster"])),
    "two-way 2step": (dict(vcov={"CRV1": "cluster+cl2"}),
                      dict(twostep=True, robust=True, cluster=["cluster", "cl2"])),
    # by panel, by a variable crossing the cells' groups and (their
    # intersection) by observation
    "panel+cl2 2step": (dict(vcov={"CRV1": "id+cl2"}),
                        dict(twostep=True, robust=True, cluster=["id", "cl2"])),
    "panel+cl2 2step small": (dict(vcov={"CRV1": "id+cl2"}, small=True),
                              dict(twostep=True, robust=True, cluster=["id", "cl2"], small=True)),
    "two-way 1step": (dict(steps=1, vcov={"CRV1": "cluster+cl2"}),
                      dict(robust=True, cluster=["cluster", "cl2"])),
}

CASES_NOGAPS = {
    "fod two-way 2step": (dict(transform="fod", vcov={"CRV1": "cluster+cl2"}),
                          dict(twostep=True, robust=True, orthogonal=True,
                               cluster=["cluster", "cl2"])),
    "fod panel+cl2 2step small": (dict(transform="fod", vcov={"CRV1": "id+cl2"}, small=True),
                                  dict(twostep=True, robust=True, orthogonal=True,
                                       cluster=["id", "cl2"], small=True)),
    "fod two-way 1step": (dict(steps=1, transform="fod", vcov={"CRV1": "cluster+cl2"}),
                          dict(robust=True, orthogonal=True, cluster=["cluster", "cl2"])),
}


@pytest.mark.parametrize("case", list(CASES))
def test_absorbed_cells(cell_data, case):
    _check(*cell_data, case, *CASES[case])


@pytest.mark.parametrize("case", list(CASES_NOGAPS))
def test_absorbed_cells_fod_clusters(request, case):
    # (the panel + crossing cluster with `small` is ill conditioned in the
    # dummies model when there are gaps: its Hansen statistic is 6e-5 apart)
    data = request.getfixturevalue("cell_data_nogaps" if "panel" in case else "cell_data")
    _check(*data, case, *CASES_NOGAPS[case])


def _check(df, dummies, case, kw, okw):
    term = "g^year"
    r = dpd_stream(f"y ~ l(y, 1) + x + w | {term} | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) "
                   "+ iv(w)", df, panel=("id", "year"), verbose=False, **kw)
    data = {c: df[c].to_numpy().astype(float)
            for c in ["id", "year", "y", "x", "w", "wt", "cluster", "cl2", *dummies]}
    data["l(y, 1)"] = lag_by_time(df, "y", 1)
    o = xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w", *dummies],
                 [GMM(["y"], lags=(2, 4)), GMM(["x"], lags=(1, 3), collapse=True)],
                 [IV(["w"]), IV(dummies)], system=False, **okw)
    keep = [o.names.index(n) for n in ["l(y, 1)", "x", "w"]]
    np.testing.assert_allclose(r.beta, o.b[keep], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(r.vcov, o.V[np.ix_(keep, keep)], rtol=1e-6, atol=1e-12)
    if "2step" in case and not kw.get("small"):
        np.testing.assert_allclose(r.with_vcov("iid").vcov, o.V2[np.ix_(keep, keep)],
                                   rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(r.sargan["statistic"], o.sargan, rtol=1e-6)
    assert r.sargan["df"] == o.sargan_df
    if o.hansen is not None:
        np.testing.assert_allclose(r.hansen["statistic"], o.hansen, rtol=1e-6)
    np.testing.assert_allclose(r.sigma2, o.sig2, rtol=1e-6)
    assert r.n_obs == o.n_obs
    for d, od in zip(r.diff_hansen, [x for x in o.diffsargan if "D" not in x["name"]
                                     and "Y" not in x["name"]]):
        np.testing.assert_allclose(d["excluded"], od["sargan"], rtol=1e-5)
    assert len(r.ar) == len(o.ar)
    for a, (z, _) in zip(r.ar, o.ar):
        np.testing.assert_allclose(a["z"], z, rtol=1e-6)


def test_time_effects_difference(panel_data):
    """'| year' in difference GMM: absorbed, estimated as dummies that are
    each their own instrument, so everything matches the explicit dummies'
    fit, less the dummies' coefficients."""
    years = sorted(panel_data["year"].unique())
    d2 = panel_data.with_columns([(pl.col("year") == y).cast(pl.Float64).alias(f"yr{y}")
                                  for y in years])
    yr = " + ".join(f"yr{y}" for y in years)
    a = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + iv(w)", panel_data,
                   panel=("id", "year"), verbose=False)
    b = dpd_stream(f"y ~ l(y, 1) + x + w + {yr} | gmm(y, 2:4) + iv(w) + iv({yr})", d2,
                   panel=("id", "year"), verbose=False)
    assert a.coefnames == ["l(y, 1)", "x", "w"]
    np.testing.assert_allclose(a.beta, b.beta[:3], rtol=1e-10)
    np.testing.assert_allclose(a.se, b.se[:3], rtol=1e-10)
    np.testing.assert_allclose(a.hansen["statistic"], b.hansen["statistic"], rtol=1e-10)
    for x, y in zip(a.ar, b.ar):
        np.testing.assert_allclose(x["z"], y["z"], rtol=1e-10)
    assert a.absorbed == dict(term="year", effects=len(years) - 2, method="dummies")
    assert a.collin_vars == []
    pa = a.predict().collect()["fitted"].to_numpy()
    pb = b.predict().collect()["fitted"].to_numpy()
    np.testing.assert_allclose(pa, pb, rtol=1e-9, atol=1e-12)


def test_absorb_refusals(cell_data):
    df, _ = cell_data
    fml = "y ~ l(y, 1) + x + w | {} | gmm(y, 2:4) + iv(w)"
    with pytest.raises(ValueError, match="specific to a period"):
        dpd_stream(fml.format("g"), df, panel=("id", "year"), verbose=False)
    with pytest.raises(ValueError, match="constant within each panel"):
        dpd_stream(fml.format("x^year"), df, panel=("id", "year"), verbose=False)
    with pytest.raises(NotImplementedError, match="pca"):
        dpd_stream(fml.format("g^year"), df, panel=("id", "year"), pca=True, verbose=False)
    with pytest.raises(NotImplementedError, match="nonlinear in the absorbed effects"):
        dpd_stream(fml.format("g^year"), df, panel=("id", "year"), nl="noserial",
                   verbose=False)


@pytest.mark.parametrize("transform", ["fd", "fod"])
def test_absorbed_iterated(cell_data, transform):
    """Iterated GMM with absorbed cells: the explicit dummies' estimates."""
    df, dummies = cell_data
    base = "y ~ l(y, 1) + x + w{} | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w){}"
    a = dpd_stream(base.format(" | g^year", ""), df, panel=("id", "year"), steps="iterated",
                   transform=transform, verbose=False)
    yr = " + ".join(dummies)
    b = dpd_stream(base.format(f" + {yr}", f" + iv({yr})"), df, panel=("id", "year"),
                   steps="iterated", transform=transform, verbose=False)
    np.testing.assert_allclose(a.beta, b.beta[:3], rtol=1e-6)
    np.testing.assert_allclose(a.se, b.se[:3], rtol=1e-6)
    np.testing.assert_allclose(a.hansen["statistic"], b.hansen["statistic"], rtol=1e-6)
    for x, y in zip(a.ar, b.ar):
        np.testing.assert_allclose(x["z"], y["z"], rtol=1e-6)
