"""dpd-stream against the dense reference implementation (`_oracle`) on
unbalanced simulated panels with gaps: every option, every statistic."""

import numpy as np
import pytest
from _oracle import GMM, IV, xtabond2
from conftest import lag_by_time

from dpd_stream import Gmm, Iv, dpd_stream

TOL = 1e-8


def _oracle_data(df, extra=()):
    data = {c: df[c].to_numpy().astype(float) for c in df.columns}
    for v, k in [("y", 1), ("y", 2), ("x", 1), ("w", 1), *extra]:
        data[f"l({v}, {k})"] = lag_by_time(df, v, k)
    return data


def _to_oracle_terms(terms):
    return [t.replace("l(y,1)", "l(y, 1)") for t in terms]


CASES = {
    # name: (formula kwargs, oracle kwargs)
    "diff 2step": (dict(system=False, steps=2),
                   dict(system=False, twostep=True, robust=True)),
    "diff 1step robust": (dict(system=False, steps=1),
                          dict(system=False, robust=True)),
    "diff 1step iid": (dict(system=False, steps=1, vcov="iid"),
                       dict(system=False)),
    "diff 2step iid": (dict(system=False, steps=2, vcov="iid"),
                       dict(system=False, twostep=True)),
    "diff fod 2step": (dict(system=False, transform="fod"),
                       dict(system=False, twostep=True, robust=True, orthogonal=True)),
    "diff fod 1step iid": (dict(system=False, transform="fod", steps=1, vcov="iid"),
                           dict(system=False, orthogonal=True)),
    "sys 2step": (dict(system=True), dict(system=True, twostep=True, robust=True)),
    "sys 1step iid": (dict(system=True, steps=1, vcov="iid"), dict(system=True)),
    "sys 1step robust small": (dict(system=True, steps=1, small=True),
                               dict(system=True, robust=True, small=True)),
    "sys fod 2step": (dict(system=True, transform="fod"),
                      dict(system=True, twostep=True, robust=True, orthogonal=True)),
    "sys h1": (dict(system=True, h=1), dict(system=True, twostep=True, robust=True, h=1)),
    "sys h2 1step iid": (dict(system=True, h=2, steps=1, vcov="iid"), dict(system=True, h=2)),
    "sys h1 1step iid": (dict(system=True, h=1, steps=1, vcov="iid"), dict(system=True, h=1)),
    "diff small 2step": (dict(system=False, small=True),
                         dict(system=False, twostep=True, robust=True, small=True)),
    "sys nocons": (dict(system=True, constant=False),
                   dict(system=True, constant=False, twostep=True, robust=True)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_against_oracle(panel_data, case):
    kw, okw = CASES[case]
    df = panel_data
    fml = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"
    r = dpd_stream(fml, df, panel=("id", "year"), verbose=False, **kw)
    data = _oracle_data(df)
    o = xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w"],
                 [GMM(["y"], lags=(2, 4)), GMM(["x"], lags=(1, 3), collapse=True)],
                 [IV(["w"])], **okw)
    _compare(r, o)


def _compare(r, o, tol=TOL):
    assert r.coefnames == [n if n != "_cons" else "Intercept" for n in o.names]
    np.testing.assert_allclose(r.beta, o.b, rtol=tol, atol=tol)
    np.testing.assert_allclose(r.vcov, o.V, rtol=1e-6, atol=1e-10)
    assert r.n_obs == o.n_obs and r.n_groups == o.n_groups
    assert r.n_instruments == o.j0
    np.testing.assert_allclose(r.sargan["statistic"], o.sargan, rtol=1e-7)
    if o.hansen is None:
        assert r.hansen is None
    else:
        np.testing.assert_allclose(r.hansen["statistic"], o.hansen, rtol=1e-7)
    for a, (z, _) in zip(r.ar, o.ar):
        if np.isnan(z):
            assert np.isnan(a["z"])
        else:
            np.testing.assert_allclose(a["z"], z, rtol=1e-6)
    # the groups come in the same order (IV-style, GMM-style, levels)
    assert len(r.diff_hansen) == len(o.diffsargan)
    for d, od in zip(r.diff_hansen, o.diffsargan):
        assert ("levels" in d["group"]) == ("levels" in od["name"]), (d["group"], od["name"])
        np.testing.assert_allclose(d["excluded"], od["sargan"], rtol=1e-6, atol=1e-8)
        assert d["difference_df"] == od["n"]
    if o.chi2 is not None:
        np.testing.assert_allclose(r.model_test["statistic"], o.chi2, rtol=1e-6)
    if o.F is not None:
        np.testing.assert_allclose(r.model_test["statistic"], o.F, rtol=1e-6)
        assert r.model_test["df2"] == o.df_r


CLUSTER_CASES = {
    "nested": ("cluster", dict(system=False)),
    "nested sys": ("cluster", dict(system=True)),
    "by year": ("year", dict(system=False)),
    "two-way id+year": ("id+year", dict(system=False)),
    "two-way sys small": ("cluster+year", dict(system=True, small=True)),
    "panel id": ("id", dict(system=True)),
}


@pytest.mark.parametrize("case", list(CLUSTER_CASES))
def test_clusters(panel_data, case):
    var, kw = CLUSTER_CASES[case]
    df = panel_data
    fml = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"
    r = dpd_stream(fml, df, panel=("id", "year"), vcov={"CRV1": var}, verbose=False, **kw)
    data = _oracle_data(df)
    o = xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w"],
                 [GMM(["y"], lags=(2, 4)), GMM(["x"], lags=(1, 3), collapse=True)],
                 [IV(["w"])], twostep=True, robust=True, cluster=var.split("+"),
                 system=kw["system"], small=kw.get("small", False))
    _compare(r, o)


INSTRUMENT_CASES = {
    "uncollapsed all lags": (
        "y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)",
        [GMM(["y"], lags=(2, None)), GMM(["x"], lags=(1, None))], [IV(["w"])], {}),
    "two vars one group": (
        "y ~ l(y, 1) + x + w | gmm(y + x, 2:3) + iv(w)",
        [GMM(["y", "x"], lags=(2, 3))], [IV(["w"])], {}),
    "eq level iv, passthru": (
        "y ~ l(y, 1) + x + w | gmm(y, 2:3) + gmm(x, 1:2, eq='diff') + iv(w, eq='level') "
        "+ iv(l(w, 1), passthru=True, eq='diff')",
        [GMM(["y"], lags=(2, 3)), GMM(["x"], lags=(1, 2), eq="diff")],
        [IV(["w"], eq="level"), IV(["l(w, 1)"], passthru=True, eq="diff")], {}),
    "gmm eq level": (
        "y ~ l(y, 1) + x + w | gmm(y, 2:3, eq='diff') + gmm(x, 1:1, eq='level') + iv(w)",
        [GMM(["y"], lags=(2, 3), eq="diff"), GMM(["x"], lags=(1, 1), eq="level")],
        [IV(["w"])], {}),
    "split": (
        "y ~ l(y, 1) + x + w | gmm(y, 2:3, split=True) + gmm(x, 1:2, collapse=True) + iv(w)",
        [GMM(["y"], lags=(2, 3), split=True), GMM(["x"], lags=(1, 2), collapse=True)],
        [IV(["w"])], {}),
    "mz": (
        "y ~ l(y, 1) + x | gmm(y, 2:3) + iv(l(w, 1), mz=True)",
        [GMM(["y"], lags=(2, 3))], [IV(["l(w, 1)"], mz=True)], {}),
    "bod": (
        "y ~ l(y, 1) + x + w | gmm(y, 1:2, orthogonal=True, eq='diff') + iv(w)",
        [GMM(["y"], lags=(1, 2), orthogonal=True, eq="diff")], [IV(["w"])],
        dict(orthogonal=True)),
    "two lags of y": (
        "y ~ l(y, 1:2) + x + w | gmm(y, 2:4) + iv(w)",
        [GMM(["y"], lags=(2, 4))], [IV(["w"])], {}),
}


@pytest.mark.parametrize("case", list(INSTRUMENT_CASES))
@pytest.mark.parametrize("system", [False, True])
def test_instruments(panel_data, case, system):
    fml, ogmm, oiv, extra = INSTRUMENT_CASES[case]
    df = panel_data
    xs = ["l(y, 1)", "l(y, 2)", "x", "w"] if "l(y, 1:2)" in fml else ["l(y, 1)", "x"]
    if "+ w |" in fml:
        xs = [*xs, "w"] if "w" not in xs else xs
    kw = dict(transform="fod") if extra.get("orthogonal") else {}
    r = dpd_stream(fml, df, panel=("id", "year"), system=system, verbose=False, **kw)
    data = _oracle_data(df)
    o = xtabond2(data["id"], data["year"], data, "y", xs, ogmm, oiv, system=system,
                 twostep=True, robust=True, **extra)
    _compare(r, o)


def test_objects_equal_formula(panel_data):
    a = dpd_stream("y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
                   panel_data, panel=("id", "year"), verbose=False)
    b = dpd_stream("y ~ l(y, 1) + x + w", panel_data, panel=("id", "year"), verbose=False,
                   gmm=[Gmm("y", lags=(2, 4)), Gmm("x", lags=(1, 3), collapse=True)],
                   iv=[Iv("w")])
    np.testing.assert_array_equal(a.beta, b.beta)
    np.testing.assert_array_equal(a.vcov, b.vcov)


@pytest.mark.parametrize("wtype", ["aweights", "pweights"])
@pytest.mark.parametrize("system", [False, True])
def test_weights(panel_data, wtype, system):
    fml = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"
    for steps, okw in [(2, dict(twostep=True, robust=True)), (1, {})]:
        vcov = "robust" if steps == 2 else "iid"
        if wtype == "pweights" and steps == 1:
            okw = dict(robust=True)
        r = dpd_stream(fml, panel_data, panel=("id", "year"), system=system, steps=steps,
                       vcov=vcov, weights="wt", weights_type=wtype, verbose=False)
        data = _oracle_data(panel_data)
        o = xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w"],
                     [GMM(["y"], lags=(2, 4)), GMM(["x"], lags=(1, 3), collapse=True)],
                     [IV(["w"])], system=system, weights="wt", wtype=wtype[0], **okw)
        _compare(r, o)


@pytest.mark.parametrize("system", [False, True])
def test_pca(panel_data, system):
    fml = "y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)"
    r = dpd_stream(fml, panel_data, panel=("id", "year"), system=system, pca=True,
                   verbose=False)
    data = _oracle_data(panel_data)
    o = xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w"],
                 [GMM(["y"], lags=(2, None)), GMM(["x"], lags=(1, None))], [IV(["w"])],
                 system=system, twostep=True, robust=True, pca=True)
    assert r.pca["components"] == o.pca_components
    np.testing.assert_allclose(r.beta, o.b, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(r.se, np.sqrt(np.diag(o.V)), rtol=1e-6)
    np.testing.assert_allclose(r.hansen["statistic"], o.hansen, rtol=1e-6)
    np.testing.assert_allclose(r.pca["kmo"], o.kmo, rtol=1e-8)
