"""The identification tests against a dense computation from the oracle's
instrument matrices (the partialling, the cluster-robust covariances and
the auxiliary regressions; ranktest's rk algebra is `hdfe_stream.core.ivtests.kp_rk`)."""

import numpy as np
import polars as pl
import pytest
from _oracle import GMM, IV, xtabond2
from conftest import lag_by_time
from hdfe_stream.core.ivtests import kp_rk
from scipy import stats

from dpd_stream import dpd_stream
from dpd_stream.inference import invsym


def _dense(o, exog_cols, endog):
    """Partial the included exogenous regressors out of the excluded
    instruments and the endogenous regressors over the stacked rows."""
    d = o.dense
    Z = d["Z"].reshape(-1, d["Z"].shape[-1])
    X = d["X"].reshape(-1, d["X"].shape[-1])
    N, R = d["Z"].shape[:2]
    keepz = [c for c in range(Z.shape[1]) if c not in exog_cols and np.any(Z[:, c])]
    X2 = Z[:, exog_cols]
    Z1, Y = Z[:, keepz], X[:, endog]
    if X2.shape[1]:
        B = np.linalg.lstsq(X2, np.column_stack([Z1, Y]), rcond=None)[0]
        R_ = np.column_stack([Z1, Y]) - X2 @ B
        Z1, Y = R_[:, :Z1.shape[1]], R_[:, Z1.shape[1]:]
    return Z1, Y, N, R


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("two_endog", [False, True])
def test_idtests(panel_data, system, two_endog):
    df = panel_data
    fml = ("y ~ l(y, 1) + x + w | gmm(y, 2:4, collapse=True) + gmm(x, 1:3, collapse=True) "
           "+ iv(w)") if two_endog else \
        "y ~ l(y, 1) + w | gmm(y, 2:4, collapse=True) + iv(w)"
    r = dpd_stream(fml, df, panel=("id", "year"), system=system, verbose=False)
    it = r.idtests
    data = {c: df[c].to_numpy().astype(float) for c in df.columns}
    data["l(y, 1)"] = lag_by_time(df, "y", 1)
    xs = ["l(y, 1)", "x", "w"] if two_endog else ["l(y, 1)", "w"]
    gm = [GMM(["y"], lags=(2, 4), collapse=True)]
    if two_endog:
        gm.append(GMM(["x"], lags=(1, 3), collapse=True))
    o = xtabond2(data["id"], data["year"], data, "y", xs, gm, [IV(["w"])], system=system,
                 twostep=True, robust=True)
    # w (and the constant) are their own instruments: IV columns 0 (and 1)
    exog = [0, 1] if system else [0]
    endog = [0, 1] if two_endog else [0]
    assert it["endogenous"] == [xs[e] for e in endog]
    Z1, Y, N, R = _dense(o, exog, endog)
    L1, Ke = Z1.shape[1], Y.shape[1]
    assert it["excluded_instruments"] == L1
    Pi = np.linalg.lstsq(Z1, Y, rcond=None)[0]
    V = Y - Z1 @ Pi
    pid = np.repeat(np.arange(N), R)

    def omega(E):
        U = np.zeros((N, L1 * Ke))
        for k in range(Ke):
            np.add.at(U[:, k * L1:(k + 1) * L1], pid, Z1 * E[:, [k]])
        return U.T @ U
    lm, df_ = kp_rk(Z1.T @ Z1, Pi, Y.T @ Y, omega(Y))
    wald, _ = kp_rk(Z1.T @ Z1, Pi, Y.T @ Y, omega(V))
    np.testing.assert_allclose(it["kp_lm"]["statistic"], lm, rtol=1e-6)
    np.testing.assert_allclose(it["kp_wald"]["statistic"], wald, rtol=1e-6)
    assert it["kp_lm"]["df"] == df_ == L1 - Ke + 1
    # Shea's partial R^2 (Godfrey): [(Y'Y)^-1]_kk / [(Yh'Yh)^-1]_kk
    Yh = Z1 @ Pi
    shea = np.diag(np.linalg.inv(Y.T @ Y)) / np.diag(np.linalg.inv(Yh.T @ Yh))
    np.testing.assert_allclose([it["partial_r2"][xs[e]]["shea"] for e in endog], shea,
                               rtol=1e-6)
    # Anderson-Rubin: the reduced form of y (net of the exogenous columns)
    Zall = o.dense["Z"].reshape(-1, o.dense["Z"].shape[-1])
    yv = o.dense["Y"].reshape(-1)
    yv = yv - Zall[:, exog] @ np.linalg.lstsq(Zall[:, exog], yv, rcond=None)[0]
    iZZ = np.linalg.inv(Z1.T @ Z1)

    def ar(b0):
        u = yv - Y @ np.atleast_1d(b0)
        p = iZZ @ Z1.T @ u
        g = np.zeros((N, L1))
        np.add.at(g, pid, Z1 * (u - Z1 @ p)[:, None])
        Vp = iZZ @ (g.T @ g) @ iZZ
        return p @ np.linalg.solve(Vp, p)
    ab = it["anderson_rubin"]
    np.testing.assert_allclose(ab["statistic"], ar(np.zeros(Ke)), rtol=1e-6)
    if Ke == 1:
        assert it["effective_f"] is not None
        for lo, hi in ab["confidence_set"]["intervals"]:
            for end in (lo, hi):
                if np.isfinite(end):
                    np.testing.assert_allclose(ar(end), stats.chi2.ppf(0.95, L1), rtol=1e-6)
    else:
        assert ab["confidence_set"] is None and it["effective_f"] is None
    if Ke == 1:
        # one endogenous regressor: the KP LM statistic is the robust J of
        # y1 on nothing, the Sanderson-Windmeijer statistic
        m = Z1.T @ Y[:, 0]
        S = omega(Y)
        J = m @ invsym(S)[0] @ m
        np.testing.assert_allclose(lm, J, rtol=1e-8)
        np.testing.assert_allclose(it["sanderson_windmeijer"][0]["statistic"], J, rtol=1e-6)
    else:
        # Sanderson-Windmeijer for each: 2-step GMM J of y_k on the others
        for k in range(Ke):
            o_ = [i for i in range(Ke) if i != k]
            A, yk = Y[:, o_], Y[:, k]
            P = Z1 @ np.linalg.solve(Z1.T @ Z1, Z1.T)
            d = np.linalg.solve(A.T @ P @ A, A.T @ P @ yk)
            e = yk - A @ d
            U = np.zeros((N, L1))
            np.add.at(U, pid, Z1 * e[:, None])
            iS = invsym(U.T @ U)[0]
            zA, zy = Z1.T @ A, Z1.T @ yk
            d2 = np.linalg.solve(zA.T @ iS @ zA, zA.T @ iS @ zy)
            mm = zy - zA @ d2
            np.testing.assert_allclose(it["sanderson_windmeijer"][k]["statistic"],
                                       mm @ iS @ mm, rtol=1e-6)


def _omega(Z1, E, codes):
    """sum over clusters of vec(Z1_c'E_c) vec(Z1_c'E_c)'."""
    L1, Ke = Z1.shape[1], E.shape[1]
    _, inv = np.unique(codes, return_inverse=True)
    U = np.zeros((inv.max() + 1, L1 * Ke))
    for k in range(Ke):
        np.add.at(U[:, k * L1:(k + 1) * L1], inv, Z1 * E[:, [k]])
    return U.T @ U


@pytest.mark.parametrize("case", ["nested", "crossing", "aweights"])
def test_idtests_clusters_weights(panel_data, case):
    """Clustered by a variable (the rows a cluster has are those whose
    period is in the main equation's sample, as xtabond2 assigns clusters),
    and with weights (the weighted regressions)."""
    df = panel_data.with_columns(((pl.col("id") * 7 + pl.col("year")) % 40).alias("cl2"))
    kw, okw = {}, {}
    if case == "nested":
        kw, okw = dict(vcov={"CRV1": "cluster"}), dict(cluster=["cluster"])
    elif case == "crossing":
        kw, okw = dict(vcov={"CRV1": "cl2"}), dict(cluster=["cl2"])
    else:
        kw, okw = dict(weights="wt"), dict(weights="wt")
    fml = ("y ~ l(y, 1) + x + w | gmm(y, 2:4, collapse=True) + gmm(x, 1:3, collapse=True) "
           "+ iv(w)")
    r = dpd_stream(fml, df, panel=("id", "year"), system=True, verbose=False, **kw)
    it = r.idtests
    data = {c: df[c].to_numpy().astype(float) for c in df.columns}
    data["l(y, 1)"] = lag_by_time(df, "y", 1)
    xs = ["l(y, 1)", "x", "w"]
    o = xtabond2(data["id"], data["year"], data, "y", xs,
                 [GMM(["y"], lags=(2, 4), collapse=True), GMM(["x"], lags=(1, 3), collapse=True)],
                 [IV(["w"])], system=True, twostep=True, robust=True, **okw)
    d = o.dense
    N, R = d["Z"].shape[:2]
    T = R // 2
    Z = d["Z"].reshape(-1, d["Z"].shape[-1])
    X = d["X"].reshape(-1, d["X"].shape[-1])
    touse = d["touse"].reshape(-1)
    ids = np.unique(data["id"])
    t0 = int(data["year"].min())
    if case == "aweights":
        # the oracle's X is weighted (by the normalized weight): back to the
        # weighted regressions' rows, sqrt(w) x and sqrt(w) z (the scale of
        # the weights does not matter)
        wmap = dict(zip(data["id"], data["wt"]))
        w = np.repeat([wmap[i] for i in ids], R) * touse
        sw = np.sqrt(np.where(w > 0, w, 1.0))
        X = np.where(touse[:, None], X / (w[:, None] * (o.n_obs / w.reshape(N, R)[:, T:].sum()))
                     * sw[:, None], 0)
        Z = Z * sw[:, None] * touse[:, None]
    exog, endog = [0, 1], [0, 1]
    keepz = [c for c in range(Z.shape[1]) if c not in exog and np.any(Z[:, c])]
    X2, Z1, Y = Z[:, exog], Z[:, keepz], X[:, endog]
    B = np.linalg.lstsq(X2, np.column_stack([Z1, Y]), rcond=None)[0]
    Rz = np.column_stack([Z1, Y]) - X2 @ B
    Z1, Y = Rz[:, :Z1.shape[1]], Rz[:, Z1.shape[1]:]
    Pi = np.linalg.lstsq(Z1, Y, rcond=None)[0]
    V = Y - Z1 @ Pi
    pid = np.repeat(np.arange(N), R)
    period = np.tile(np.arange(R) % T, N)
    if case == "aweights":
        codes = pid
        mask = np.ones(len(pid), bool)
    else:
        var = okw["cluster"][0]
        grid = {(i, int(t) - t0): c for i, t, c in zip(data["id"], data["year"], data[var])}
        codes = np.array([grid.get((ids[n], p), -1) for n, p in zip(pid, period)])
        main = d["touse"][:, T:].reshape(-1)
        mask = main[pid * T + period] & (codes >= 0)
    lm, _ = kp_rk(Z1.T @ Z1, Pi, Y.T @ Y, _omega(Z1[mask], Y[mask], codes[mask]))
    wald, _ = kp_rk(Z1.T @ Z1, Pi, Y.T @ Y, _omega(Z1[mask], V[mask], codes[mask]))
    np.testing.assert_allclose(it["kp_lm"]["statistic"], lm, rtol=1e-6)
    np.testing.assert_allclose(it["kp_wald"]["statistic"], wald, rtol=1e-6)


def test_idtests_pca_all_components(panel_data):
    """PCA keeping every component spans the same instruments, which the
    rk statistics do not depend on."""
    fml = ("y ~ l(y, 1) + x + w | gmm(y, 2:4, collapse=True) + gmm(x, 1:3, collapse=True) "
           "+ iv(w)")
    a = dpd_stream(fml, panel_data, panel=("id", "year"), system=True, verbose=False)
    n_gmm = a.n_instruments - 2                 # less w's column and the constant
    b = dpd_stream(fml, panel_data, panel=("id", "year"), system=True, pca=True,
                   components=n_gmm, verbose=False)
    assert b.pca["components"] == n_gmm
    for k in ("kp_lm", "kp_wald"):
        np.testing.assert_allclose(b.idtests[k]["statistic"], a.idtests[k]["statistic"],
                                   rtol=1e-7)


@pytest.mark.parametrize("transform", ["fd", "fod"])
def test_idtests_absorbed(panel_data, transform):
    """An absorbed g^year effect is partialled out of the tests, as its
    dummies (included exogenous regressors) would be."""
    df = panel_data.with_columns((pl.col("id") % 4).alias("g"))
    df = df.with_columns((pl.col("g").cast(pl.Utf8) + "_" + pl.col("year").cast(pl.Utf8))
                         .alias("cell"))
    cells = sorted(df["cell"].unique())
    df = df.with_columns([(pl.col("cell") == c).cast(pl.Float64).alias(f"D{c}") for c in cells])
    dm = " + ".join(f"D{c}" for c in cells)
    base = "y ~ l(y, 1) + x + w{} | gmm(y, 2:4, collapse=True) + gmm(x, 1:3, collapse=True) "
    a = dpd_stream(base.format(" | g^year") + "+ iv(w)", df, panel=("id", "year"),
                   transform=transform, verbose=False)
    b = dpd_stream(base.format(f" + {dm}") + f"+ iv(w) + iv({dm})", df, panel=("id", "year"),
                   transform=transform, verbose=False)
    assert a.idtests["endogenous"] == b.idtests["endogenous"]
    for k in ("kp_lm", "kp_wald"):
        np.testing.assert_allclose(a.idtests[k]["statistic"], b.idtests[k]["statistic"],
                                   rtol=1e-6)
    for x, y in zip(a.idtests["sanderson_windmeijer"], b.idtests["sanderson_windmeijer"]):
        np.testing.assert_allclose(x["statistic"], y["statistic"], rtol=1e-6)
