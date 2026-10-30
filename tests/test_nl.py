"""Ahn and Schmidt's nonlinear moments against a dense computation: every
panel's moments built directly, the two-step estimate by scipy's
minimizer, Windmeijer's general correction by numerical derivatives."""

import numpy as np
import pytest
from _oracle import GMM, IV, invsym, xtabond2
from conftest import lag_by_time
from scipy import optimize

from dpd_stream import dpd_stream

FML = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"
XS = ["l(y, 1)", "x", "w"]


def _panels(df, xs):
    """Per panel, the levels on the time grid: y (T,), X (T, K), with NaN
    where missing."""
    lags = {"l(y, 1)": lag_by_time(df, "y", 1)}
    ids = df["id"].to_numpy()
    tt = df["year"].to_numpy() - df["year"].min()
    T = tt.max() + 1
    cols = {v: (lags[v] if v in lags else df[v].to_numpy().astype(float)) for v in xs}
    out = []
    for i in np.unique(ids):
        m = ids == i
        y = np.full(T, np.nan)
        X = np.full((T, len(xs)), np.nan)
        y[tt[m]] = df["y"].to_numpy()[m]
        for k, v in enumerate(xs):
            X[tt[m], k] = cols[v][m]
        out.append((y, X))
    return out, T


def _nl_moments(y, X, b, T, collapse, kind="noserial"):
    """One panel's Ahn-Schmidt moments at b, and how many terms each sums."""
    ok = ~(np.isnan(y) | np.isnan(X).any(1))
    L = (1 if collapse else T) * (2 if kind == "iid" else 1)
    m, n = np.zeros(L), np.zeros(L)
    idx = np.flatnonzero(ok)
    u = y - X @ b
    if kind == "noserial":
        if len(idx) == 0 or idx[-1] < 2:
            return m, n
        Ti = idx[-1]
        for t in range(1, Ti):
            # as xtdpdgmm: the term needs the next period observed too
            if ok[t] and ok[t - 1] and ok[t + 1]:
                c = 0 if collapse else t
                m[c] += u[Ti] * (u[t] - u[t - 1])
                n[c] += 1
        return m, n
    if len(idx) < 2:
        return m, n
    ubar = u[ok].mean()
    off = 1 if collapse else T
    for t in range(1, T):
        if ok[t] and ok[t - 1]:
            c = 0 if collapse else t
            m[c] += ubar * (u[t] - u[t - 1])
            n[c] += 1
            if t >= 2 and ok[t - 2]:
                m[off + c] += y[t - 2] * (u[t - 1] - u[t - 2]) - y[t - 1] * (u[t] - u[t - 1])
                n[off + c] += 1
    return m, n


CASES = [  # system, collapse, kind, nl_weight, steps
    (False, False, "noserial", 0.0, 2), (True, True, "noserial", 0.0, 2),
    (False, False, "noserial", 1.0, 2), (True, False, "noserial", 1.0, 2),
    (False, True, "noserial", 1.0, 2), (False, False, "noserial", 1.0, 1),
    (False, False, "iid", 1.0, 2), (False, True, "iid", 1.0, 2), (True, False, "iid", 0.0, 2),
    (True, True, "iid", 1.0, 1),
]


@pytest.mark.parametrize("system,collapse,kind,weight,steps", CASES)
def test_nl(panel_data, system, collapse, kind, weight, steps):
    df = panel_data
    r = dpd_stream(FML, df, panel=("id", "year"), system=system, nl=kind, nl_weight=weight,
                   collapse=collapse, steps=steps, verbose=False)
    data = {c: df[c].to_numpy().astype(float) for c in df.columns}
    data["l(y, 1)"] = lag_by_time(df, "y", 1)
    o = xtabond2(data["id"], data["year"], data, "y", XS,
                 [GMM(["y"], lags=(2, 4), collapse=collapse), GMM(["x"], lags=(1, 3),
                                                                 collapse=True)],
                 [IV(["w"])], system=system, robust=True)
    d = o.dense
    Z, X, Y, touse = d["Z"], d["X"], d["Y"], d["touse"]
    panels, T = _panels(df, XS)
    K = X.shape[-1]
    j = Z.shape[-1]

    def levels_X(Xl):
        return np.column_stack([Xl, np.ones(len(Xl))]) if system else Xl

    def g_i(b):
        lin = np.einsum("nri,nr->ni", Z, Y - X @ b)
        nl = np.array([_nl_moments(y, levels_X(Xl), b, T, collapse, kind)[0]
                       for y, Xl in panels])
        return np.hstack([lin, nl])
    counts = np.array([_nl_moments(y, levels_X(Xl), o.b1, T, collapse, kind)[1]
                       for y, Xl in panels]).sum(0)
    lin_live = np.flatnonzero(np.abs(Z).sum((0, 1)) > 0)
    nl_live = np.flatnonzero(counts > 0)
    keep_cols = np.concatenate([lin_live, j + nl_live])

    def gbar(b):
        return g_i(b)[:, keep_cols].sum(0)

    def jac(fun, b, h=1e-6):
        return np.stack([(fun(b + h * e) - fun(b - h * e)) / (2 * h) for e in np.eye(K)], -1)

    def argmin(A, b0):
        res = optimize.minimize(lambda b: gbar(b) @ A @ gbar(b), b0, method="BFGS",
                                options=dict(gtol=1e-13, maxiter=10_000))
        return res.x

    # the first step: xtdpdgmm's initial weight, (Z'HZ)^- and the nonlinear
    # moments' 1 / (number of panels with an observation in levels) times the
    # weight, the same for every moment
    from dpd_stream.layout import h_matrix
    Zu = Z * touse[..., None]
    H = h_matrix(3, False, False, system, T)
    S1 = np.einsum("nri,rs,nsj->ij", Zu, H, Zu)[np.ix_(lin_live, lin_live)]
    W1 = np.zeros((len(keep_cols),) * 2)
    W1[:len(lin_live), :len(lin_live)] = invsym(S1)[0]
    n_levels = sum(1 for y, Xl in panels if (~(np.isnan(y) | np.isnan(levels_X(Xl)).any(1))).any())
    W1[len(lin_live):, len(lin_live):] = np.eye(len(nl_live)) * weight / n_levels
    half = 1 if collapse else T
    sel = nl_live[nl_live >= half] - half if kind == "iid" else np.zeros(0, int)
    if len(sel):
        # nl(iid): the linear moments it adds are instruments of the differenced
        # equation (y_{t-2} in row t - 1, -y_{t-1} in row t) in the linear block,
        # weighted together with it by (Z'HZ)^-
        C = np.zeros(Z.shape[:2] + (half,))
        for n, (y, Xl) in enumerate(panels):
            ok = ~(np.isnan(y) | np.isnan(levels_X(Xl)).any(1))
            for t in range(2, T):
                if ok[t] and ok[t - 1] and ok[t - 2]:
                    cc = 0 if collapse else t
                    C[n, t - 1, cc] += y[t - 2]
                    C[n, t, cc] -= y[t - 1]
        Zaug = np.concatenate([Zu[..., lin_live], C[..., sel] * touse[..., None]], -1)
        Saug = np.einsum("nri,rs,nsj->ij", Zaug, H, Zaug)
        nL, nin = len(lin_live), len(sel)
        pos = list(range(nL)) + list(range(W1.shape[0] - nin, W1.shape[0]))
        W1[np.ix_(pos, pos)] = invsym(Saug)[0]
    b1 = o.b1 if weight == 0 else argmin(W1, o.b1)
    np.testing.assert_allclose(r.onestep.beta if steps == 2 else r.beta, b1, rtol=1e-6,
                               atol=1e-8)
    G1 = g_i(b1)[:, keep_cols]
    S = G1.T @ G1
    A, _ = invsym(S)
    Gb1 = jac(gbar, b1)
    B1 = np.linalg.inv(Gb1.T @ W1 @ Gb1)
    V1r = B1 @ Gb1.T @ W1 @ S @ W1 @ Gb1 @ B1
    if steps == 1:
        np.testing.assert_allclose(r.se, np.sqrt(np.diag(V1r)), rtol=1e-5)
        return
    b2 = argmin(A, b1)
    np.testing.assert_allclose(r.beta, b2, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r.hansen["statistic"], gbar(b2) @ A @ gbar(b2), rtol=1e-6)
    # Windmeijer (2005), the general form, with numerical derivatives
    Gb = jac(gbar, b2)
    V2 = np.linalg.inv(Gb.T @ A @ Gb)
    Ji = -jac(lambda b: g_i(b)[:, keep_cols], b1)              # n x L x K
    v = A @ gbar(b2)
    a = G1 @ v
    Draw = np.einsum("n,nlk->lk", a, Ji) + np.einsum("nl,nmk,m->lk", G1, Ji, v)
    D = V2 @ (-Gb).T @ A @ Draw
    Vc = V2 + D @ V1r @ D.T + D @ V2 + V2 @ D.T
    np.testing.assert_allclose(r.se, np.sqrt(np.diag(Vc)), rtol=1e-4)
    assert r.nl["converged"]
