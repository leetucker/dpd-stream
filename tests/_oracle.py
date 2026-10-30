"""A dense reference implementation of xtabond2's estimators, for tests.

This is the yardstick every dpd-stream estimate is checked against. It
builds every panel's instrument matrix Z_i in full on xtabond2's
balanced time grid and follows xtabond2's algebra step by step (Roodman,
2009, "How to do xtabond2", and the conventions of xtabond2 3.7.2's Mata
code: the column order of the instruments, the generalized inverse by
sweeping, sigma^2 for the one-step variance, the Windmeijer correction,
the Arellano-Bond tests and the difference-in-Sargan/Hansen tests). It
is written for clarity, not speed: O(N T^2 L) memory.

Inputs are long data as numpy arrays (`panel`, `time`, and a dict of
variables). Lags of regressors are the caller's: pass them as variables
(lagged by time value, NaN where missing), as Stata's L. operator would
give them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- helpers


DROP_NEGATIVE_PIVOTS = True


def invsym(A, tol=1e-13):
    """Stata's generalized inverse of a symmetric matrix by sweeping the
    pivots in order, dropping (zeroing) any whose pivot is negligible or
    negative (as for a matrix that is not positive semidefinite, such as
    two-way clustering's) relative to its original diagonal element.
    Returns (inverse, kept)."""
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    d0 = np.abs(np.diag(A)).copy()
    kept = np.zeros(n, bool)
    for k in range(n):
        d = A[k, k]
        if d0[k] == 0 or (d if DROP_NEGATIVE_PIVOTS else abs(d)) <= tol * d0[k]:
            A[k, :] = 0.0
            A[:, k] = 0.0
            continue
        row = A[k, :].copy()
        col = A[:, k].copy()
        A -= np.outer(col, row) / d
        A[k, :] = row / d
        A[:, k] = col / d
        A[k, k] = -1.0 / d
        kept[k] = True
    out = np.zeros_like(A)
    idx = np.flatnonzero(kept)
    out[np.ix_(idx, idx)] = -A[np.ix_(idx, idx)]
    return out, kept


def diff(x):
    """First differences along axis 1 (time); the first period is NaN."""
    out = np.full_like(x, np.nan)
    out[:, 1:] = x[:, 1:] - x[:, :-1]
    return out


def fod(x, complete):
    """Forward orthogonal deviations, stored one period later (as xtabond2
    does, to parallel first differences): row t+1 holds the deviation of
    x_t from the mean of the complete observations after t."""
    out = np.full_like(x, np.nan)
    N, T = complete.shape
    for i in range(N):
        n = 0
        s = np.zeros(x.shape[2:]) if x.ndim > 2 else 0.0
        for t in range(T - 1, -1, -1):
            if n:
                out[i, t + 1] = np.sqrt(n / (n + 1)) * (x[i, t] - s / n)
            if complete[i, t]:
                n += 1
                s = s + x[i, t]
    return out


def bod(x, complete):
    """Backward orthogonal deviations (gmm()'s orthogonal suboption), stored
    in place; missing values become 0."""
    out = np.zeros_like(x)
    N, T = complete.shape
    for i in range(N):
        n = 0
        s = np.zeros(x.shape[2:]) if x.ndim > 2 else 0.0
        for t in range(T):
            if n:
                out[i, t] = np.sqrt(n / (n + 1)) * (x[i, t] - s / n)
            if complete[i, t]:
                n += 1
                s = s + x[i, t]
    return out


def xform_matrix(orthogonal, T):
    """xtabond2's transformation matrix M (columns are the weights of each
    transformed row): first differences, or forward orthogonal deviations
    in a balanced panel."""
    M = np.zeros((T, T))
    if orthogonal:
        for c in range(1, T):
            M[c - 1, c] = 1.0
            M[c:, c] = -1.0 / (T - c)
        norms = np.sqrt((M ** 2).sum(0))
        norms[norms == 0] = 1
        return M / norms
    for c in range(1, T):
        M[c, c] = 1.0
        M[c - 1, c] = -1.0
    return M


def h_matrix(h, left, right, system, T):
    Ml = np.eye(T) if h == 1 else xform_matrix(left, T)
    Mr = np.eye(T) if h == 1 else xform_matrix(right, T)
    if h == 3 or not system:
        if system:
            Ml = np.hstack([Ml, np.eye(T)])
            Mr = np.hstack([Mr, np.eye(T)])
        H = Ml.T @ Mr
    else:
        H = np.zeros((2 * T, 2 * T))
        H[:T, :T] = Ml.T @ Mr
        H[T:, T:] = np.eye(T)
    H[np.abs(H) < 1e-12] = 0.0
    return H


# ---------------------------------------------------------------- specs


@dataclass
class GMM:
    vars: list
    lags: tuple = (1, None)
    collapse: bool = False
    eq: str = "both"            # "both", "diff", "level"
    passthru: bool = False
    orthogonal: bool = False
    split: bool = False


@dataclass
class IV:
    vars: list
    eq: str = "both"
    passthru: bool = False
    mz: bool = False


_EQ = {"level": 0, "diff": 1, "both": 2}


@dataclass
class _G:
    """A parsed gmm() group, as xtabond2 holds one."""
    vars: list
    equation: int
    passthru: bool
    collapse: bool
    orthogonal: bool
    laglim: list
    name: str = ""
    full_eq: int = 0
    full_diffed: int = 0
    extra: bool = False
    n_insts: int = 0
    first_extra: int = -1


@dataclass
class Result:
    b: np.ndarray
    V: np.ndarray
    names: list
    b1: np.ndarray = None
    b2: np.ndarray = None
    V1: np.ndarray = None
    V1robust: np.ndarray = None
    V2: np.ndarray = None
    V2robust: np.ndarray = None
    sig2: float = None
    sargan: float = None
    sargan_df: int = None
    hansen: float = None
    ar: list = field(default_factory=list)
    diffsargan: list = field(default_factory=list)
    n_obs: int = 0
    n_groups: int = 0
    j: int = 0
    j0: int = 0
    df_m: int = 0
    df_r: int = None
    chi2: float = None
    F: float = None
    g_min: int = 0
    g_max: int = 0
    pca_components: int = None
    pca_eigenvalues: np.ndarray = None
    kmo: float = None
    Z_names: list = None
    dense: dict = None


# ---------------------------------------------------------------- the estimator


def xtabond2(panel, time, data, y, x, gmm=(), iv=(), *, system=True, constant=True,
             orthogonal=False, h=3, twostep=False, robust=False, cluster=(), small=False,
             artests=2, pca=False, components=0, weights=None, wtype="aweight",
             diffsargan=True):
    """Fit xtabond2 ``y x, gmm(...) iv(...) [options]`` on long data.

    `data` maps variable names to 1-D arrays aligned with `panel` and
    `time`; `x` lists the regressors; `gmm` and `iv` are lists of `GMM` and
    `IV`, in the order they would be written on xtabond2's command line.
    `system=False` is xtabond2's ``noleveleq``.
    """
    ids, pid = np.unique(panel, return_inverse=True)
    tmin = int(np.min(time))
    tt = (np.asarray(time) - tmin).astype(int)
    T = int(tt.max()) + 1
    N = len(ids)

    def grid(v):
        g = np.full((N, T), np.nan)
        g[pid, tt] = np.asarray(v, float)
        return g

    Y0 = grid(data[y])
    X0 = np.stack([grid(data[v]) for v in x], -1) if x else np.zeros((N, T, 0))
    xnames = list(x)
    consopt = system and constant
    if consopt:
        X0 = np.concatenate([X0, np.ones((N, T, 1))], -1)
        xnames.append("_cons")
    present = np.zeros((N, T), bool)
    present[pid, tt] = True
    wt0 = None
    if weights is not None:
        wt0 = grid(data[weights])

    # Stata's marksample: the depvar, regressors (and weights) present
    touse0 = present & ~np.isnan(Y0) & ~np.isnan(X0).any(-1)
    if wt0 is not None:
        touse0 &= ~np.isnan(wt0)
    complete = ~(np.isnan(X0).any(-1) | np.isnan(Y0))

    # ---- instrument groups, as xtabond2's linked lists (last given first)
    iv_groups = []
    for spec in iv:
        eqn = _EQ[spec.eq]
        if not (system or eqn):
            continue
        if spec.passthru and eqn == 2 and system:
            raise ValueError("passthru not valid with equation(both) in system GMM")
        iv_groups.insert(0, spec)
    if consopt:
        iv_groups.append(IV(["_cons"], eq="level"))     # the constant, at the end
    for spec in iv:
        if not spec.mz and _EQ[spec.eq] in ((1, 2) if not system else (0, 1, 2)):
            base = np.stack([grid(data[v]) for v in spec.vars], -1)
            complete &= ~np.isnan(base).any(-1)

    gmm_groups = []
    for spec in gmm:
        eqn = _EQ[spec.eq]
        split = spec.split
        if split and eqn != 2:
            split = False
        if split and not system:
            split = False
        if not (system or eqn):
            continue
        if system and spec.passthru and eqn == 2:
            raise ValueError("passthru not valid with equation(both) in system GMM")
        a, b = spec.lags
        a = 1 if a is None else a
        b = np.inf if b is None else b
        if a > b:
            a, b = b, a
        g = _G(list(spec.vars), eqn, spec.passthru, spec.collapse, spec.orthogonal, [a, b],
               name=_gmm_name(spec))
        if split:
            g.equation = 1
        pieces = [g]
        if split:
            lev = _G(list(spec.vars), 0, spec.passthru, spec.collapse, spec.orthogonal,
                     [a - 1, a - 1], name=_gmm_name(spec) + " (levels)")
            g.name = _gmm_name(spec) + " (diff)"
            pieces = [lev, g]          # the levels copy precedes, as in xtabond2
        for p in pieces:
            p.extra = p.equation == 2 and system
            p.full_eq = int(not p.equation)
            p.full_diffed = int(not (p.equation or p.passthru))
            p.n_insts = len(p.vars) * _insts_per_base(p, T)
        gmm_groups = pieces + gmm_groups

    xf = (lambda v: fod(v, complete)) if orthogonal else diff

    # ---- regressors and outcome by equation: diff rows then level rows
    R = (2 if system else 1) * T

    def stack(dv, lv):
        return np.concatenate([dv, lv], 1) if system else dv

    X = stack(xf(X0), X0)
    Y = stack(xf(Y0), Y0)
    if system:
        touse = np.concatenate([np.pad(touse0, ((0, 0), (1, 0)))[:, :T] if orthogonal
                                else touse0, touse0], 1)
    else:
        touse = np.pad(touse0, ((0, 0), (1, 0)))[:, :T] if orthogonal else touse0.copy()

    # IV-style instruments
    ziv_cols, ziv_names, iv_index = [], [], []
    for spec in iv_groups:
        eqn = _EQ[spec.eq]
        if spec.vars == ["_cons"]:
            base = np.ones((N, T, 1))
        else:
            base = np.stack([grid(data[v]) for v in spec.vars], -1)
        start = sum(c.shape[-1] for c in ziv_cols)
        dpart = np.zeros_like(base)
        if eqn:
            dpart = base if spec.passthru else xf(base)
            if spec.mz:
                dpart = np.nan_to_num(dpart, nan=0.0)
        if system:
            lpart = np.zeros_like(base)
            if eqn != 1:
                lpart = np.nan_to_num(base, nan=0.0) if spec.mz else base
            col = np.concatenate([dpart, lpart], 1)
        else:
            col = dpart
        ziv_cols.append(col)
        ziv_names += spec.vars
        if spec.vars != ["_cons"]:
            iv_index.append((_iv_name(spec), list(range(start, start + base.shape[-1]))))
    Z_IV = np.concatenate(ziv_cols, -1) if ziv_cols else np.zeros((N, R, 0))
    j_IV = Z_IV.shape[-1]

    touse &= ~(np.isnan(Z_IV).any(-1) | np.isnan(X).any(-1) | np.isnan(Y))

    # GMM-style instruments
    Z_GMM, gmm_index, zgmm_names = _make_gmm(gmm_groups, data, grid, complete, touse, N, T,
                                             system)
    j_GMM = Z_GMM.shape[-1]

    # zero out excluded rows
    m = touse[..., None]
    X = np.where(m, np.nan_to_num(X, nan=0.0), 0.0)
    Y = np.where(touse, np.nan_to_num(Y, nan=0.0), 0.0)
    Z_IV = np.where(m, np.nan_to_num(Z_IV, nan=0.0), 0.0)
    Z_GMM = np.where(m, Z_GMM, 0.0)

    keep = _rmcoll(X.reshape(-1, X.shape[-1]), consopt)
    X = X[..., keep]
    names = [xnames[k] for k in keep]
    k = X.shape[-1]
    if k == 0:
        raise ValueError("no regressors")

    main = touse[:, T:] if system else touse
    n_obs = int(main.sum())
    n_groups = int(main.any(1).sum())
    ti = main.sum(1)
    ti = ti[ti > 0]
    n_obs_eff = n_obs

    # weights
    wt = None
    wttot = int(touse[:, :T].sum()) if (system and h > 1) else n_obs
    if wt0 is not None:
        wt = np.where(touse, np.nan_to_num(np.concatenate([wt0, wt0], 1) if system else wt0,
                                           nan=0.0), 0.0)
        main_wt = wt[:, T:] if system else wt
        tot = main_wt.sum()
        if wtype.startswith("f"):
            if np.any(np.nan_to_num(diff(np.where(present, wt0, np.nan))) != 0):
                raise ValueError("frequency weights must be constant over time")
            n_obs = tot
            if not (twostep or robust or cluster):
                n_obs_eff = tot
            if system and h > 1:
                wttot = wt[:, :T].sum()
            else:
                wt = wt * (n_obs_eff / tot)
                wttot = int(touse[:, :T].sum()) if (system and h > 1) else n_obs_eff
        else:
            wt = wt * (n_obs / tot)
            wttot = int(touse[:, :T].sum()) if (system and h > 1) else n_obs
        X = X * wt[..., None]
        Y = Y * wt

    pca_info = {}
    if pca and j_GMM:
        nz = (Z_GMM != 0).any((0, 1))
        Z_GMM = Z_GMM[..., nz]
        zgmm_names = [n for n, z in zip(zgmm_names, nz) if z]
        w = (wt if wt is not None else touse.astype(float)).reshape(-1)
        Zf = Z_GMM.reshape(-1, Z_GMM.shape[-1])
        sw = w.sum()
        mu = (w @ Zf) / sw
        C = ((Zf - mu) * w[:, None]).T @ (Zf - mu)
        sd = np.sqrt(np.diag(C))
        S = C / np.outer(sd, sd)
        Sinv, _ = invsym(S)
        dS = np.sqrt(np.diag(Sinv))
        P = Sinv / np.outer(dS, dS)
        lo = np.tril_indices_from(S, -1)
        kmo = 1 / (1 + (P[lo] ** 2).sum() / (S[lo] ** 2).sum())
        evals, evecs = np.linalg.eigh(S)
        order = np.argsort(-evals, kind="stable")
        evals, evecs = evals[order], evecs[:, order]
        evals[np.abs(evals) < 1e-12] = 0.0
        if components:
            jg = min(components, Z_GMM.shape[-1])
        else:
            jg = max(int((evals > .999999).sum()), k - j_IV)
        Z_GMM = Z_GMM @ evecs[:, :jg]
        j_GMM = jg
        pca_info = dict(components=jg, eigenvalues=evals, kmo=kmo)

    j = j_GMM + j_IV
    if j < k:
        raise ValueError("equation not identified")
    Z = np.concatenate([Z_IV, Z_GMM], -1)

    # within each panel: rows diff eq (t = 0..T-1) then level eq
    H = h_matrix(h, orthogonal, orthogonal, system, T)
    sw = np.sqrt(wt) if wt is not None else np.ones_like(Y)
    S1 = np.einsum("nri,nr,rs,ns,nsj->ij", Z, sw, H, sw, Z)
    ZX = np.einsum("nri,nrk->ik", Z, X)
    ZY = np.einsum("nri,nr->i", Z, Y)

    A1, _ = invsym(S1)
    V1, _ = invsym(ZX.T @ A1 @ ZX)
    b1 = V1 @ ZX.T @ A1 @ ZY
    e1 = Y - X @ b1

    def sigma2(e):
        we = e / np.sqrt(wt) if wt is not None else e
        we = np.nan_to_num(we)
        if system:
            mask = np.zeros(R, bool)
            if h != 1:
                mask[:T] = True
            else:
                mask[T:] = True
            we = we[:, mask]
        return (we ** 2).sum() / (2 - (orthogonal or h == 1)) / wttot

    sig2 = sigma2(e1)
    A1 = A1 / sig2
    V1 = V1 * sig2
    j0 = j - int((np.diag(A1) == 0).sum())

    Ze = np.einsum("nri,nr->i", Z, e1)
    sargan = float(Ze @ A1 @ Ze)
    sargan_df = j0 - k

    steps = 2 if twostep else 1
    robust = bool(cluster) or robust or (wtype.startswith("p") and weights is not None
                                         and steps == 1)
    onestepnonrobust = steps == 1 and not robust

    # clusters: every combination of the cluster variables, by
    # inclusion-exclusion, each as a vector of cluster ids per row
    clusts = []
    if cluster:
        cl_grids = [grid(data[c]) for c in cluster]
        for combo in _combos(len(cluster)):
            key = np.stack([cl_grids[c] for c in combo], -1)
            full = np.concatenate([key, key], 1) if system else key
            flat = full.reshape(-1, len(combo))
            # a row with no data of its own (the orthogonal deviation stored
            # at a gap) has no cluster, as in xtabond2
            tu = touse.reshape(-1) & ~np.isnan(flat).any(1)
            cid = np.full(flat.shape[0], -1)
            _, inv = np.unique(flat[tu], axis=0, return_inverse=True)
            cid[tu] = inv.reshape(-1)
            clusts.append(dict(id=cid, n=int(inv.max()) + 1,
                               sign=1 if len(combo) % 2 else -1, onevar=len(combo) == 1))

    hansen = None
    V1robust = V2 = V2robust = b2 = e2 = A2 = None
    Zf = Z.reshape(-1, j)
    Xf = X.reshape(-1, k)
    if onestepnonrobust:
        S = S1 * sig2
    else:
        def cluster_sums(e):
            """[(sign * factor, [Z_c'e_c per cluster], cluster ids)]."""
            out = []
            if clusts:
                for c in clusts:
                    fac = c["sign"] * (c["n"] / (c["n"] - 1) if small else 1)
                    Zei = np.zeros((c["n"], j))
                    ok = c["id"] >= 0
                    np.add.at(Zei, c["id"][ok], Zf[ok] * e.reshape(-1)[ok, None])
                    out.append((fac, Zei, c["id"]))
            else:
                out.append((1.0, np.einsum("nri,nr->ni", Z, e), None))
            return out

        sums1 = cluster_sums(e1)
        S = sum(f * Zei.T @ Zei for f, Zei, _ in sums1)
        VXZA = V1 @ ZX.T @ A1
        V1robust = VXZA @ S @ VXZA.T
        A2, _ = invsym(S)
        V2, _ = invsym(ZX.T @ A2 @ ZX)
        VXZA2 = V2 @ ZX.T @ A2
        b2 = VXZA2 @ ZY
        e2 = Y - X @ b2
        if steps == 2:
            sig2 = sigma2(e2)
        Ze = np.einsum("nri,nr->i", Z, e2)
        A2Ze = A2 @ Ze
        if steps == 2 and robust:
            Dr = np.zeros((j, k))
            for f, Zei, cid in sums1:
                if cid is None:
                    ZXi = np.einsum("nri,nrk->nik", Z, X)
                else:
                    ZXi = np.zeros((Zei.shape[0], j, k))
                    ok = cid >= 0
                    np.add.at(ZXi, cid[ok], Zf[ok][:, :, None] * Xf[ok][:, None, :])
                s = Zei @ A2Ze                                    # e1_i'Z_i A2 Z'e2
                Dr += f * (np.einsum("n,nik->ik", s, ZXi)
                           + np.einsum("ni,nlk,l->ik", Zei, ZXi, A2Ze))
            D = VXZA2 @ Dr
            V2robust = V2 + D @ V1robust @ D.T + 2 * D @ V2
        hansen = float(Ze @ A2Ze)

    if steps == 1:
        b = b1
        V = V1robust if robust else V1
    else:
        b = b2
        V = V2robust if robust else V2
    V = (V + V.T) / 2

    pV, pA, pe = (V1, A1, e1) if steps == 1 else (V2, A2, e2)
    m2VZXA = -2 * pV @ ZX.T @ pA
    pVar = (V1robust if steps == 1 else V2robust) if robust else pV

    df_r = None
    if small:
        tmp = wttot / (wttot - k)
        if onestepnonrobust:
            fac = tmp
        elif clusts:
            fac = (n_obs - 1) / (n_obs - k)
        else:
            fac = (n_obs - 1) / (n_obs - k) * n_groups / (n_groups - 1)
        V = V * fac
        sig2 = sig2 * tmp

    pX = X[:, T:] if system else X
    pXf = pX.reshape(-1, k)
    # xtabond2 also tries to drop the constant from the count when it lies in
    # the column space of X, with a tolerance (eps * rows) that roundoff
    # defeats in practice; its published output counts the rank (e.g. Wald
    # chi2(16) in Roodman 2009, p. 111), and so does this
    df_m = np.linalg.matrix_rank(pXf) - consopt
    min_clust = min(c["n"] for c in clusts) if clusts else None
    chi2 = F = None
    Vinv, _ = invsym(V)
    wald = float(b @ Vinv @ b)
    if df_m:
        if small:
            F = wald / df_m
            df_r = (n_obs_eff - df_m if onestepnonrobust
                    else (min_clust if clusts else n_groups)) - consopt
        else:
            chi2 = wald
    elif small:
        df_r = (n_obs_eff if onestepnonrobust else (min_clust if clusts else n_groups)) - consopt

    # difference-in-Sargan/Hansen
    dsargans = []
    if diffsargan and not pca:
        A1diag = np.diag(A1)
        groups = [(name, ("iv", cols)) for name, cols in iv_index] + \
                 [(name, ("gmm", cols)) for name, cols in gmm_index]
        if system:
            # the levels instruments: every column of eq(level) groups, and
            # the extra columns of eq(both) groups (xtabond2 allots T of them
            # per variable but fills T - 1; the diff ones are its first
            # NumInsts - T columns either way)
            lev = set()
            for g, (_, cols) in zip(gmm_groups, gmm_index):
                if g.full_eq == 0:
                    lev |= set(range(g.first_extra, cols[-1] + 1)) if g.extra else set()
                else:
                    lev |= set(cols)
            groups.append(("GMM instruments for levels", ("gmm", sorted(lev))))
        for name, (kind, cols) in groups:
            drop = set(cols) if kind == "iv" else {j_IV + c for c in cols}
            p = np.array([c for c in range(j) if c not in drop], int)
            n_other = int((A1diag[p] != 0).sum())
            if not (n_other >= k and n_other < j0):
                continue
            App, _ = invsym(S[np.ix_(p, p)])
            ZXp = ZX[p]
            _V, _ = invsym(ZXp.T @ App @ ZXp)
            if (np.diag(_V) == 0).sum() != (np.diag(pVar) == 0).sum():
                continue
            br = _V @ ZXp.T @ App @ ZY[p]
            er = Y - X @ br
            Zer = np.einsum("nri,nr->i", Z[..., p], er)
            unres = float(Zer @ App @ Zer)
            n_g = j0 - n_other - int((np.diag(V2 if V2 is not None else V1) == 0).sum())
            full = hansen if hansen is not None else sargan
            dsargans.append(dict(name=name, sargan=unres, diff=full - unres, n=n_g,
                                 sargan_p=stats.chi2.sf(unres, sargan_df - n_g),
                                 diff_p=stats.chi2.sf(full - unres, n_g)))

    # Arellano-Bond tests
    ar = _ar_tests(artests, onestepnonrobust, h, N, T, sig2, orthogonal, system, Z, X, X0, Y0,
                   keep, b, wt, wt0, pe, touse, complete, m2VZXA, pVar)

    return Result(b=b, V=V, names=names, b1=b1, b2=b2, V1=V1, V1robust=V1robust, V2=V2,
                  V2robust=V2robust, sig2=sig2, sargan=sargan, sargan_df=sargan_df,
                  hansen=hansen, ar=ar, diffsargan=dsargans, n_obs=n_obs, n_groups=n_groups,
                  j=j, j0=j0, df_m=int(df_m), df_r=df_r, chi2=chi2, F=F,
                  g_min=int(ti.min()), g_max=int(ti.max()),
                  pca_components=pca_info.get("components"),
                  pca_eigenvalues=pca_info.get("eigenvalues"), kmo=pca_info.get("kmo"),
                  Z_names=ziv_names + zgmm_names,
                  dense=dict(Z=Z, X=X, Y=Y, touse=touse, j_IV=j_IV))


def _combos(n):
    """The nonempty subsets of range(n), in xtabond2's order (strCombs)."""
    def rec(i):
        if i == n - 1:
            return [(), (i,)]
        rest = rec(i + 1)
        return [r for r in rest] + [(i,) + r for r in rest]
    return [c for c in rec(0) if c]


def _gmm_name(spec):
    s = "gmm(" + " ".join(spec.vars)
    a, b = spec.lags
    opts = []
    if spec.collapse:
        opts.append("collapse")
    if spec.eq != "both":
        opts.append(f"eq({spec.eq})")
    opts.append(f"lag({1 if a is None else a} {'.' if b is None else b})")
    return s + ", " + " ".join(opts) + ")"


def _iv_name(spec):
    s = "iv(" + " ".join(spec.vars)
    if spec.eq != "both":
        s += f", eq({spec.eq})"
    return s + ")"


def _insts_per_base(g, T):
    """Instruments per base variable (and clips the lag limits), as
    xtabond2's _GMMinstPerBaseVar."""
    lag_max = T - 1 - (g.full_eq and g.full_diffed)
    fwd_max = T - 2 + g.full_eq
    g.laglim = [max(-fwd_max, g.laglim[0]), min(lag_max, g.laglim[1])]
    a, b = g.laglim
    if g.collapse:
        return int(b - a + 1 + g.extra)
    if a > 0 or b <= 0:
        if a >= 0:
            n1, n2 = lag_max - a + 1, lag_max - b
        else:
            n1, n2 = fwd_max + b + 1, fwd_max + a
        n = (n1 * (n1 + 1) - n2 * (n2 + 1)) // 2
    else:
        n1, n2 = fwd_max + a, lag_max - b
        n = (lag_max + 1) * (fwd_max + 1) - (n1 * (n1 + 1) + n2 * (n2 + 1)) // 2
    return int(n + (T - g.full_diffed if g.extra else 0))


def _make_gmm(groups, data, grid, complete, touse, N, T, system):
    """The GMM-style instruments, (N, R, j_GMM), by xtabond2's construction:
    for each group, for each lag, one column per (source period, variable)
    -- or per variable if collapsed -- then the extra levels instruments
    of a system's eq(both) groups."""
    R = (2 if system else 1) * T
    cols, index, names = [], [], []
    for g in groups:
        nb = len(g.vars)
        base = np.stack([grid(data[v]) for v in g.vars], -1)          # (N, T, nb)
        lev_base = bod(base, complete) if g.orthogonal else np.nan_to_num(base, nan=0.0)
        dbase = np.nan_to_num(diff(base), nan=0.0)
        src = dbase if g.full_diffed else lev_base
        off = T if g.full_eq else 0               # rows of the target equation
        a, b = g.laglim
        start = len(cols)
        # source range (1-based periods), as xtabond2's SubscriptsBase
        if a + g.full_eq > 0:
            lo, hi = 1 + g.full_diffed, T - a
        else:
            lo, hi = 2 - g.full_eq - a, T
        lag = a
        while lag <= b:
            sources = list(range(lo, hi + 1))
            if g.collapse:
                for v in range(nb):
                    z = np.zeros((N, R))
                    for s in sources:
                        z[:, off + s - 1 + lag] = src[:, s - 1, v]
                    cols.append(z)
                    names.append((g.name, lag, v, None))
            else:
                for s in sources:
                    for v in range(nb):
                        z = np.zeros((N, R))
                        z[:, off + s - 1 + lag] = src[:, s - 1, v]
                        cols.append(z)
                        names.append((g.name, lag, v, s - 1))
            if lag >= 0:
                hi -= 1
            if lag < 0 or (lag == 0 and not g.full_diffed):
                lo += 1
            lag += 1
        if g.extra:
            a, b = g.laglim
            if a > 0:
                lag, stop, direction = a - 1, b - 1, 1
            elif b > 0:
                lag, stop, direction = 0, b - 1, 1
            else:
                lag, stop, direction = b - 1, a - 1, -1
            lo, hi = (1, T - lag) if lag >= 0 else (1 - lag, T)
            first = len(cols)
            g.first_extra = first
            if g.collapse:
                for v in range(nb):
                    z = np.zeros((N, R))
                    for s in range(lo, hi + 1):
                        z[:, T + s - 1 + lag] = dbase[:, s - 1, v]
                    cols.append(z)
                    names.append((g.name + " levels", lag, v, None))
            else:
                for s in range(lo, hi + 1):
                    for v in range(nb):
                        z = np.zeros((N, R))
                        z[:, T + s - 1 + lag] = dbase[:, s - 1, v]
                        cols.append(z)
                        names.append((g.name + " levels", lag, v, s - 1))
            # columns with no nonzero value in the sample move to deeper lags,
            # one row at a time, with one lag counter for the whole group (as
            # xtabond2 does, on the whole stacked matrix)
            for c in range(first, len(cols)):
                while (lag < stop if direction == 1 else lag > stop) and \
                        not np.any((cols[c] != 0) & touse):
                    flat = cols[c][:, T:].reshape(-1)
                    if direction == 1:
                        flat = np.concatenate([[0.0], flat[:-1]])
                    else:
                        flat = np.concatenate([flat[1:], [0.0]])
                    cols[c][:, T:] = flat.reshape(N, T)
                    lag += direction
        index.append((g.name, list(range(start, len(cols)))))
    Z = np.stack(cols, -1) if cols else np.zeros((N, R, 0))
    return Z, index, names


def _rmcoll(X, hascons):
    """Columns kept after xtabond2's collinearity check (the constant, last,
    always kept)."""
    K = X.shape[1]
    if K <= 1:
        return list(range(K))
    if hascons:
        t = X[:, -1]
        pX = X - np.outer(t, X.T @ t / t.sum())
        U = pX.T @ pX
        U = U[:-1, :-1]
    else:
        U = X.T @ X
    d = np.diag(U)
    inv = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 0.0)
    U = U * inv[:, None] * inv[None, :]
    _, kept = invsym(U)
    keep = [i for i in range(U.shape[0]) if kept[i]]
    if hascons:
        keep.append(K - 1)
    return keep


def _ar_tests(artests, onestepnonrobust, h, N, T, sig2, orthogonal, system, Z, X, X0, Y0, keep,
              b, wt, wt0, e, touse, complete, m2VZXA, pV):
    """Arellano-Bond tests for AR(1)..AR(artests) in the differenced
    residuals, as xtabond2 computes them (Arellano & Bond 1991, eq. 8)."""
    tdiff = touse[:, :T]
    if orthogonal:
        X0k = X0[..., keep]
        wl = diff(Y0 - np.nan_to_num(X0k) @ b * np.where(np.isnan(X0k).any(-1), np.nan, 1))
        pX = diff(X0k)
        if wt0 is not None:
            pX = pX * wt0[..., None]
            w = wl * wt0
        else:
            w = wl.copy()
        wl = np.nan_to_num(wl)
        w = np.nan_to_num(w)
        pX = np.nan_to_num(pX)
    else:
        w = e[:, :T]
        pX = X[:, :T]
        wl = w / np.where(wt[:, :T] > 0, wt[:, :T], 1) if wt is not None else w.copy()
    if onestepnonrobust:
        Hd = h_matrix(h, False, False, False, T) * sig2
        psit = h_matrix(h, False, orthogonal, False, T) * sig2
        if system:
            psit = np.hstack([psit, np.zeros((T, T))])
    out = []
    wl = wl.copy()
    for _ in range(1, artests + 1):
        wl = np.concatenate([np.zeros((N, 1)), wl[:, :-1]], 1)
        s = (w * wl).sum(1)
        if not np.any(s):
            out.append((np.nan, np.nan))
            continue
        if onestepnonrobust:
            wli = wl * tdiff
            _wt = wt[:, :T] if wt is not None else np.ones((N, T))
            wHw = float(np.einsum("nt,ts,ns,ns->", wli, Hd, _wt, wli))
            psiw = np.einsum("tr,nt,nt->nr", psit, _wt, wli)
            ZHw = np.einsum("nri,nr->i", Z, psiw)
        else:
            wHw = float((s * s).sum())
            ZHw = np.einsum("nri,nr,n->i", Z, e, s)
        tmp = np.einsum("ntk,nt->k", pX, wl)
        z = s.sum() / np.sqrt(wHw + tmp @ (m2VZXA @ ZHw + pV @ tmp))
        out.append((float(z), float(2 * stats.norm.sf(abs(z)))))
    return out
