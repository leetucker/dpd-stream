"""The algebra after the passes: everything here works on j x j and j x K
sums, never on data.

The conventions are xtabond2's (version 3.7.2), as
follows: the generalized inverse by sweeping in column order (`invsym`),
sigma^2 for the one-step variance and the Sargan test, the Windmeijer
(2005) correction, the Arellano-Bond (1991) tests, and the
difference-in-Sargan/Hansen tests, which re-estimate without each group
of instruments using the corresponding submatrix of the full model's
moment covariance (so they need no further pass over the data).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import stats

# Stata's invsym() treats a matrix that is not positive semidefinite as
# singular: a pivot that is negative is dropped like a zero one (as with
# two-way clustering's moment covariance). Tests of equivalences that need an
# exact inverse turn it off.
DROP_NEGATIVE_PIVOTS = True


@njit(cache=True)
def _sweep(A, tol, neg):
    n = A.shape[0]
    d0 = np.empty(n)
    for k in range(n):
        d0[k] = abs(A[k, k])
    kept = np.zeros(n, np.bool_)
    for k in range(n):
        d = A[k, k]
        if d0[k] == 0.0 or (d if neg else abs(d)) <= tol * d0[k]:
            for i in range(n):
                A[k, i] = 0.0
                A[i, k] = 0.0
            continue
        row = A[k, :].copy()
        col = A[:, k].copy()
        for i in range(n):
            ci = col[i] / d
            if ci != 0.0:
                for jj in range(n):
                    A[i, jj] -= ci * row[jj]
        for i in range(n):
            A[k, i] = row[i] / d
            A[i, k] = col[i] / d
        A[k, k] = -1.0 / d
        kept[k] = True
    return kept


def invsym(A, tol=1e-13):
    """The inverse of symmetric `A`, or with A singular a generalized
    inverse: pivots are swept in column order and a column whose pivot is
    negligible or negative (relative to its diagonal element) is dropped --
    zero rows and columns in the result -- as Stata's ``invsym()`` does,
    which treats a matrix that is not positive semidefinite (two-way
    clustering's, say) as singular. Returns (inverse, kept)."""
    A = np.array(A, dtype=np.float64, copy=True)
    if A.size == 0:
        return A, np.zeros(0, bool)
    kept = _sweep(A, tol, DROP_NEGATIVE_PIVOTS)
    out = np.zeros_like(A)
    idx = np.flatnonzero(kept)
    out[np.ix_(idx, idx)] = -A[np.ix_(idx, idx)]
    return (out + out.T) / 2, kept


def rmcoll(XX, hascons, tol=1e-9):
    """The regressors to keep (indices) from X'X (`XX`, K x K), as
    xtabond2: with a constant (the last column) the others are taken net of
    it; the cross-product matrix is scaled to unit diagonal and swept, and
    columns that are collinear with earlier ones are dropped.

    The tolerance is looser than the generalized inverse's: Stata forms X'X
    in quad precision, where an exact collinearity leaves a pivot near
    1e-16; summed in double precision over millions of rows it can leave
    1e-12 (forward orthogonal deviations of a full set of time dummies, say,
    which sum to exactly zero)."""
    K = XX.shape[0]
    if K <= 1:
        return list(range(K))
    if hascons:
        n1 = XX[-1, -1]
        U = XX - np.outer(XX[:, -1], XX[-1, :]) / n1 if n1 > 0 else XX.copy()
        U = U[:-1, :-1]
    else:
        U = XX.copy()
    d = np.diag(U)
    inv = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 0.0)
    U = U * inv[:, None] * inv[None, :]
    _, kept = invsym(U, tol=tol)
    keep = [i for i in range(U.shape[0]) if kept[i]]
    if hascons:
        keep.append(K - 1)
    return keep


def gmm(ZX, ZY, A):
    """b = (X'Z A Z'X)^- X'Z A Z'y, with V = (X'Z A Z'X)^-."""
    XZA = ZX.T @ A
    V, _ = invsym(XZA @ ZX)
    return V @ (XZA @ ZY), V


def sigma2(Q, b, system, h, orthogonal, wttot):
    """xtabond2's sigma^2 from the residual sum of squares of the
    transformed equation (levels with ``h=1`` in system GMM), divided by 2
    for first differences."""
    Qe = Q[1] if (system and h == 1) else Q[0]
    v = np.append(-b, 1.0)
    return float(v @ Qe @ v) / (2 - (orthogonal or h == 1)) / wttot


def windmeijer(V2, ZX, A2, Draw, V1robust):
    """Windmeijer's corrected two-step vcov from xtabond2's sums `Draw`
    (j x K): D = V2 X'Z A2 Draw, V = V2 + D V1r D' + D V2 + V2 D'."""
    D = V2 @ ZX.T @ A2 @ Draw
    V = V2 + D @ V1robust @ D.T + D @ V2 + V2 @ D.T
    return (V + V.T) / 2


def ar_stat(s_sum, wHw, tmp, ZHw, m2VZXA, Vr, extra=0.0):
    """The Arellano-Bond z: sum_i w_i'w_{-m,i} over its standard error
    (`extra`: further terms of its variance)."""
    den = wHw + tmp @ (m2VZXA @ ZHw + Vr @ tmp) + extra
    if not np.isfinite(den) or den <= 0:
        return np.nan, np.nan
    z = s_sum / np.sqrt(den)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def diff_sargan(groups, j0, k, S, ZX, ZY, A1diag, Vfull, sargan_df, full_stat):
    """Difference-in-Sargan/Hansen tests, one per instrument group: the
    model re-estimated without the group's columns, weighted by the
    inverse of the matching submatrix of `S` (the full model's moment
    covariance, so the difference is never negative unless S is singular).
    `groups` is [(name, columns tested)]; returns a list of dicts."""
    j = S.shape[0]
    out = []
    n_unid = int((np.diag(Vfull) == 0).sum())
    for name, cols in groups:
        p = np.array([c for c in range(j) if c not in cols], np.int64)
        if len(p) == j:
            continue
        n_other = int((A1diag[p] != 0).sum())
        if not (n_other >= k and n_other < j0):
            continue
        App, _ = invsym(S[np.ix_(p, p)])
        ZXp = ZX[p]
        br, Vr = gmm(ZXp, ZY[p], App)
        if int((np.diag(Vr) == 0).sum()) != n_unid:
            continue
        Zer = ZY[p] - ZXp @ br
        stat_r = float(Zer @ App @ Zer)
        n_g = j0 - n_other - n_unid
        diff = full_stat - stat_r
        out.append(dict(group=name, excluded=stat_r, excluded_df=sargan_df - n_g,
                        excluded_p=float(stats.chi2.sf(stat_r, sargan_df - n_g)),
                        difference=diff, difference_df=n_g,
                        difference_p=float(stats.chi2.sf(diff, n_g)) if n_g > 0 else np.nan))
    return out


def wald_all(b, V, df_m, small, df_r):
    """xtabond2's model test of all the coefficients: chi2(df_m), or with
    ``small`` F(df_m, df_r)."""
    if not df_m:
        return None
    Vi, _ = invsym(V)
    w = float(b @ Vi @ b)
    if small:
        F = w / df_m
        return dict(statistic=F, df1=df_m, df2=df_r, distribution="F",
                    pvalue=float(stats.f.sf(F, df_m, df_r)))
    return dict(statistic=w, df1=df_m, df2=None, distribution="chi2",
                pvalue=float(stats.chi2.sf(w, df_m)))


def mmsc(J, n_inst, k, n):
    """Andrews and Lu's (2001) model and moment selection criteria from the
    Hansen statistic J with n_inst instruments, k coefficients and n
    observations (as pydynpd): BIC, AIC and HQIC (lower is better)."""
    over = n_inst - k
    return dict(bic=J - over * np.log(n), aic=J - 2.0 * over,
                hqic=J - 2.1 * over * np.log(np.log(n)))


class Moments:
    """The sample moments as functions of the coefficients b (the kept
    ones): the linear moments Z'(y - X b), from Z'X and Z'y, and Ahn and
    Schmidt's quadratic ones, s0 - s1 b + b' s2 b each (`s1`, `s2` over all K
    regressors, `keep` selecting b's). Everything a nonlinear GMM step
    needs, from sums, in memory."""

    def __init__(self, ZX, ZY, s0=None, s1=None, s2=None, keep=None):
        self.ZX, self.ZY = ZX, ZY
        self.j = ZX.shape[0]
        self.nl = s0 is not None and len(s0) > 0
        if self.nl:
            K = s1.shape[1]
            self.keep = np.asarray(keep)
            self.s0 = s0
            self.s1 = s1[:, self.keep]
            s2k = s2[:, self.keep][:, :, self.keep]
            self.s2 = (s2k + s2k.transpose(0, 2, 1)) / 2
            del K
        self.L = self.j + (len(s0) if self.nl else 0)

    def g(self, b):
        out = self.ZY - self.ZX @ b
        if self.nl:
            out = np.concatenate([out, self.s0 - self.s1 @ b + np.einsum("lij,i,j->l",
                                                                           self.s2, b, b)])
        return out

    def G(self, b):
        """d g / d b."""
        out = -self.ZX
        if self.nl:
            out = np.vstack([out, -self.s1 + 2 * np.einsum("lij,j->li", self.s2, b)])
        return out

    def solve(self, A, b0, cols=None, tol=1e-12, maxiter=200):
        """argmin g_p(b)' A g_p(b) over the moments `cols` (all by default),
        A their weight: in closed form if they are linear, else by
        Gauss-Newton from b0 (with step halving). Returns (b, converged)."""
        p = np.arange(self.L) if cols is None else np.asarray(cols)
        linear = (not self.nl) or np.all(p < self.j)
        if linear:
            ZX, ZY = self.ZX[p], self.ZY[p]
            XZA = ZX.T @ A
            V, _ = invsym(XZA @ ZX)
            return V @ (XZA @ ZY), True
        b = np.array(b0, float)

        def obj(bb):
            gg = self.g(bb)[p]
            return float(gg @ A @ gg)
        f = obj(b)
        for _ in range(maxiter):
            g, G = self.g(b)[p], self.G(b)[p]
            H = G.T @ A @ G
            step = np.linalg.lstsq(H, G.T @ A @ g, rcond=None)[0]
            t = 1.0
            while t > 1e-10:
                nb = b - t * step
                nf = obj(nb)
                if nf <= f + 1e-15 * abs(f):
                    break
                t /= 2
            if t <= 1e-10:
                return b, True
            b, fold, f = nb, f, nf
            if np.max(np.abs(t * step)) <= tol * (1 + np.max(np.abs(b))) or \
                    abs(fold - f) <= 1e-14 * (1 + abs(f)):
                return b, True
        return b, False


def diff_sargan_general(groups, j0, k, S, mom, A1diag, Vfull, sargan_df, full_stat, b0):
    """`diff_sargan` for moments that may be nonlinear (`mom`, a
    `Moments`): the restricted models are re-estimated by `Moments.solve`
    from b0."""
    L = S.shape[0]
    out = []
    n_unid = int((np.diag(Vfull) == 0).sum())
    for name, cols in groups:
        p = np.array([c for c in range(L) if c not in cols], np.int64)
        if len(p) == L:
            continue
        n_other = int((A1diag[p] != 0).sum())
        if not (n_other >= k and n_other < j0):
            continue
        App, _ = invsym(S[np.ix_(p, p)])
        br, _ = mom.solve(App, b0, cols=p)
        Gp = mom.G(br)[p]
        Vr, _ = invsym(Gp.T @ App @ Gp)
        if int((np.diag(Vr) == 0).sum()) != n_unid:
            continue
        gp = mom.g(br)[p]
        stat_r = float(gp @ App @ gp)
        n_g = j0 - n_other - n_unid
        diff = full_stat - stat_r
        out.append(dict(group=name, excluded=stat_r, excluded_df=sargan_df - n_g,
                        excluded_p=float(stats.chi2.sf(stat_r, sargan_df - n_g)),
                        difference=diff, difference_df=n_g,
                        difference_p=float(stats.chi2.sf(diff, n_g)) if n_g > 0 else np.nan))
    return out
