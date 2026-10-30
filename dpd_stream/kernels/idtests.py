"""The pass for the identification tests: per-panel scores for the
Kleibergen-Paap statistic."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .panel import _build, _cluster_ok, _dvec, _scratch


@njit(parallel=True, cache=True)
def idtest_pass(cbounds, pstart, tt, B, W, use_w, T, system, orth,
                term_base, term_k, term_diff, ty, tx, cmp_terms,
                iv_term, iv_eq, iv_pass, iv_mz,
                g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                maxnz, cmap, zmap, L1, pmap, Pg, poff, ycols, x2cols, Gz, Gy, Pi,
                AG, Mz1, Mx, masked, CLacc, rowmode, U, V, AccU, AccV, SRU, SRV):
    """The scores of the identification tests, per panel and row:
    vec(z1~ y~') and vec(z1~ v'), with z1 the excluded instruments, y the
    endogenous regressors (`ycols`; index K is the outcome, for the
    Anderson-Rubin test), both net of the included exogenous ones x2 (`x2cols`)
    by the pooled regressions Gz, Gy (x2 -> z1, y), and v = y~ - Pi' z1~ the
    first-stage residuals. vec stacks the columns (one per regressor).

    The excluded instruments are the live columns `zmap` maps (cmap: raw ->
    live), and with PCA the components: live GMM-style column c adds its
    value times Pg[pmap[c]] to the components, from position `poff`. With
    weights the rows are those of the weighted regressions (z, x times the
    square root of the weight). With an absorbed effect (`AG` nonempty), the
    rows are net of it: z and x less their cells' means Mz1 (in z1's
    coordinates) and Mx, through the row's dummies (`_dvec`).

    Per panel: U[p], V[p] (rows with a cluster only, if `masked`); per
    cluster code of the general clusterings: AccU, AccV; with `rowmode`
    (clustering by observation), per chunk the outer products SRU, SRV of
    each period's scores (both equations' rows together)."""
    absorb = AG.shape[0] > 0
    nch = cbounds.shape[0] - 1
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    ngt = g_terms.shape[0]
    Ke = ycols.shape[0]
    k2 = x2cols.shape[0]
    ncomp = Pg.shape[1] if pmap.shape[0] > 0 else 0
    m = L1 * Ke
    p0 = cbounds[0]
    for ch in prange(nch):
        (BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT, IVT, GS, GD,
         Xr, Yr, Wr, use, idx, val, nz) = _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz)
        z = np.zeros(max(L1, 1))
        x = np.zeros(K + 1)
        x2v = np.zeros(max(k2, 1))
        y = np.zeros(max(Ke, 1))
        v = np.zeros(max(Ke, 1))
        su = np.zeros((R, max(m, 1)))
        sv = np.zeros((R, max(m, 1)))
        tu = np.zeros(max(m, 1))
        tv = np.zeros(max(m, 1))
        didx = np.zeros(T, np.int64)
        dval = np.zeros(T)
        for p in range(cbounds[ch], cbounds[ch + 1]):
            q = p - p0
            for c in range(m):
                U[q, c] = 0.0
                V[q, c] = 0.0
            n_in = _build(pstart[p], pstart[p + 1], tt, B, W, use_w, T, system, orth,
                          term_base, term_k, term_diff, ty, tx, cmp_terms,
                          iv_term, iv_eq, iv_pass, iv_mz,
                          g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                          lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                          BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT,
                          IVT, GS, GD, Xr, Yr, Wr, use, idx, val, nz)
            if n_in == 0:
                continue
            cell0 = AG[pstart[p]] * T if absorb else 0
            for r in range(R):
                for c in range(m):
                    su[r, c] = 0.0
                    sv[r, c] = 0.0
                if not use[r]:
                    continue
                sw = np.sqrt(Wr[r])
                for c in range(L1):
                    z[c] = 0.0
                for a in range(nz[r]):
                    c = cmap[idx[r, a]]
                    if c < 0:
                        continue
                    if zmap[c] >= 0:
                        z[zmap[c]] += val[r, a]
                    if ncomp > 0 and pmap[c] >= 0:
                        for cc in range(ncomp):
                            z[poff + cc] += val[r, a] * Pg[pmap[c], cc]
                for k in range(K):
                    x[k] = Xr[r, k] / Wr[r] if Wr[r] > 0.0 else 0.0
                x[K] = Yr[r] / Wr[r] if Wr[r] > 0.0 else 0.0
                if absorb and r < T:
                    nd = _dvec(r, T, orth, complete, didx, dval)
                    for e2 in range(nd):
                        cell = cell0 + didx[e2]
                        for c in range(L1):
                            z[c] -= dval[e2] * Mz1[cell, c]
                        for k in range(K + 1):
                            x[k] -= dval[e2] * Mx[cell, k]
                for a in range(k2):
                    x2v[a] = x[x2cols[a]] * sw
                for c in range(L1):
                    s = z[c] * sw
                    for a in range(k2):
                        s -= Gz[a, c] * x2v[a]
                    z[c] = s
                for k in range(Ke):
                    s = x[ycols[k]] * sw
                    for a in range(k2):
                        s -= Gy[a, k] * x2v[a]
                    y[k] = s
                    f = 0.0
                    for c in range(L1):
                        f += z[c] * Pi[c, k]
                    v[k] = s - f
                for k in range(Ke):
                    for c in range(L1):
                        su[r, k * L1 + c] = z[c] * y[k]
                        sv[r, k * L1 + c] = z[c] * v[k]
            for r in range(R):
                if not use[r]:
                    continue
                t = r if r < T else r - T
                okc = (not masked) or _cluster_ok(t, T, system, use, rowmap)
                if not okc:
                    continue
                for c in range(m):
                    U[q, c] += su[r, c]
                    V[q, c] += sv[r, c]
                if CLacc.shape[1] > 0:
                    r0 = rowmap[t]
                    for qq in range(CLacc.shape[1]):
                        cc = CLacc[r0, qq]
                        if cc >= 0:
                            for c in range(m):
                                AccU[ch, cc, c] += su[r, c]
                                AccV[ch, cc, c] += sv[r, c]
            if rowmode:
                for t in range(T):
                    if not _cluster_ok(t, T, system, use, rowmap):
                        continue
                    for c in range(m):
                        tu[c] = 0.0
                        tv[c] = 0.0
                    for e2 in range(2 if system else 1):
                        r = t + e2 * T
                        if use[r]:
                            for c in range(m):
                                tu[c] += su[r, c]
                                tv[c] += sv[r, c]
                    for c1 in range(m):
                        if tu[c1] != 0.0 or tv[c1] != 0.0:
                            for c2 in range(m):
                                SRU[ch, c1, c2] += tu[c1] * tu[c2]
                                SRV[ch, c1, c2] += tv[c1] * tv[c2]
