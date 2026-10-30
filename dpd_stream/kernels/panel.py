"""One panel at a time: `_build` turns a panel's rows (its base columns on the
time grid) into the rows of the stacked system, transformed equation then
levels, with each row's regressors, outcome, weight, whether it is in the
sample and its instruments as a sparse list of (column, value). The helpers
every pass kernel shares."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _xform(v, complete, orth, out):
    """First differences, or forward orthogonal deviations stored a period
    later (missing where not defined)."""
    T = v.shape[0]
    for t in range(T):
        out[t] = np.nan
    if not orth:
        for t in range(1, T):
            out[t] = v[t] - v[t - 1]
        return
    n = 0
    s = 0.0
    for t in range(T - 1, -1, -1):
        if n > 0 and t + 1 < T:
            out[t + 1] = np.sqrt(n / (n + 1.0)) * (v[t] - s / n)
        if complete[t]:
            n += 1
            s += v[t]


@njit(cache=True)
def _bod(v, complete, out):
    """Backward orthogonal deviations, in place, missing as 0."""
    T = v.shape[0]
    n = 0
    s = 0.0
    for t in range(T):
        out[t] = 0.0
        if n > 0:
            val = np.sqrt(n / (n + 1.0)) * (v[t] - s / n)
            if not np.isnan(val):
                out[t] = val
        if complete[t]:
            n += 1
            s += v[t]


@njit(cache=True)
def _build(lo, hi, tt, B, W, use_w, T, system, orth,
           term_base, term_k, term_diff, ty, tx, cmp_terms,
           iv_term, iv_eq, iv_pass, iv_mz,
           g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
           lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
           BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT, IVT, GS, GD,
           Xr, Yr, Wr, use, idx, val, nz):
    """Panel rows lo..hi-1 -> the stacked rows (see the module docstring).
    Returns the number of rows in the sample."""
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    for t in range(T):
        present[t] = False
        rowmap[t] = -1
        wgrid[t] = 1.0
        for b in range(nbase):
            BG[t, b] = np.nan
    for r in range(lo, hi):
        t = tt[r]
        present[t] = True
        rowmap[t] = r
        if use_w:
            wgrid[t] = W[r]
        for b in range(nbase):
            BG[t, b] = B[r, b]
    if use_w:
        # a period the panel lacks has weight 0 (a forward orthogonal
        # deviation stored at a gap is then in the sample but weightless, as in
        # xtabond2)
        for t in range(T):
            if not present[t]:
                wgrid[t] = 0.0
    # every series, on the grid. A term exists only where the panel has an
    # observation, as a Stata variable does (L.x at a gap is missing, not
    # x's previous value); the constant is 1 throughout, as xtabond2's
    for j in range(nt):
        b = term_base[j]
        k = term_k[j]
        for t in range(T):
            if b < 0:
                TV[t, j] = 1.0
                continue
            s = t - k
            if not present[t] or s < 0 or s >= T:
                TV[t, j] = np.nan
            elif term_diff[j]:
                TV[t, j] = BG[s, b] - BG[s - 1, b] if s >= 1 else np.nan
            else:
                TV[t, j] = BG[s, b]
    for t in range(T):
        ok = not np.isnan(TV[t, ty])
        for k in range(K):
            if np.isnan(TV[t, tx[k]]):
                ok = False
        c = ok
        for q in range(cmp_terms.shape[0]):
            if np.isnan(TV[t, cmp_terms[q]]):
                c = False
        complete[t] = c
        w_ok = True
        if use_w:
            w_ok = present[t] and (not np.isnan(wgrid[t])) and wgrid[t] > 0
        touse0[t] = present[t] and ok and w_ok
    # transformed regressors, outcome, IV-style instruments
    for k in range(K):
        for t in range(T):
            s1[t] = TV[t, tx[k]]
        _xform(s1, complete, orth, s2)
        for t in range(T):
            XT[t, k] = s2[t]
    for t in range(T):
        s1[t] = TV[t, ty]
    _xform(s1, complete, orth, YT)
    for c in range(jiv):
        for t in range(T):
            s1[t] = TV[t, iv_term[c]]
        if iv_pass[c]:
            for t in range(T):
                IVT[t, c] = s1[t]
        else:
            _xform(s1, complete, orth, s2)
            for t in range(T):
                IVT[t, c] = s2[t]
    # GMM-style sources: levels (or backward orthogonal deviations), and differences
    for g in range(g_nb.shape[0]):
        for v in range(g_nb[g]):
            p = g_t0[g] + v
            tj = g_terms[p]
            for t in range(T):
                s1[t] = TV[t, tj]
            if g_orth[g]:
                _bod(s1, complete, s2)
                for t in range(T):
                    GS[t, p] = s2[t]
            else:
                for t in range(T):
                    GS[t, p] = 0.0 if np.isnan(s1[t]) else s1[t]
            GD[0, p] = 0.0
            for t in range(1, T):
                d = s1[t] - s1[t - 1]
                GD[t, p] = 0.0 if np.isnan(d) else d
    n_in = 0
    for r in range(R):
        use[r] = False
        nz[r] = 0
        eqrow = 0 if r < T else 1
        t = r - eqrow * T
        if eqrow == 0:
            if orth:
                ok = t >= 1 and touse0[t - 1]
            else:
                ok = touse0[t]
            if not ok or np.isnan(YT[t]):
                continue
            for k in range(K):
                if np.isnan(XT[t, k]):
                    ok = False
            if not ok:
                continue
        else:
            if not touse0[t]:
                continue
            ok = True
        w = wgrid[t]
        if np.isnan(w):
            continue
        # IV-style instruments
        n = 0
        for c in range(jiv):
            if eqrow == 0:
                if iv_eq[c] == 0:
                    continue
                v = IVT[t, c]
            else:
                if iv_eq[c] == 1:
                    continue
                v = TV[t, iv_term[c]]
            if np.isnan(v):
                if iv_mz[c]:
                    continue
                ok = False
                break
            if v != 0.0:
                idx[r, n] = c
                val[r, n] = v
                n += 1
        if not ok:
            continue
        # GMM-style instruments
        for g in range(g_nb.shape[0]):
            nb = g_nb[g]
            if g_eqrow[g] == eqrow:
                for li in range(g_b[g] - g_a[g] + 1):
                    lag = g_a[g] + li
                    p = g_lp[g] + li
                    s0 = t - lag
                    if s0 < lag_lo[p] - 1 or s0 > lag_hi[p] - 1:
                        continue
                    for v in range(nb):
                        x = GD[s0, g_t0[g] + v] if g_fd[g] else GS[s0, g_t0[g] + v]
                        if x != 0.0:
                            if g_coll[g]:
                                col = lag_col[p] + v
                            else:
                                col = lag_col[p] + (s0 - (lag_lo[p] - 1)) * nb + v
                            idx[r, n] = jiv + col
                            val[r, n] = x
                            n += 1
            if eqrow == 1 and g_extra[g]:
                # g_xsh: a collapsed column moved to deeper lags (xtabond2's
                # search, when its first lag is empty in the sample)
                for v in range(nb):
                    s0 = t - g_xlag[g] - g_xsh[g, v]
                    if s0 < g_xlo[g] - 1 or s0 > g_xhi[g] - 1:
                        continue
                    x = GD[s0, g_t0[g] + v]
                    if x != 0.0:
                        if g_coll[g]:
                            col = g_xcol[g] + v
                        else:
                            col = g_xcol[g] + (s0 - (g_xlo[g] - 1)) * nb + v
                        idx[r, n] = jiv + col
                        val[r, n] = x
                        n += 1
        nz[r] = n
        use[r] = True
        n_in += 1
        Wr[r] = w
        if eqrow == 0:
            Yr[r] = YT[t] * w
            for k in range(K):
                Xr[r, k] = XT[t, k] * w
        else:
            Yr[r] = TV[t, ty] * w
            for k in range(K):
                Xr[r, k] = TV[t, tx[k]] * w
    return n_in


@njit(cache=True)
def _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz):
    return (np.empty((T, nbase)), np.empty((T, nt)), np.empty(T, np.bool_),
            np.empty(T, np.bool_), np.empty(T, np.bool_), np.empty(T), np.empty(T, np.int64),
            np.empty(T), np.empty(T), np.empty((T, max(K, 1))), np.empty(T),
            np.empty((T, max(jiv, 1))), np.empty((T, max(ngt, 1))), np.empty((T, max(ngt, 1))),
            np.empty((R, max(K, 1))), np.empty(R), np.empty(R), np.empty(R, np.bool_),
            np.empty((R, max(maxnz, 1)), np.int64), np.empty((R, max(maxnz, 1))),
            np.empty(R, np.int64))


@njit(cache=True)
def _cluster_ok(t, T, system, use, rowmap):
    """Rows at grid period t have a cluster when the main equation's row t
    is in the sample (xtabond2 assigns cluster ids to its sample)."""
    m0 = T if system else 0
    return use[m0 + t] and rowmap[t] >= 0


@njit(cache=True)
def _dvec(r, T, orth, complete, didx, dval):
    """The absorbed effects' dummies in row r of the transformed equation
    (difference GMM), as periods (didx) and values (dval); returns their
    number. First differences: the period r itself (the indicator of the
    row's period spans the differenced dummies). Forward orthogonal
    deviations (row r holds the deviation of period r - 1): the deviation of
    the period dummies, c (1[s = r - 1] - 1[s later and complete] / n), as
    `_xform` computes it."""
    if not orth:
        didx[0] = r
        dval[0] = 1.0
        return 1
    t0 = r - 1
    n = 0
    for s in range(t0 + 1, T):
        if complete[s]:
            n += 1
    if t0 < 0 or n == 0:
        return 0
    c = np.sqrt(n / (n + 1.0))
    didx[0] = t0
    dval[0] = c
    m = 1
    for s in range(t0 + 1, T):
        if complete[s]:
            didx[m] = s
            dval[m] = -c / n
            m += 1
    return m
