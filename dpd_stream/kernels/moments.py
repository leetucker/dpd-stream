"""The passes over the data for the moments, the residuals and the
Windmeijer correction: `pass1` (the sums), `resid_pass` (at an estimate: the
moment covariance, the Arellano-Bond sums) and `windmeijer_pass`."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .nl import _nl_panel
from .panel import _build, _cluster_ok, _dvec, _scratch


@njit(parallel=True, cache=True)
def pass1(cbounds, pstart, tt, B, W, use_w, T, system, orth,
          term_base, term_k, term_diff, ty, tx, cmp_terms, iv_term, iv_eq, iv_pass, iv_mz,
          g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
          lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
          maxnz, h_ptr, h_idx, h_val, CL, clpres, do_pca, jg0,
          ZX, ZY, S1, Q, XX, colnz, cnt, ZZp, Zsp, xnz, g_xstop):
    """The moments that don't depend on the estimates.

    Per chunk ch: ZX[ch] = Z'X, ZY[ch] = Z'y, S1[ch] = sum_i Z_i' W^.5 H
    W^.5 Z_i, Q[ch, e] = sum (1/w) [X y]'[X y] over equation e's rows (for
    sigma^2), XX[ch, e] = sum [X y]'[X y] (collinearity), colnz[ch, c]
    whether column c is nonzero in the sample, cnt[ch] = (obs in the main
    equation, groups, min and max obs per group, obs in the transformed
    equation, sum of the main equation's weights, panels, sum of all sample
    rows' weights), xnz[ch, g, v, l] whether collapsed group g's levels
    instrument for variable v at l lags deeper than its first is nonzero in
    the sample (for xtabond2's search; up to its last lag g_xstop), ZZp/Zsp
    the (weighted) moments for PCA, of the
    GMM-style columns from jg0, or with jg0 = -1 of all the columns (for the
    identification tests, and PCA), and clpres[q, code] the clusters
    present (in the main equation's sample).
    """
    nch = cbounds.shape[0] - 1
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    ngt = g_terms.shape[0]
    for ch in prange(nch):
        (BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT, IVT, GS, GD,
         Xr, Yr, Wr, use, idx, val, nz) = _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz)
        vr = np.empty(K + 1)
        cnt[ch, 2] = 1e18
        cnt[ch, 3] = 0
        for p in range(cbounds[ch], cbounds[ch + 1]):
            n_in = _build(pstart[p], pstart[p + 1], tt, B, W, use_w, T, system, orth,
                          term_base, term_k, term_diff, ty, tx, cmp_terms,
                          iv_term, iv_eq, iv_pass, iv_mz,
                          g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                          lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                          BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT,
                          IVT, GS, GD, Xr, Yr, Wr, use, idx, val, nz)
            if n_in == 0:
                continue
            m0 = T if system else 0
            nmain = 0
            for t in range(T):
                if use[m0 + t]:
                    nmain += 1
                    cnt[ch, 5] += Wr[m0 + t]
                    r0 = rowmap[t]
                    if r0 >= 0:
                        for q in range(CL.shape[1]):
                            c = CL[r0, q]
                            if c >= 0:
                                clpres[q, c] = True
                if use[t]:
                    cnt[ch, 4] += 1
                    cnt[ch, 8] += Wr[t]
            if nmain > 0:
                cnt[ch, 0] += nmain
                cnt[ch, 1] += 1
                if nmain < cnt[ch, 2]:
                    cnt[ch, 2] = nmain
                if nmain > cnt[ch, 3]:
                    cnt[ch, 3] = nmain
            for g in range(g_nb.shape[0]):
                if not (system and g_extra[g] and g_coll[g]):
                    continue
                for t in range(T):
                    if not use[T + t]:
                        continue
                    for v in range(g_nb[g]):
                        for lsh in range(min(xnz.shape[3], g_xstop[g] - g_xlag[g] + 1)):
                            s0 = t - g_xlag[g] - lsh
                            if s0 < g_xlo[g] - 1 or s0 > g_xhi[g] - 1:
                                continue
                            if GD[s0, g_t0[g] + v] != 0.0:
                                xnz[ch, g, v, lsh] = True
            for r in range(R):
                if not use[r]:
                    continue
                cnt[ch, 7] += Wr[r] if use_w else 1.0
                e = 0 if r < T else 1
                for k in range(K):
                    vr[k] = Xr[r, k]
                vr[K] = Yr[r]
                iw = 1.0 / Wr[r] if Wr[r] > 0.0 else 0.0
                for a in range(K + 1):
                    for b2 in range(K + 1):
                        XX[ch, e, a, b2] += vr[a] * vr[b2]
                        Q[ch, e, a, b2] += vr[a] * vr[b2] * iw
                for a in range(nz[r]):
                    c = idx[r, a]
                    v = val[r, a]
                    colnz[ch, c] = True
                    ZY[ch, c] += v * Yr[r]
                    for k in range(K):
                        ZX[ch, c, k] += v * Xr[r, k]
                # S1: H's nonzeros in this row
                sw_r = np.sqrt(Wr[r])
                for hp in range(h_ptr[r], h_ptr[r + 1]):
                    s = h_idx[hp]
                    if not use[s]:
                        continue
                    coef = h_val[hp] * sw_r * np.sqrt(Wr[s])
                    for a in range(nz[r]):
                        ca = idx[r, a]
                        va = val[r, a] * coef
                        for b2 in range(nz[s]):
                            S1[ch, ca, idx[s, b2]] += va * val[s, b2]
                if do_pca:
                    w = Wr[r] if use_w else 1.0
                    off = jg0 if jg0 >= 0 else 0
                    for a in range(nz[r]):
                        ca = idx[r, a] - off
                        if ca < 0:
                            continue
                        va = val[r, a]
                        Zsp[ch, ca] += w * va
                        for b2 in range(nz[r]):
                            cb = idx[r, b2] - off
                            if cb >= 0:
                                ZZp[ch, ca, cb] += w * va * val[r, b2]
            cnt[ch, 6] += 1


@njit(parallel=True, cache=True)
def resid_pass(cbounds, pstart, tt, B, W, use_w, T, system, orth,
               term_base, term_k, term_diff, ty, tx, cmp_terms, iv_term, iv_eq, iv_pass, iv_mz,
               g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
               lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
               maxnz, cmap, j, b, masked, rowmode, CLacc, Acc,
               ar_mode, artests, keepx, Hd, Psi, wsc,
               G, SR, ARs, ARwHw, ARtmp, ARzhw,
               AG, Mz, delta, xshift, SSE, Cross, xpf, xrf, nlcoll, nlmap, Lnl, nlkind,
               ARtd, ARzhd):
    """Residuals at `b` and what they give, per chunk ch.

    G[p - pstart0]: panel p's Z_i'e_i (rows with a cluster only, if
    `masked`); SR[ch]: sum over grid periods of g_t g_t' (`rowmode`:
    clustering by observation); Acc[ch, code]: Z_c'e_c per cluster of each
    general clustering (codes in CLacc, offset per clustering).

    With weights, `W` is normalized (xtabond2's weights) and `wsc` is the
    normalization, to recover the raw weights where xtabond2 uses those.

    The Arellano-Bond tests (`ar_mode` 1: robust, 2: one-step non-robust):
    per panel and lag m, ARs[p, m] = sum_t w_t w_{t-m}; ARwHw[ch, m],
    ARtmp[ch, m] = X'w_{-m}, and ARzhw[ch, m] = sum_i Z_i'e_i s_im (robust)
    or sum_i Z_i' Psi' w_{-m} (one-step).

    With an absorbed effect (difference GMM: `AG`, each row's group code,
    nonempty), row r of panel i has the dummies d_r of the cells
    (g_i, s) (`_dvec`): its residual is net of w_r d_r'`delta`, its
    instruments of d_r'`Mz` (Z~ = Z - D Mz), its regressors (for the AR
    tests) of w_r d_r'`xshift` (the derivative of the concentrated
    residual); SSE[ch] gets the sum of squared residuals (over the weights)
    and, if it has room for them, Cross[ch, cell] the cross moments of the
    cells' moments D'e and the instruments' Z~'e: by panel times `xpf`, and
    by observation (`rowmode`) times `xrf`. The Arellano-Bond tests' terms
    in the effects, by cell: ARtd[ch, m] the derivative (as ARtmp's), and
    ARzhd[ch, m] the cells' moments' (as ARzhw's).

    With `Lnl` Ahn-Schmidt moments, G's columns j.. j+Lnl-1 hold each
    panel's (see `_nl_panel`), and the AR tests' sums include them.
    """
    nch = cbounds.shape[0] - 1
    absorb = AG.shape[0] > 0
    want_cross = Cross.shape[1] > 1
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    ngt = g_terms.shape[0]
    p0 = cbounds[0]
    for ch in prange(nch):
        (BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT, IVT, GS, GD,
         Xr, Yr, Wr, use, idx, val, nz) = _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz)
        e = np.zeros(R)
        g = np.zeros(j)
        gm = np.zeros(j)
        gt = np.zeros(j)
        wv = np.zeros(T)
        wl = np.zeros(T)
        pX = np.zeros((T, K))
        gnl = np.zeros(max(Lnl, 1))
        Jnl = np.zeros((max(Lnl, 1), max(K, 1)))
        xs_nl = np.zeros(max(K, 1))
        dn = np.zeros(R, np.int64)
        di = np.zeros((R, T), np.int64)
        dv = np.zeros((R, T))
        zm = np.zeros((R if absorb else 1, j))
        de = np.zeros(T)
        du = np.zeros(T)
        for p in range(cbounds[ch], cbounds[ch + 1]):
            n_in = _build(pstart[p], pstart[p + 1], tt, B, W, use_w, T, system, orth,
                          term_base, term_k, term_diff, ty, tx, cmp_terms,
                          iv_term, iv_eq, iv_pass, iv_mz,
                          g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                          lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                          BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT,
                          IVT, GS, GD, Xr, Yr, Wr, use, idx, val, nz)
            q = p - p0
            for c in range(Lnl):
                gnl[c] = 0.0
                G[q, j + c] = 0.0
            if Lnl > 0 and n_in > 0:
                _nl_panel(TV, touse0, ty, tx, b, T, nlcoll, nlmap, gnl, Jnl, nlkind, xs_nl)
                for c in range(Lnl):
                    G[q, j + c] = gnl[c]
            if n_in == 0:
                for c in range(j):
                    G[q, c] = 0.0
                for m in range(artests):
                    ARs[q, m] = 0.0
                continue
            for c in range(j):
                g[c] = 0.0
                gm[c] = 0.0
            cell0 = AG[pstart[p]] * T if absorb else 0
            for s in range(T):
                de[s] = 0.0
                du[s] = 0.0
            for r in range(R):
                dn[r] = 0
                if not use[r]:
                    e[r] = 0.0
                    continue
                er = Yr[r]
                for k in range(K):
                    er -= Xr[r, k] * b[k]
                if absorb:
                    dn[r] = _dvec(r, T, orth, complete, di[r], dv[r])
                    for c in range(j):
                        zm[r, c] = 0.0
                    for e2 in range(dn[r]):
                        cell = cell0 + di[r, e2]
                        er -= Wr[r] * dv[r, e2] * delta[cell]
                        for c in range(j):
                            zm[r, c] += dv[r, e2] * Mz[cell, c]
                    SSE[ch] += (er * er / Wr[r] if Wr[r] > 0.0 else 0.0) if use_w else er * er
                e[r] = er
                t = r if r < T else r - T
                okc = (not masked) or _cluster_ok(t, T, system, use, rowmap)
                for a in range(nz[r]):
                    c = cmap[idx[r, a]]
                    if c >= 0:
                        g[c] += val[r, a] * er
                        if okc:
                            gm[c] += val[r, a] * er
                if absorb:
                    for c in range(j):
                        g[c] -= zm[r, c] * er
                        if okc:
                            gm[c] -= zm[r, c] * er
                    for e2 in range(dn[r]):
                        du[di[r, e2]] += dv[r, e2] * er
                        if okc:
                            de[di[r, e2]] += dv[r, e2] * er
                if okc and CLacc.shape[1] > 0:
                    r0 = rowmap[t]
                    for qq in range(CLacc.shape[1]):
                        cc = CLacc[r0, qq]
                        if cc >= 0:
                            for a in range(nz[r]):
                                c = cmap[idx[r, a]]
                                if c >= 0:
                                    Acc[ch, cc, c] += val[r, a] * er
                            if absorb:
                                for c in range(j):
                                    Acc[ch, cc, c] -= zm[r, c] * er
            for c in range(j):
                G[q, c] = gm[c] if masked else g[c]
            if absorb and want_cross and xpf != 0.0:
                for s in range(T):
                    if de[s] != 0.0:
                        for c in range(j):
                            Cross[ch, cell0 + s, c] += xpf * de[s] * G[q, c]
            if rowmode:
                for t in range(T):
                    if not _cluster_ok(t, T, system, use, rowmap):
                        continue
                    for c in range(j):
                        gt[c] = 0.0
                    for e2 in range(2 if system else 1):
                        r = t + e2 * T
                        if use[r]:
                            for a in range(nz[r]):
                                c = cmap[idx[r, a]]
                                if c >= 0:
                                    gt[c] += val[r, a] * e[r]
                            if absorb:
                                for c in range(j):
                                    gt[c] -= zm[r, c] * e[r]
                    for c1 in range(j):
                        if gt[c1] != 0.0:
                            for c2 in range(j):
                                SR[ch, c1, c2] += gt[c1] * gt[c2]
                    if absorb and want_cross and xrf != 0.0:
                        for e2 in range(dn[t]):
                            f = xrf * dv[t, e2] * e[t]
                            for c in range(j):
                                Cross[ch, cell0 + di[t, e2], c] += f * gt[c]
            if ar_mode == 0:
                continue
            # the residuals tested: in first differences
            if orth:
                # differences of the levels residuals, missing as 0
                for t in range(T):
                    lev = TV[t, ty]
                    for k in range(K):
                        if keepx[k]:
                            lev -= TV[t, tx[k]] * b[k]
                    if absorb:
                        lev -= delta[cell0 + t]     # the level effect of period t
                    s1[t] = lev
                for t in range(T):
                    wv[t] = 0.0
                    for k in range(K):
                        pX[t, k] = 0.0
                for t in range(1, T):
                    d = s1[t] - s1[t - 1]
                    if not np.isnan(d):
                        wv[t] = d
                    for k in range(K):
                        if keepx[k]:
                            dx = TV[t, tx[k]] - TV[t - 1, tx[k]]
                            if absorb:
                                dx -= xshift[cell0 + t, k] - xshift[cell0 + t - 1, k]
                            if not np.isnan(dx):
                                pX[t, k] = dx * (wgrid[t] / wsc if use_w else 1.0)
                for t in range(T):
                    wl[t] = wv[t]
                    if use_w:
                        wv[t] = wv[t] * wgrid[t] / wsc
                        if np.isnan(wv[t]):
                            wv[t] = 0.0
            else:
                for t in range(T):
                    wv[t] = e[t] if use[t] else 0.0
                    wl[t] = wv[t]
                    if use[t] and use_w:
                        wl[t] = wv[t] * wsc / Wr[t] if Wr[t] > 0.0 else 0.0
                    for k in range(K):
                        pX[t, k] = 0.0
                        if use[t]:
                            pX[t, k] = Xr[t, k] - (Wr[t] * xshift[cell0 + t, k] if absorb
                                                   else 0.0)
            for m in range(artests):
                # lag wl by one more period
                for t in range(T - 1, 0, -1):
                    wl[t] = wl[t - 1]
                wl[0] = 0.0
                s = 0.0
                for t in range(T):
                    s += wv[t] * wl[t]
                ARs[q, m] = s
                for t in range(T):
                    if wl[t] != 0.0:
                        for k in range(K):
                            ARtmp[ch, m, k] += pX[t, k] * wl[t]
                if absorb:
                    # the derivative in the effects: their dummies' (in first
                    # differences, of the transformed rows; with orthogonal
                    # deviations, the differences of the level dummies)
                    if orth:
                        for t in range(1, T):
                            if wl[t] != 0.0 and present[t] and present[t - 1]:
                                f = wl[t] * (wgrid[t] / wsc if use_w else 1.0)
                                ARtd[ch, m, cell0 + t] += f
                                ARtd[ch, m, cell0 + t - 1] -= f
                    else:
                        for t in range(T):
                            if wl[t] != 0.0 and use[t]:
                                for e2 in range(dn[t]):
                                    ARtd[ch, m, cell0 + di[t, e2]] += Wr[t] * dv[t, e2] * wl[t]
                if ar_mode == 1:
                    ARwHw[ch, m] += s * s
                    if s != 0.0:
                        for c in range(j):
                            ARzhw[ch, m, c] += g[c] * s
                        for c in range(Lnl):
                            ARzhw[ch, m, j + c] += gnl[c] * s
                        if absorb:
                            for t in range(T):
                                ARzhd[ch, m, cell0 + t] += du[t] * s
                else:
                    # one-step non-robust: sigma^2 H and psi in place of the
                    # outer products (wl masked by the sample, weighted)
                    for t in range(T):
                        wli = wl[t] if use[t] else 0.0
                        if wli == 0.0:
                            continue
                        for t2 in range(T):
                            wl2 = wl[t2] if use[t2] else 0.0
                            if wl2 != 0.0 and Hd[t2, t] != 0.0:
                                ARwHw[ch, m] += wli * Hd[t2, t] * (Wr[t2] if use_w else 1.0) \
                                    * wl2
                    for r in range(R):
                        if not use[r]:
                            continue
                        ps = 0.0
                        for t in range(T):
                            wli = wl[t] if use[t] else 0.0
                            if wli != 0.0 and Psi[t, r] != 0.0:
                                ps += Psi[t, r] * (Wr[t] if use_w else 1.0) * wli
                        if ps != 0.0:
                            for a in range(nz[r]):
                                c = cmap[idx[r, a]]
                                if c >= 0:
                                    ARzhw[ch, m, c] += val[r, a] * ps
                            for e2 in range(dn[r]):
                                ARzhd[ch, m, cell0 + di[r, e2]] += dv[r, e2] * ps
                            if absorb:
                                for c in range(j):
                                    ARzhw[ch, m, c] -= zm[r, c] * ps


@njit(parallel=True, cache=True)
def windmeijer_pass(cbounds, pstart, tt, B, W, use_w, T, system, orth,
                    term_base, term_k, term_diff, ty, tx, cmp_terms,
                    iv_term, iv_eq, iv_pass, iv_mz,
                    g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                    lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                    maxnz, cmap, j, b1, v, panel_f, masked, row_f, CLacc, alpha,
                    D, U, AG, Mz, Mzv, delta, xshift, Dcell, nlcoll, nlmap, Lnl, Dnl,
                    nlkind, Lf, Rf, Gd, cfc, Ob, Od):
    """The sums behind Windmeijer's (2005) correction, at the one-step
    residuals e1 and v = A2 Z'e2. For each clustering c with factor f_c
    (sign, small-sample), xtabond2's

        sum_c f_c [ (e1_c'Z_c v) Z_c'X_c + Z_c'e1_c (v'Z_c'X_c) ].

    Per chunk ch: D[ch] gets the panel (`panel_f`) and observation
    (`row_f`) clusterings' terms whole, and the first term of the general
    clusterings, whose coefficients `alpha` (already f_c e1_c'Z_c v, per
    cluster code) come from their stored sums; U[ch, code] = Z_c'X_c' v for
    their second term, completed outside.

    With an absorbed effect (`AG` nonempty; see `resid_pass`), the
    instruments are Z~ = Z - D Mz, the residuals net of the effects
    `delta`, and the regressors x^ = x - w d'`xshift` (the concentrated
    residual's derivative); Mzv = Mz v, and the terms in the cells' means
    of Z of the first sum go to Dcell[ch, cell] = sum coef_r d_r x^_r, for
    D -= Mz' Dcell outside.

    With an absorbed effect, the same sums with the effects' dummies in
    place of the regressors, and the cells' moments D'e1 joining the
    instruments', contracted with `nf` functionals (lambda_f, rho_f) of the
    moments, lambda_f (`Lf`) on Z~'e and rho_f (`Rf`) on D'e: each row's
    lambda_f'z~_r + rho_f'd_r (`Gd` = Rf - Lf Mz') and each cluster's
    lambda_f'Z~_c'e1_c + rho_f'D_c'e1_c (given, times the clustering's
    factor, for the general clusterings: `cfc`, by code). Ob[ch, f] gets
    the regressors' columns (x^), Od[ch, fu, cell] the dummies'.

    With `Lnl` Ahn-Schmidt moments (by panel; v's last Lnl entries are
    theirs), their part of each panel's moments and of the derivative join
    the terms, and their rows of the result go to Dnl[ch].
    """
    absorb = AG.shape[0] > 0
    nf = Lf.shape[0]
    nch = cbounds.shape[0] - 1
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    ngt = g_terms.shape[0]
    for ch in prange(nch):
        (BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT, IVT, GS, GD,
         Xr, Yr, Wr, use, idx, val, nz) = _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz)
        e = np.zeros(R)
        g = np.zeros(j)
        gt = np.zeros(j)
        u = np.zeros(K)
        ut = np.zeros(K)
        coef = np.zeros(R)
        vz = np.zeros(R)
        xh = np.zeros((R, max(K, 1)))
        gnl = np.zeros(max(Lnl, 1))
        Jnl = np.zeros((max(Lnl, 1), max(K, 1)))
        xs_nl = np.zeros(max(K, 1))
        dn = np.zeros(R, np.int64)
        di = np.zeros((R, T), np.int64)
        dv = np.zeros((R, T))
        zm = np.zeros((R if absorb else 1, j))
        af = np.zeros((R, max(nf, 1)))
        de = np.zeros(T)
        ud = np.zeros(T)
        for p in range(cbounds[ch], cbounds[ch + 1]):
            n_in = _build(pstart[p], pstart[p + 1], tt, B, W, use_w, T, system, orth,
                          term_base, term_k, term_diff, ty, tx, cmp_terms,
                          iv_term, iv_eq, iv_pass, iv_mz,
                          g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                          lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                          BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT,
                          IVT, GS, GD, Xr, Yr, Wr, use, idx, val, nz)
            if n_in == 0:
                continue
            for c in range(j):
                g[c] = 0.0
            for k in range(K):
                u[k] = 0.0
            for s in range(T):
                de[s] = 0.0
                ud[s] = 0.0
            cell0 = AG[pstart[p]] * T if absorb else 0
            for r in range(R):
                coef[r] = 0.0
                dn[r] = 0
                if not use[r]:
                    continue
                er = Yr[r]
                for k in range(K):
                    er -= Xr[r, k] * b1[k]
                    xh[r, k] = Xr[r, k]
                z = 0.0
                t = r if r < T else r - T
                okc = (not masked) or _cluster_ok(t, T, system, use, rowmap)
                if absorb:
                    dn[r] = _dvec(r, T, orth, complete, di[r], dv[r])
                    for c in range(j):
                        zm[r, c] = 0.0
                    for e2 in range(dn[r]):
                        cell = cell0 + di[r, e2]
                        f = dv[r, e2]
                        er -= Wr[r] * f * delta[cell]
                        for k in range(K):
                            xh[r, k] -= Wr[r] * f * xshift[cell, k]
                        z -= f * Mzv[cell]
                        for c in range(j):
                            zm[r, c] += f * Mz[cell, c]
                    if okc:
                        for c in range(j):
                            g[c] -= zm[r, c] * er
                e[r] = er
                for a in range(nz[r]):
                    c = cmap[idx[r, a]]
                    if c >= 0:
                        if okc:
                            g[c] += val[r, a] * er
                        z += val[r, a] * v[c]
                vz[r] = z
                if okc:
                    for k in range(K):
                        u[k] += z * xh[r, k]
                if nf > 0:
                    for fu in range(nf):
                        av = 0.0
                        for aa in range(nz[r]):
                            c = cmap[idx[r, aa]]
                            if c >= 0:
                                av += val[r, aa] * Lf[fu, c]
                        for e2 in range(dn[r]):
                            av += dv[r, e2] * Gd[fu, cell0 + di[r, e2]]
                        af[r, fu] = av
                    if okc:
                        for e2 in range(dn[r]):
                            de[di[r, e2]] += dv[r, e2] * er
                            ud[di[r, e2]] += z * Wr[r] * dv[r, e2]
            if Lnl > 0:
                for c in range(Lnl):
                    gnl[c] = 0.0
                    for k in range(K):
                        Jnl[c, k] = 0.0
                _nl_panel(TV, touse0, ty, tx, b1, T, nlcoll, nlmap, gnl, Jnl, nlkind, xs_nl)
                for c in range(Lnl):
                    for k in range(K):
                        u[k] += Jnl[c, k] * v[j + c]
            if panel_f != 0.0:
                a_i = 0.0
                for c in range(j):
                    a_i += g[c] * v[c]
                for c in range(Lnl):
                    a_i += gnl[c] * v[j + c]
                for c in range(Lnl):
                    for k in range(K):
                        Dnl[ch, c, k] += panel_f * (a_i * Jnl[c, k] + gnl[c] * u[k])
                for r in range(R):
                    t = r if r < T else r - T
                    if use[r] and ((not masked) or _cluster_ok(t, T, system, use, rowmap)):
                        coef[r] += panel_f * a_i
                for c in range(j):
                    if g[c] != 0.0:
                        for k in range(K):
                            D[ch, c, k] += panel_f * g[c] * u[k]
                for fu in range(nf):
                    cp = 0.0
                    for c in range(j):
                        cp += Lf[fu, c] * g[c]
                    for s in range(T):
                        cp += Rf[fu, cell0 + s] * de[s]
                    cp *= panel_f
                    for k in range(K):
                        Ob[ch, fu, k] += cp * u[k]
                    for s in range(T):
                        Od[ch, fu, cell0 + s] += cp * ud[s]
            for t in range(T):
                if not _cluster_ok(t, T, system, use, rowmap):
                    continue
                if row_f != 0.0:
                    for c in range(j):
                        gt[c] = 0.0
                    for k in range(K):
                        ut[k] = 0.0
                    a_t = 0.0
                    for e2 in range(2 if system else 1):
                        r = t + e2 * T
                        if use[r]:
                            for a in range(nz[r]):
                                c = cmap[idx[r, a]]
                                if c >= 0:
                                    gt[c] += val[r, a] * e[r]
                            if absorb:
                                for c in range(j):
                                    gt[c] -= zm[r, c] * e[r]
                            for k in range(K):
                                ut[k] += vz[r] * xh[r, k]
                    for c in range(j):
                        a_t += gt[c] * v[c]
                    for e2 in range(2 if system else 1):
                        r = t + e2 * T
                        if use[r]:
                            coef[r] += row_f * a_t
                    for c in range(j):
                        if gt[c] != 0.0:
                            for k in range(K):
                                D[ch, c, k] += row_f * gt[c] * ut[k]
                    for fu in range(nf):
                        ct = 0.0
                        for c in range(j):
                            ct += Lf[fu, c] * gt[c]
                        for e2 in range(2 if system else 1):
                            r = t + e2 * T
                            if use[r]:
                                for ee in range(dn[r]):
                                    ct += Rf[fu, cell0 + di[r, ee]] * dv[r, ee] * e[r]
                        ct *= row_f
                        for e2 in range(2 if system else 1):
                            r = t + e2 * T
                            if use[r]:
                                for k in range(K):
                                    Ob[ch, fu, k] += ct * vz[r] * xh[r, k]
                                for ee in range(dn[r]):
                                    Od[ch, fu, cell0 + di[r, ee]] += ct * vz[r] * Wr[r] * dv[r, ee]
                r0 = rowmap[t]
                for qq in range(CLacc.shape[1]):
                    cc = CLacc[r0, qq]
                    if cc < 0:
                        continue
                    for e2 in range(2 if system else 1):
                        r = t + e2 * T
                        if use[r]:
                            coef[r] += alpha[cc]
                            for k in range(K):
                                U[ch, cc, k] += vz[r] * xh[r, k]
                            for fu in range(nf):
                                cf = cfc[cc, fu] * vz[r]
                                for k in range(K):
                                    Ob[ch, fu, k] += cf * xh[r, k]
                                for ee in range(dn[r]):
                                    Od[ch, fu, cell0 + di[r, ee]] += cf * Wr[r] * dv[r, ee]
            for r in range(R):
                if use[r] and coef[r] != 0.0:
                    for a in range(nz[r]):
                        c = cmap[idx[r, a]]
                        if c >= 0:
                            for k in range(K):
                                D[ch, c, k] += coef[r] * val[r, a] * xh[r, k]
                    if absorb:
                        for e2 in range(dn[r]):
                            f = coef[r] * dv[r, e2]
                            for k in range(K):
                                Dcell[ch, cell0 + di[r, e2], k] += f * xh[r, k]
                    for fu in range(nf):
                        cf = coef[r] * af[r, fu]
                        for k in range(K):
                            Ob[ch, fu, k] += cf * xh[r, k]
                        for ee in range(dn[r]):
                            Od[ch, fu, cell0 + di[r, ee]] += cf * Wr[r] * dv[r, ee]
