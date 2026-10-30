"""The passes for absorbed period-by-group effects (`g^year`): the cells'
sums, their cross-moments with the instruments, and the effects' moments."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .panel import _build, _cluster_ok, _dvec, _scratch


@njit(parallel=True, cache=True)
def pass1_cells(cbounds, pstart, tt, B, W, use_w, T, system, orth,
                term_base, term_k, term_diff, ty, tx, cmp_terms,
                iv_term, iv_eq, iv_pass, iv_mz,
                g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                maxnz, h_ptr, h_idx, h_val, AG, DDx, DHD, CZw, CHZ, CX):
    """Pass 1's sums by cell (g, s) for an absorbed effect (difference
    GMM), with D the absorbed dummies of the transformed rows (`_dvec`), W
    the weights and H the one-step weight's error covariance:
    DDx[ch, g] = D_g'W D_g and DHD[ch, g] = D_g'W^.5 H W^.5 D_g (T x T per
    group), CZw = D'W Z, CHZ = D'W^.5 H W^.5 Z and CX = D'[X y] (by cell; X
    and y are already weighted)."""
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
        dn = np.zeros(T, np.int64)
        di = np.zeros((T, T), np.int64)
        dv = np.zeros((T, T))
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
            gi = AG[pstart[p]]
            cell0 = gi * T
            for r in range(T):
                dn[r] = _dvec(r, T, orth, complete, di[r], dv[r]) if use[r] else 0
            for r in range(T):
                if dn[r] == 0:
                    continue
                for e in range(dn[r]):
                    sa = di[r, e]
                    c = cell0 + sa
                    wd = Wr[r] * dv[r, e]
                    for e2 in range(dn[r]):
                        DDx[ch, gi, sa, di[r, e2]] += wd * dv[r, e2]
                    for a in range(nz[r]):
                        CZw[ch, c, idx[r, a]] += wd * val[r, a]
                    for k in range(K):
                        CX[ch, c, k] += dv[r, e] * Xr[r, k]
                    CX[ch, c, K] += dv[r, e] * Yr[r]
                for hp in range(h_ptr[r], h_ptr[r + 1]):
                    u = h_idx[hp]
                    if not use[u]:
                        continue
                    coef = h_val[hp] * np.sqrt(Wr[r] * Wr[u])
                    for e in range(dn[r]):
                        sa = di[r, e]
                        c = cell0 + sa
                        f = dv[r, e] * coef
                        for a in range(nz[u]):
                            CHZ[ch, c, idx[u, a]] += f * val[u, a]
                        for e2 in range(dn[u]):
                            DHD[ch, gi, sa, di[u, e2]] += f * dv[u, e2]


@njit(parallel=True, cache=True)
def cross_pass(cbounds, pstart, tt, B, W, use_w, T, system, orth,
               term_base, term_k, term_diff, ty, tx, cmp_terms,
               iv_term, iv_eq, iv_pass, iv_mz,
               g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
               lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
               maxnz, b, AG, delta, CLc, Gtot, Cross):
    """For an absorbed effect clustered by variables other than the panel:
    the cross moments of the cells' moments D'e and the instruments' Z~'e
    over the clusters, sum_c (D_c'e_c)(Z~_c'e_c)', given each cluster's
    Z~_c'e_c already scaled by its clustering's factor (Gtot, by the codes
    in CLc). Per chunk: Cross[ch, cell]."""
    nch = cbounds.shape[0] - 1
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    ngt = g_terms.shape[0]
    j = Gtot.shape[1]
    for ch in prange(nch):
        (BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT, IVT, GS, GD,
         Xr, Yr, Wr, use, idx, val, nz) = _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz)
        didx = np.zeros(T, np.int64)
        dval = np.zeros(T)
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
            cell0 = AG[pstart[p]] * T
            for r in range(T):
                if not (use[r] and _cluster_ok(r, T, system, use, rowmap)):
                    continue
                nd = _dvec(r, T, orth, complete, didx, dval)
                er = Yr[r]
                for k in range(K):
                    er -= Xr[r, k] * b[k]
                for e2 in range(nd):
                    er -= Wr[r] * dval[e2] * delta[cell0 + didx[e2]]
                r0 = rowmap[r]
                for qq in range(CLc.shape[1]):
                    cc = CLc[r0, qq]
                    if cc < 0:
                        continue
                    for e2 in range(nd):
                        f = dval[e2] * er
                        for c in range(j):
                            Cross[ch, cell0 + didx[e2], c] += f * Gtot[cc, c]


@njit(parallel=True, cache=True)
def cellscore_pass(cbounds, pstart, tt, B, W, use_w, T, system, orth,
                   term_base, term_k, term_diff, ty, tx, cmp_terms,
                   iv_term, iv_eq, iv_pass, iv_mz,
                   g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                   lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                   maxnz, b, AG, delta, Hf, masked, rowmode, CLacc, Hp, AccH, SRH):
    """An absorbed effect's moments D'e at `b` and the effects `delta`, in
    the directions Hf (cells x nf): h_r = Hf' d_r e_r for each row with a
    cluster (all rows unless `masked`), summed per panel into Hp[p -
    pstart0], per cluster of the general clusterings (codes in CLacc) into
    AccH[ch, code], and with clustering by observation (`rowmode`) as
    SRH[ch] = sum_r h_r h_r'."""
    nch = cbounds.shape[0] - 1
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    ngt = g_terms.shape[0]
    nf = Hf.shape[1]
    p0 = cbounds[0]
    for ch in prange(nch):
        (BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT, IVT, GS, GD,
         Xr, Yr, Wr, use, idx, val, nz) = _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz)
        didx = np.zeros(T, np.int64)
        dval = np.zeros(T)
        h = np.zeros(nf)
        for p in range(cbounds[ch], cbounds[ch + 1]):
            n_in = _build(pstart[p], pstart[p + 1], tt, B, W, use_w, T, system, orth,
                          term_base, term_k, term_diff, ty, tx, cmp_terms,
                          iv_term, iv_eq, iv_pass, iv_mz,
                          g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                          lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                          BG, TV, present, complete, touse0, wgrid, rowmap, s1, s2, XT, YT,
                          IVT, GS, GD, Xr, Yr, Wr, use, idx, val, nz)
            q = p - p0
            for f in range(nf):
                Hp[q, f] = 0.0
            if n_in == 0:
                continue
            cell0 = AG[pstart[p]] * T
            for r in range(T):
                if not use[r]:
                    continue
                if masked and not _cluster_ok(r, T, system, use, rowmap):
                    continue
                nd = _dvec(r, T, orth, complete, didx, dval)
                er = Yr[r]
                for k in range(K):
                    er -= Xr[r, k] * b[k]
                for e2 in range(nd):
                    er -= Wr[r] * dval[e2] * delta[cell0 + didx[e2]]
                for f in range(nf):
                    h[f] = 0.0
                    for e2 in range(nd):
                        h[f] += dval[e2] * er * Hf[cell0 + didx[e2], f]
                    Hp[q, f] += h[f]
                r0 = rowmap[r]
                for qq in range(CLacc.shape[1]):
                    cc = CLacc[r0, qq]
                    if cc >= 0:
                        for f in range(nf):
                            AccH[ch, cc, f] += h[f]
                if rowmode:
                    for f in range(nf):
                        for f2 in range(nf):
                            SRH[ch, f, f2] += h[f] * h[f2]
