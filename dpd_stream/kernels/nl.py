"""Ahn and Schmidt's nonlinear moments: one panel's (`_nl_panel`) and the
sums over panels (`nl_pass1`)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .panel import _build, _scratch


@njit(cache=True)
def _nl_last(T, touse0):
    """The panel's last period with an observation (levels), or -1."""
    for t in range(T - 1, -1, -1):
        if touse0[t]:
            return t
    return -1


@njit(cache=True)
def _nl_panel(TV, touse0, ty, tx, b, T, nlcoll, nlmap, gnl, Jnl, nlkind, xs):
    """Ahn and Schmidt's (1995) moments of one panel at b, u = y - x'b in
    levels (with the panel effect), into gnl (per moment, by `nlmap`), and
    their derivative's negative into Jnl; `xs` is scratch (K).

    nlkind 1 (no serial correlation): u_iT (u_it - u_i,t-1) for each period
    t < T (the panel's last) whose next period is also observed (as
    xtdpdgmm's), moment t (or 0, collapsed). nlkind 2 (also
    homoskedasticity): ubar_i (u_it - u_i,t-1) for every period t with its
    predecessor (ubar the panel's mean), moment t (or 0); and the linear
    y_i,t-2 du_i,t-1 - y_i,t-1 du_it for each t with t - 1 and t - 2,
    moment T + t (or T)."""
    K = tx.shape[0]
    if nlkind == 1:
        Ti = _nl_last(T, touse0)
        if Ti < 2:
            return
        uT = TV[Ti, ty]
        for k in range(K):
            uT -= TV[Ti, tx[k]] * b[k]
        for t in range(1, Ti):
            if not (touse0[t] and touse0[t - 1] and touse0[t + 1]):
                continue
            c = nlmap[0 if nlcoll else t]
            if c < 0:
                continue
            du = TV[t, ty] - TV[t - 1, ty]
            for k in range(K):
                du -= (TV[t, tx[k]] - TV[t - 1, tx[k]]) * b[k]
            gnl[c] += uT * du
            for k in range(K):
                Jnl[c, k] += TV[Ti, tx[k]] * du + (TV[t, tx[k]] - TV[t - 1, tx[k]]) * uT
        return
    n = 0
    ybar = 0.0
    for k in range(K):
        xs[k] = 0.0
    for t in range(T):
        if touse0[t]:
            n += 1
            ybar += TV[t, ty]
            for k in range(K):
                xs[k] += TV[t, tx[k]]
    if n < 2:
        return
    ybar /= n
    for k in range(K):
        xs[k] /= n
    ubar = ybar
    for k in range(K):
        ubar -= xs[k] * b[k]
    for t in range(1, T):
        if not (touse0[t] and touse0[t - 1]):
            continue
        du = TV[t, ty] - TV[t - 1, ty]
        for k in range(K):
            du -= (TV[t, tx[k]] - TV[t - 1, tx[k]]) * b[k]
        c = nlmap[0 if nlcoll else t]
        if c >= 0:
            gnl[c] += ubar * du
            for k in range(K):
                Jnl[c, k] += xs[k] * du + (TV[t, tx[k]] - TV[t - 1, tx[k]]) * ubar
        if t < 2 or not touse0[t - 2]:
            continue
        c = nlmap[T + (0 if nlcoll else t)]
        if c < 0:
            continue
        dp = TV[t - 1, ty] - TV[t - 2, ty]
        for k in range(K):
            dp -= (TV[t - 1, tx[k]] - TV[t - 2, tx[k]]) * b[k]
        y2, y1 = TV[t - 2, ty], TV[t - 1, ty]
        gnl[c] += y2 * dp - y1 * du
        for k in range(K):
            Jnl[c, k] += (y2 * (TV[t - 1, tx[k]] - TV[t - 2, tx[k]])
                          - y1 * (TV[t, tx[k]] - TV[t - 1, tx[k]]))


@njit(parallel=True, cache=True)
def nl_pass1(cbounds, pstart, tt, B, W, use_w, T, system, orth,
             term_base, term_k, term_diff, ty, tx, cmp_terms,
             iv_term, iv_eq, iv_pass, iv_mz,
             g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
             lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
             maxnz, nlcoll, nlkind, s0, s1, s2, ncnt, npan, h_ptr, h_idx, h_val, zhc, chc):
    """Pass 1's sums of the Ahn-Schmidt moments (see `_nl_panel`), which
    are quadratic in b: for a moment a d, with a = (y_T or ybar) - (x_T or
    xbar)'b and d = dy_t - dx_t'b, s0 = sum a d, s1 = sum (a_y dx + dy a_x),
    s2 = sum a_x dx', so that the moment's sum is s0 - s1'b + b's2 b; the
    linear ones (homoskedasticity) have s2 = 0. ncnt counts the terms, and
    npan[ch] the panels with an observation in levels (the nonlinear moments'
    sample: xtdpdgmm's number of groups).

    With homoskedasticity (`nlkind` 2, not with orthogonal deviations) the
    linear moments it adds, y_{t-2} du_{t-1} - y_{t-1} du_t, are instruments
    of the differenced equation, with y_{t-2} in row t - 1 and -y_{t-1} in row
    t (one column per t, or one in all if `nlcoll`). xtdpdgmm weights them with
    the linear block (Z'HZ) in the first step; for that, zhc[ch, j, c] gets
    the sums Z'HC over the panels' rows (H's non-zero entries `h_ptr`...), and
    chc[ch, c, c'] the sums C'HC."""
    nch = cbounds.shape[0] - 1
    nbase = B.shape[1]
    nt = term_base.shape[0]
    K = tx.shape[0]
    jiv = iv_term.shape[0]
    R = 2 * T if system else T
    ngt = g_terms.shape[0]
    for ch in prange(nch):
        (BG, TV, present, complete, touse0, wgrid, rowmap, sa, sb, XT, YT, IVT, GS, GD,
         Xr, Yr, Wr, use, idx, val, nz) = _scratch(T, R, nbase, nt, K, jiv, ngt, maxnz)
        xa = np.zeros(max(K, 1))
        Cm = np.zeros((T, T))
        want_c = nlkind == 2 and not orth and zhc.shape[1] > 1
        for p in range(cbounds[ch], cbounds[ch + 1]):
            n_in = _build(pstart[p], pstart[p + 1], tt, B, W, use_w, T, system, orth,
                          term_base, term_k, term_diff, ty, tx, cmp_terms,
                          iv_term, iv_eq, iv_pass, iv_mz,
                          g_nb, g_t0, g_terms, g_eqrow, g_fd, g_coll, g_orth, g_a, g_b, g_lp,
                          lag_lo, lag_hi, lag_col, g_extra, g_xlag, g_xlo, g_xhi, g_xcol, g_xsh,
                          BG, TV, present, complete, touse0, wgrid, rowmap, sa, sb, XT, YT,
                          IVT, GS, GD, Xr, Yr, Wr, use, idx, val, nz)
            for t in range(T):
                if touse0[t]:
                    npan[ch] += 1.0
                    break
            has_c = False
            if want_c:
                for r in range(T):
                    for c in range(T):
                        Cm[r, c] = 0.0
            if n_in == 0:
                continue
            if nlkind == 1:
                Ti = _nl_last(T, touse0)
                if Ti < 2:
                    continue
                ay = TV[Ti, ty]
                for k in range(K):
                    xa[k] = TV[Ti, tx[k]]
                t_hi = Ti
            else:
                n = 0
                ay = 0.0
                for k in range(K):
                    xa[k] = 0.0
                for t in range(T):
                    if touse0[t]:
                        n += 1
                        ay += TV[t, ty]
                        for k in range(K):
                            xa[k] += TV[t, tx[k]]
                if n < 2:
                    continue
                ay /= n
                for k in range(K):
                    xa[k] /= n
                t_hi = T
            for t in range(1, t_hi):
                if not (touse0[t] and touse0[t - 1]):
                    continue
                if nlkind == 1 and not touse0[t + 1]:      # as xtdpdgmm: the next period too
                    continue
                c = 0 if nlcoll else t
                d = TV[t, ty] - TV[t - 1, ty]
                s0[ch, c] += ay * d
                ncnt[ch, c] += 1
                for k in range(K):
                    f = TV[t, tx[k]] - TV[t - 1, tx[k]]
                    s1[ch, c, k] += ay * f + d * xa[k]
                    for k2 in range(K):
                        s2[ch, c, k, k2] += xa[k] * (TV[t, tx[k2]] - TV[t - 1, tx[k2]])
                if nlkind == 1 or t < 2 or not touse0[t - 2]:
                    continue
                c = T + (0 if nlcoll else t)
                y2, y1 = TV[t - 2, ty], TV[t - 1, ty]
                s0[ch, c] += y2 * (TV[t - 1, ty] - TV[t - 2, ty]) - y1 * d
                ncnt[ch, c] += 1
                if want_c:
                    cc = 0 if nlcoll else t
                    Cm[t - 1, cc] += y2
                    Cm[t, cc] -= y1
                    has_c = True
                for k in range(K):
                    s1[ch, c, k] += (y2 * (TV[t - 1, tx[k]] - TV[t - 2, tx[k]])
                                     - y1 * (TV[t, tx[k]] - TV[t - 1, tx[k]]))
            if has_c:
                for r in range(R):
                    if not use[r]:
                        continue
                    for hp in range(h_ptr[r], h_ptr[r + 1]):
                        u = h_idx[hp]
                        if u >= T or not use[u]:
                            continue
                        coef = h_val[hp] * np.sqrt(Wr[r] * Wr[u])
                        for c in range(T):
                            cu = Cm[u, c]
                            if cu == 0.0:
                                continue
                            for a in range(nz[r]):
                                zhc[ch, idx[r, a], c] += coef * val[r, a] * cu
                            if r < T:
                                for c2 in range(T):
                                    if Cm[r, c2] != 0.0:
                                        chc[ch, c2, c] += coef * Cm[r, c2] * cu
