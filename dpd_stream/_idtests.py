"""The identification tests of a fit (Kleibergen-Paap, Sanderson-Windmeijer,
Anderson-Rubin, effective F) from the sums and one pass for the scores.
"""

from __future__ import annotations

import numpy as np

from . import inference as inf
from . import kernels as kn
from ._blocks import _blocks, _chunks


class _IdTests:
    """Methods of `estimator.DPD`."""

    def _idtests(self, sums, ZX, ZY, keep, live, b, V):
        """Underidentification and weak-identification tests (Windmeijer
        2025; as Stata's underid with ``kp`` and ``sw``, by ranktest's
        algebra; see `hdfe_stream.core.ivtests`): the regressors that are
        their own IV-style instruments are the included exogenous ones
        (partialled out, with the constant and any absorbed effect); the
        others are endogenous, instrumented by the remaining columns (with
        PCA, the IV-style ones and the components). Robust to the fit's
        clustering (by panel unless given), in the weighted regressions with
        weights. `ZX` is the moments' Z'X (live columns, scaled, net of an
        absorbed effect), and ZY its Z'y. Also the Anderson-Rubin test (y is
        one more column of the first stage: the reduced form), and with one
        endogenous regressor its confidence set, around the estimate `b`
        (variance `V`)."""
        from hdfe_stream.core.ivtests import anderson_rubin, identification
        lay = self.lay
        K = lay.K
        c = self.wscale
        # each regressor's own IV-style column, if any
        own = {}
        for col, (term, eq, passthru, _, _) in enumerate(lay.iv_cols):
            ok = (eq == 2 and not passthru) if self.system else (eq in (1, 2) and not passthru)
            if lay.terms[term].base < 0:
                ok = self.system and eq == 0          # the constant
            if ok:
                own.setdefault(term, col)
        cmap = self.cmap
        x2, x2cols, ycols = [], set(), []
        for kk in keep:
            col = own.get(lay.x[kk])
            if col is not None and cmap[col] >= 0:
                x2.append(kk)
                x2cols.add(int(cmap[col]))
            else:
                ycols.append(kk)
        # the own columns of regressors dropped as collinear add nothing
        for kk in range(K):
            col = own.get(lay.x[kk])
            if kk not in keep and col is not None and cmap[col] >= 0:
                x2cols.add(int(cmap[col]))
        Ke = len(ycols)
        if Ke == 0:
            return None
        nl_ = len(live)
        P = self.P
        jiv_live = int((live < lay.j_iv).sum())
        direct = [col for col in range(jiv_live if P is not None else nl_)
                  if col not in x2cols]
        ncomp = P.shape[1] - jiv_live if P is not None else 0
        L1 = len(direct) + ncomp
        if self.idtests == "auto" and L1 * Ke > 500:
            self.log(f"identification tests skipped: {L1} excluded instruments x {Ke} "
                     "endogenous regressors (idtests=True forces them; collapse=True makes "
                     "them cheap)")
            return None
        self.log("pass 4: identification tests")
        # the excluded instruments as combinations of the live columns
        Sel = np.zeros((nl_, L1))
        Sel[direct, np.arange(len(direct))] = 1.0
        if ncomp:
            Sel[jiv_live:, len(direct):] = P[jiv_live:, jiv_live:]
        ZZ = sums.ZZp[np.ix_(live, live)] * c
        XX = (sums.Q[0] + sums.Q[1]) * c                 # with y, column K
        ZX = np.column_stack([ZX, ZY])
        ya = ycols + [K]
        ab = self.absorb_info
        Mz1 = Mx = np.zeros((1, 1))
        dof_extra = 0
        if ab:
            CZw = ab["CZw"]
            ZZ = ZZ - CZw.T @ ab["Mz"]
            XX = XX - ab["CX"].T @ ab["Mx"]
            Mz1 = np.ascontiguousarray(ab["Mz"] @ Sel)
            Mx = np.ascontiguousarray(ab["Mx"])
            dof_extra = ab["n_cells"]
        Z1Z1 = Sel.T @ ZZ @ Sel
        Z1X = Sel.T @ ZX
        iX2, _ = inf.invsym(XX[np.ix_(x2, x2)]) if x2 else (np.zeros((0, 0)), None)
        Gz = iX2 @ Z1X[:, x2].T if x2 else np.zeros((0, L1))
        Gy = iX2 @ XX[np.ix_(x2, ya)] if x2 else np.zeros((0, Ke + 1))
        ZtZt = Z1Z1 - Z1X[:, x2] @ Gz
        ZtY = Z1X[:, ya] - Z1X[:, x2] @ Gy
        YtY = XX[np.ix_(ya, ya)] - XX[np.ix_(x2, ya)].T @ Gy
        iZZ, _ = inf.invsym(ZtZt)
        Pi = iZZ @ ZtY
        zmap = np.full(nl_, -1, np.int64)
        zmap[direct] = np.arange(len(direct))
        pmap = np.full(nl_ if ncomp else 0, -1, np.int64)
        if ncomp:
            pmap[jiv_live:] = np.arange(nl_ - jiv_live)
        Pg = np.ascontiguousarray(P[jiv_live:, jiv_live:]) if ncomp else np.zeros((1, 1))
        m = L1 * (Ke + 1)
        clus = self.clusters
        rowmode = any(cc.by_obs for cc in clus)
        general = [cc for cc in clus if cc.col >= 0 and not cc.nested]
        nested = [cc for cc in clus if cc.col >= 0 and cc.nested]
        n_acc = sum(cc.n_codes for cc in general)
        off = 0
        for cc in general:
            cc.offset = off
            off += cc.n_codes
        nch = self.n_threads
        nch_acc = self._copies(n_acc * m * 16, nch)
        AccU = np.zeros((nch_acc, max(n_acc, 1), m) if general else (1, 1, m))
        AccV = np.zeros_like(AccU)
        SRU = np.zeros((nch, m, m) if rowmode else (1, 1, 1))
        SRV = np.zeros_like(SRU)
        Nacc = {id(cc): (np.zeros((cc.n_codes, m)), np.zeros((cc.n_codes, m))) for cc in nested}
        OU = np.zeros((m, m))
        OV = np.zeros((m, m))
        panel_f = 1.0 if not clus else sum(cc.sign for cc in clus if cc.is_panel)
        for blk, pstart in _blocks(self.store):
            npan = len(pstart) - 1
            CL = blk.get("cl", np.zeros((len(blk["t"]), 0), np.int64))
            CLacc = np.full((len(blk["t"]), len(general)), -1, np.int64)
            for q, cc in enumerate(general):
                vv = CL[:, cc.col]
                CLacc[:, q] = np.where(vv >= 0, vv + cc.offset, -1)
            U = np.zeros((npan, m))
            V = np.zeros((npan, m))
            AG = blk["ag"] if ab else np.zeros(0, np.int64)
            kn.idtest_pass(*self._kargs(blk, _chunks(pstart, nch if not general else nch_acc),
                                        pstart),
                           cmap, zmap, L1, pmap, Pg, len(direct),
                           np.array(ya, np.int64), np.array(x2, np.int64),
                           np.ascontiguousarray(Gz), np.ascontiguousarray(Gy),
                           np.ascontiguousarray(Pi), AG, Mz1, Mx, bool(clus), CLacc, rowmode,
                           U, V, AccU, AccV, SRU, SRV)
            if panel_f:
                OU += panel_f * (U.T @ U)
                OV += panel_f * (V.T @ V)
            for cc in nested:
                codes = CL[pstart[:-1], cc.col]
                ok = codes >= 0
                np.add.at(Nacc[id(cc)][0], codes[ok], U[ok])
                np.add.at(Nacc[id(cc)][1], codes[ok], V[ok])
        for cc in nested:
            A, B = Nacc[id(cc)]
            OU += cc.sign * (A.T @ A)
            OV += cc.sign * (B.T @ B)
        if general:
            AU, AV = AccU.sum(0), AccV.sum(0)
            for cc in general:
                A = AU[cc.offset:cc.offset + cc.n_codes]
                B = AV[cc.offset:cc.offset + cc.n_codes]
                OU += cc.sign * (A.T @ A)
                OV += cc.sign * (B.T @ B)
        if rowmode:
            f = sum(cc.sign for cc in clus if cc.by_obs)
            OU += f * SRU.sum(0)
            OV += f * SRV.sum(0)
        n_rows = sums.n_diff + (sums.n_obs if self.system else 0)
        G = min((cc.n for cc in clus), default=sums.n_groups)
        OU, OV = (OU + OU.T) / 2, (OV + OV.T) / 2
        m1 = L1 * Ke
        f = {"n_obs": n_rows, "n_inst": self.jc_final, "n_clusters": G, "dof_extra": dof_extra}
        out = identification(ZtZt, ZtY[:, :Ke], YtY[:Ke, :Ke], OU[:m1, :m1], OV[:m1, :m1],
                             endog=[lay.xnames[i] for i in ycols],
                             exog=[lay.xnames[i] for i in x2], sw="gmm", **f)
        ctr = scl = None
        if Ke == 1:
            i = list(keep).index(ycols[0])
            ctr, scl = float(b[i]), float(np.sqrt(max(V[i, i], 0.0)))
        out["anderson_rubin"] = anderson_rubin(ZtZt, ZtY, OV, center=ctr, scale=scl, **f)
        return out
