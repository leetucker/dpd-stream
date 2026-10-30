"""The passes over the blocks as methods of the fit: the residual pass (moment
covariance and Arellano-Bond sums at an estimate), the cells' cross-moments
and Windmeijer's correction.
"""

from __future__ import annotations

import numpy as np

from . import kernels as kn
from . import layout as lay_mod
from ._blocks import _blocks, _chunks


class _Passes:
    """Methods of `estimator.DPD`."""

    def _resid(self, b_full, ar_mode, need_S=True, delta=None, xshift=None, cross=False):
        """Pass 2: the moment covariance at b and, if asked, the AR tests
        (for one-step non-robust ones, with sigma^2 = 1: their terms are
        linear in it). With an absorbed effect, the cells' `delta` and
        `xshift` (see `kernels.moments.resid_pass`), and with `cross` the cells'
        cross moments."""
        lay, st = self.lay, self.store
        j, K = self.jc, lay.K
        nch = self.n_threads
        T = lay.T
        R = 2 * T if self.system else T
        clus = self.clusters
        rowmode = any(c.by_obs for c in clus)
        general = [c for c in clus if c.col >= 0 and not c.nested]
        nested = [c for c in clus if c.col >= 0 and c.nested]
        masked = bool(clus)
        n_acc = sum(c.n_codes for c in general)
        off = 0
        for c in general:
            c.offset = off
            off += c.n_codes
        nch_acc = nch if n_acc * j * 8 * nch < 2e9 else 1
        if n_acc * j * 8 > 8e9:
            raise MemoryError(f"clustering by {[c.vars for c in general]}: "
                              f"{n_acc:,} clusters x {j:,} instruments do not fit in memory")
        Acc = np.zeros((nch_acc, max(n_acc, 1), j)) if general else np.zeros((1, 1, j))
        Nacc = {id(c): np.zeros((c.n_codes, j)) for c in nested}
        SR = np.zeros((nch, j, j)) if rowmode else np.zeros((1, 1, 1))
        S = np.zeros((j + self.Lnl, j + self.Lnl))
        ARwHw = np.zeros((nch, max(self.artests, 1)))
        ARtmp = np.zeros((nch, max(self.artests, 1), K))
        ARzhw = np.zeros((nch, max(self.artests, 1), j + self.Lnl))
        ar_sum = np.zeros(max(self.artests, 1))
        Hd = np.zeros((T, T))
        Psi = np.zeros((T, R))
        if ar_mode == 2:
            Hd = lay_mod.h_matrix(self.h, False, False, False, T)
            Psi[:, :T] = lay_mod.h_matrix(self.h, False, self.orth, False, T)
        ab = self.absorb_info
        SSE = np.zeros(nch)
        Cross = (np.zeros((nch, ab["ncell"], j)) if (ab and cross) else np.zeros((1, 1, 1)))
        nar = max(self.artests, 1)
        ARtd = np.zeros((nch, nar, ab["ncell"]) if ab else (1, 1, 1))
        ARzhd = np.zeros_like(ARtd)

        def fac(c):
            return c.sign * (c.n / (c.n - 1) if self.small else 1.0)
        xpf = 1.0 if not clus else sum(fac(c) for c in clus if c.is_panel)
        xrf = sum(fac(c) for c in clus if c.by_obs)
        keepx = np.zeros(K, np.bool_)
        keepx[self.keep] = True
        for blk, pstart in _blocks(st):
            npan = len(pstart) - 1
            cb = _chunks(pstart, nch if not general else nch_acc)
            CL = blk.get("cl", np.zeros((len(blk["t"]), 0), np.int64))
            CLacc = np.full((len(blk["t"]), len(general)), -1, np.int64)
            for q, c in enumerate(general):
                v = CL[:, c.col]
                CLacc[:, q] = np.where(v >= 0, v + c.offset, -1)
            G = np.zeros((npan, j + self.Lnl))
            ARs = np.zeros((npan, max(self.artests, 1)))
            kn.resid_pass(*self._kargs(blk, cb, pstart), self.cmap, j, b_full, masked,
                          rowmode, CLacc, Acc, ar_mode, self.artests if ar_mode else 0, keepx,
                          Hd, Psi, self.wscale, G, SR, ARs, ARwHw, ARtmp, ARzhw,
                          *self._absorb_args(blk, delta, xshift), SSE, Cross, xpf, xrf,
                          self.collapse, self.nlmap, self.Lnl, self.nlkind, ARtd, ARzhd)
            if need_S:
                if not clus:
                    S += G.T @ G
                for c in clus:
                    if c.is_panel:
                        f = c.sign * (c.n / (c.n - 1) if self.small else 1.0)
                        S += f * (G.T @ G)
                for c in nested:
                    codes = CL[pstart[:-1], c.col]
                    ok = codes >= 0
                    np.add.at(Nacc[id(c)], codes[ok], G[ok])
            ar_sum += ARs.sum(0)
        info = {}
        if need_S:
            for c in nested:
                f = c.sign * (c.n / (c.n - 1) if self.small else 1.0)
                A = Nacc[id(c)]
                S += f * (A.T @ A)
            if general:
                Acc_t = Acc.sum(0)
                for c in general:
                    f = c.sign * (c.n / (c.n - 1) if self.small else 1.0)
                    A = Acc_t[c.offset:c.offset + c.n_codes]
                    S += f * (A.T @ A)
                info["Acc"] = Acc_t
            if rowmode:
                SRt = SR.sum(0)
                for c in clus:
                    if c.by_obs:
                        f = c.sign * (c.n / (c.n - 1) if self.small else 1.0)
                        S += f * SRt
            info["Nacc"] = Nacc
        info["S"] = (S + S.T) / 2
        info["SSE"] = float(SSE.sum())
        if ab and cross:
            info["cross"] = Cross.sum(0)
            if any(c.col >= 0 for c in clus):
                info["cross"] += self._cross(b_full, delta, info, fac)
        if ar_mode:
            info["ar"] = (ar_sum, ARwHw.sum(0), ARtmp.sum(0), ARzhw.sum(0), ARtd.sum(0),
                          ARzhd.sum(0))
        return info

    def _cross(self, b_full, delta, info, fac):
        """The cross moments of an absorbed effect's moments and the
        instruments' over the clusters of the clusterings by variables (one
        more pass: each cluster's Z~_c'e_c is known only after pass 2)."""
        st, ab = self.store, self.absorb_info
        by_code = [c for c in self.clusters if c.col >= 0]
        parts, offs, off = [], {}, 0
        for c in by_code:
            A = (info["Nacc"][id(c)] if c.nested
                 else info["Acc"][c.offset:c.offset + c.n_codes])
            parts.append(fac(c) * A)
            offs[id(c)] = off
            off += c.n_codes
        Gtot = np.ascontiguousarray(np.vstack(parts))
        nch = self.n_threads
        Cross = np.zeros((nch, ab["ncell"], Gtot.shape[1]))
        self.log("pass 2b: absorbed effects' cross moments by cluster")
        for blk, pstart in _blocks(st):
            CL = blk["cl"]
            CLc = np.full((len(blk["t"]), len(by_code)), -1, np.int64)
            for q, c in enumerate(by_code):
                vv = CL[:, c.col]
                CLc[:, q] = np.where(vv >= 0, vv + offs[id(c)], -1)
            kn.cross_pass(*self._kargs(blk, _chunks(pstart, nch), pstart), b_full, blk["ag"],
                          delta, CLc, Gtot, Cross)
        return Cross.sum(0)

    def _windmeijer(self, b1_full, v_full, acc_info, delta=None, xshift=None, funcs=None):
        """Pass 3: the sums of Windmeijer's correction (see
        `kernels.moments.windmeijer_pass`). Clusterings by panel and by observation
        are done whole in the kernel; the others from their cluster sums of
        pass 2 (the coefficients e1_c'Z_c v per cluster) and the cluster
        sums of Z_c'X_c' v this pass accumulates.

        With an absorbed effect, `funcs` = (Lf, Rf, beta): the functionals
        whose contractions with the sums the kernel also forms, and beta
        each general cluster's Rf'D_c'e1_c (by clustering, or None if Rf is
        zero). Returns Draw, or with `funcs` (Draw, Ob, Od)."""
        lay, st = self.lay, self.store
        j, K = self.jc, lay.K
        nch = self.n_threads
        clus = self.clusters
        by_code = [c for c in clus if c.col >= 0]
        fac = {id(c): c.sign * (c.n / (c.n - 1) if self.small else 1.0) for c in clus}
        panel_f = 1.0 if not clus else sum(fac[id(c)] for c in clus if c.is_panel)
        row_f = sum(fac[id(c)] for c in clus if c.by_obs)
        sums = {}
        off = 0
        for c in by_code:
            if c.nested:
                A = acc_info["Nacc"][id(c)]
            else:
                A = acc_info["Acc"][c.offset:c.offset + c.n_codes]
            sums[id(c)] = (off, A)
            off += c.n_codes
        alpha = np.zeros(max(off, 1))
        for c in by_code:
            o, A = sums[id(c)]
            alpha[o:o + c.n_codes] = fac[id(c)] * (A @ v_full)
        D = np.zeros((nch, j, K))
        U = np.zeros((nch, max(off, 1), K))
        ab = self.absorb_info
        Dcell = np.zeros((nch, ab["ncell"], K)) if ab else np.zeros((1, 1, 1))
        Mzv = ab["Mz"] @ v_full[:j] if ab else np.zeros(1)
        Dnl = np.zeros((nch, max(self.Lnl, 1), K))
        if funcs is not None:
            Lf, Rf, beta = funcs
            nf = Lf.shape[0]
            Gd = np.ascontiguousarray(Rf - Lf @ ab["Mz"].T)
            cfc = np.zeros((max(off, 1), nf))
            for c in by_code:
                o, A = sums[id(c)]
                cfc[o:o + c.n_codes] = fac[id(c)] * (A @ Lf.T + (0 if beta is None
                                                                 else beta[id(c)]))
            Ob = np.zeros((nch, nf, K))
            Od = np.zeros((nch, nf, ab["ncell"]))
        else:
            Lf = Rf = Gd = cfc = np.zeros((0, 1))
            Ob = Od = np.zeros((1, 1, 1))
        for blk, pstart in _blocks(st):
            cb = _chunks(pstart, nch)
            CL = blk.get("cl", np.zeros((len(blk["t"]), 0), np.int64))
            CLc = np.full((len(blk["t"]), len(by_code)), -1, np.int64)
            for q, c in enumerate(by_code):
                vv = CL[:, c.col]
                CLc[:, q] = np.where(vv >= 0, vv + sums[id(c)][0], -1)
            AG, Mz, dl, xs = self._absorb_args(blk, delta, xshift)
            kn.windmeijer_pass(*self._kargs(blk, cb, pstart), self.cmap, j, b1_full, v_full,
                               panel_f, bool(clus), row_f, CLc, alpha, D, U,
                               AG, Mz, Mzv, dl, xs, Dcell, self.collapse, self.nlmap, self.Lnl,
                               Dnl, self.nlkind, Lf, Rf, Gd, cfc, Ob, Od)
        Draw = D.sum(0)
        if self.Lnl:
            Draw = np.vstack([Draw, Dnl.sum(0)])
        if ab:
            Draw -= ab["Mz"].T @ Dcell.sum(0)
        Ut = U.sum(0)
        for c in by_code:
            o, A = sums[id(c)]
            Draw += fac[id(c)] * (A.T @ Ut[o:o + c.n_codes])
        if funcs is not None:
            return Draw, Ob.sum(0), Od.sum(0)
        return Draw
