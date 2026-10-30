"""The fit with absorbed period-by-group effects (`g^year`): the cells'
moments, their inference (Windmeijer and the Arellano-Bond sums, exactly as
with the dummies) and the partialling.
"""

from __future__ import annotations

import numpy as np

from . import kernels as kn
from ._blocks import _blocks, _chunks


class _Absorb:
    """Methods of `estimator.DPD`."""

    def _check_absorb(self, terms):
        """The absorbed term must be one period-specific effect: the time
        variable, or g^time with g constant within panels (difference GMM,
        first differences)."""
        tvar = self.panel[1]
        if len(terms) != 1:
            raise NotImplementedError(
                "one absorbed fixed-effect term is available (e.g. '| year' or "
                f"'| industry^year'), got {list(terms)}")
        parts = [p.strip() for p in terms[0].split("^")]
        if tvar not in parts or "[" in terms[0]:
            raise ValueError(
                f"absorbed fixed effect {terms[0]!r}: difference GMM absorbs effects specific "
                f"to a period, '{tvar}' or 'g^{tvar}' with g constant within panels (any "
                "effect constant within a panel is removed by the differencing already)")
        if self.pca:
            raise NotImplementedError("absorbed fixed effects with pca are not available")
        return [p for p in parts if p != tvar]

    # ------------------------------------------------------------------ passes

    def _cellscores(self, b_full, delta, Hf):
        """A pass for the absorbed effect's moments g2 = D'e at `b_full` and
        `delta` (`kernels.cells.cellscore_pass`): their covariance in the
        directions Hf (cells x nf), Hf' S22 Hf, with the clusterings and
        factors of the moments' S; and each general cluster's Hf'D_c'e_c, by
        clustering (unscaled)."""
        st = self.store
        nch = self.n_threads
        clus = self.clusters
        nf = Hf.shape[1]
        Hf = np.ascontiguousarray(Hf)
        rowmode = any(c.by_obs for c in clus)
        general = [c for c in clus if c.col >= 0 and not c.nested]
        nested = [c for c in clus if c.col >= 0 and c.nested]
        off = 0
        for c in general:
            c.offset = off
            off += c.n_codes
        AccH = np.zeros((nch, max(off, 1), nf))
        Nacc = {id(c): np.zeros((c.n_codes, nf)) for c in nested}
        SRH = np.zeros((nch, nf, nf))
        H = np.zeros((nf, nf))

        def fac(c):
            return c.sign * (c.n / (c.n - 1) if self.small else 1.0)
        for blk, pstart in _blocks(st):
            npan = len(pstart) - 1
            CL = blk.get("cl", np.zeros((len(blk["t"]), 0), np.int64))
            CLacc = np.full((len(blk["t"]), len(general)), -1, np.int64)
            for q, c in enumerate(general):
                v = CL[:, c.col]
                CLacc[:, q] = np.where(v >= 0, v + c.offset, -1)
            Hp = np.zeros((npan, nf))
            kn.cellscore_pass(*self._kargs(blk, _chunks(pstart, nch), pstart), b_full,
                              blk["ag"], delta, Hf, bool(clus), rowmode, CLacc, Hp, AccH, SRH)
            if not clus:
                H += Hp.T @ Hp
            for c in clus:
                if c.is_panel:
                    H += fac(c) * (Hp.T @ Hp)
            for c in nested:
                codes = CL[pstart[:-1], c.col]
                ok = codes >= 0
                np.add.at(Nacc[id(c)], codes[ok], Hp[ok])
        AccH = AccH.sum(0)
        by_code = {}
        for c in nested:
            A = Nacc[id(c)]
            H += fac(c) * (A.T @ A)
            by_code[id(c)] = A
        for c in general:
            A = AccH[c.offset:c.offset + c.n_codes]
            H += fac(c) * (A.T @ A)
            by_code[id(c)] = A
        if rowmode:
            SRt = SRH.sum(0)
            for c in clus:
                if c.by_obs:
                    H += fac(c) * SRt
        return (H + H.T) / 2, by_code

    def _cells_inference(self, kind, c):
        """Windmeijer's correction and the Arellano-Bond tests with an
        absorbed effect, exactly as with its dummies among the regressors
        and instruments (where concentrating the effects out is an
        approximation): both depend on the effects' estimates, and through
        them on the effects' moments g2 = D'e, whose covariance S22 (cells
        x cells) is never formed, only its products with a few directions
        (`_cellscores`, a pass each).

        The estimates are linear in the moments. With the coefficients' b^ -
        b = G g~ (g~ = Z~'e), the effects' are delta^ - delta = -xshift (b^ -
        b) + DDinv (g2 - SA g~), SA the effects' moments' covariance with the
        instruments' times the instruments' weight (D'HZ~ A1 one-step, the
        cross moments times A2 two-step). So the functional tmp_b'b +
        tmp_d'delta of the AR tests has the influence (G'tmp_x - SA'rho)'g~
        + rho'g2, with tmp_x = tmp_b - xshift'tmp_d (the concentrated
        derivative) and rho = DDinv tmp_d. Windmeijer's correction is the
        same algebra one level down: the two-step estimates depend on the
        one-step coefficients and effects through the weight matrix,
        D_b (b1 - b) + D_d (delta1 - delta) = Dc (b1 - b) + R (g2 - SA1 g~)
        with Dc the concentrated derivative (`Draw`), D_d the derivative in
        the effects (the functionals' `Od`) and R = D_d DDinv.

        `kind`: "1iid" (one-step, non-robust), "1rob" (one-step, robust), "2"
        (two-step or iterated, uncorrected) or "2rob" (two-step with
        Windmeijer's correction). Returns (the corrected two-step variance
        or None, each AR test's variance less what `inference.ar_stat`
        computes from the concentrated terms)."""
        ab = self.absorb_info
        bmul, DDinv = ab["bmul"], ab["DDinv"]
        keep = c["keep"]
        k = len(keep)

        def dinv(M):
            return bmul(DDinv, M)

        def cols(a, b):
            return np.einsum("im,im->m", a, b)
        nar = 0
        if c["ar"] is not None:
            ar_sum, _, tmp, zhw, td, zhd = c["ar"]
            nar = len(ar_sum)
            tmpx = tmp[:, keep].T                       # k x m
            zhw, zhd = zhw.T, zhd.T                     # L x m, cells x m
            if kind == "1iid":
                zhw, zhd = zhw * c["s2ar"], zhd * c["s2ar"]
            rho = dinv(td.T)                            # cells x m
        SA1 = ab["S21"] @ c["A1raw"]
        SA = SA1 if kind[0] == "1" else c["cross"] @ c["A2"]
        extra = np.zeros(nar)
        if nar:
            # the effects' part of the estimates' covariance with the tests'
            extra -= 2 * cols(rho, zhd - SA @ zhw)
        if kind == "1iid":
            if nar:
                # the effects' variance: sigma^2 times the Schur complement of
                # D'HD in the one-step weight's inverse
                S21r = ab["S21"].T @ rho
                extra += c["sig2"] * (cols(rho, bmul(ab["DHD"], rho))
                                      - cols(S21r, c["A1raw"] @ S21r))
            return None, extra
        S, Cr = c["S"], c["cross"]
        bS, dS = c["point_S"]
        if kind == "1rob":
            if nar:
                self.log("pass 3: the absorbed effects' moments")
                H, _ = self._cellscores(bS, dS, rho)
                a = c["G1"].T @ tmpx
                lam = a - SA.T @ rho
                extra += (cols(lam, S @ lam) - cols(a, S @ a) + 2 * cols(rho, Cr @ lam)
                          + np.diag(H))
            return None, extra
        A2, V2 = c["A2"], c["V2"]
        if kind == "2":
            if nar:
                self.log("pass 4: the absorbed effects' moments")
                H, _ = self._cellscores(bS, dS, rho)
                CAr = Cr.T @ rho
                extra += np.diag(H) - cols(CAr, A2 @ CAr)
            return None, extra
        # Windmeijer: the derivatives in the one-step effects from the
        # functionals P (the two-step coefficients' influence) and, for the
        # AR tests, each test's own influence (lam2, rho)
        G1 = c["G1"]
        P = V2 @ c["ZXk"].T @ A2
        Lf, Rf, beta = P, np.zeros((k, ab["ncell"])), None
        if nar:
            lam2 = P.T @ tmpx - A2 @ (Cr.T @ rho)
            Lf = np.vstack([P, lam2.T])
            Rf = np.vstack([Rf, rho.T])
            if any(cc.col >= 0 for cc in self.clusters):
                self.log("pass 4: the absorbed effects' moments by cluster")
                _, bc = self._cellscores(bS, dS, rho)
                beta = {key: np.hstack([np.zeros((v.shape[0], k)), v]) for key, v in bc.items()}
        self.log("pass 4: Windmeijer correction")
        Draw, Ob, Od = self._windmeijer(c["b1"], c["v"], c["info_S"], delta=c["d1"],
                                        xshift=c["xs1"], funcs=(Lf, Rf, beta))
        Dc = P @ Draw[:, keep]
        Dd = Od[:k]
        R = dinv(Dd.T).T
        dxs = (c["xs1"] - c["xs2"])[:, keep]
        Hf = [R.T]
        if nar:
            wx1 = Ob[k:, keep].T
            wd = Od[k:].T
            nu = dinv(wd)
            Hf += [rho, nu]
        self.log("pass 5: the absorbed effects' moments")
        H, _ = self._cellscores(bS, dS, np.hstack(Hf))
        # V2 + D V1r D' + D V2 + V2 D', with the effects among the
        # parameters: D V1r D' is the variance of Dc (b1 - b) + R (g2 - SA1
        # g~), and the effects' part of V2 enters D V2 through xshift2
        Lam1 = Dc @ G1 - R @ SA1
        LCR = Lam1 @ Cr.T @ R.T
        Vc = Lam1 @ S @ Lam1.T + LCR + LCR.T + H[:k, :k]
        E = Dc + Dd @ dxs
        V2r = V2 + Vc + E @ V2 + V2 @ E.T
        V2r = (V2r + V2r.T) / 2
        if nar:
            m = nar
            Hrr = np.diag(H[k:k + m, k:k + m])
            Hnn = np.diag(H[k + m:, k + m:])
            Hnr = np.diag(H[k + m:, k:k + m])
            CAr, CAn = Cr.T @ rho, Cr.T @ nu
            cV2c = cols(tmpx, V2 @ tmpx) + Hrr - cols(CAr, A2 @ CAr)
            mu = G1.T @ wx1 - SA1.T @ nu
            wV1w = cols(mu, S @ mu) + 2 * cols(nu, Cr @ mu) + Hnn
            wx2 = wx1 + dxs.T @ wd
            wV2c = cols(wx2, V2 @ tmpx) + Hnr - cols(CAn, A2 @ CAr)
            extra += cV2c + wV1w + 2 * wV2c - cols(tmpx, V2r @ tmpx)
        return V2r, extra

    def _partial(self, cells, live, ZX, ZY, S1, XX, K):
        """The moments net of an absorbed effect (Frisch-Waugh-Lovell for
        GMM). With D the absorbed dummies of the transformed rows and W the
        weights (the regressors' dummies are W D, the instruments' D):
        instruments Z~ = Z - D Mz with Mz = (D'WD)^- D'WZ, the dummies'
        coefficients concentrated out of the regressors and outcome by
        Mx = (D'WD)^- D'[X y], and Z~'HZ~ = Z'HZ - Z'HD Mz - Mz'D'HZ +
        Mz'D'HD Mz. Concentrating out the effects, which their own
        (IV-style) moments exactly identify, leaves the same estimates of the
        other coefficients as including the dummies. D'WD and D'HD are block
        diagonal, one T x T block per group (diagonal and tridiagonal with
        first differences); with forward orthogonal deviations they are
        singular (the deviations of a full set of dummies sum to zero), and
        their generalized inverse leaves the level effects determined up to
        a constant per group, which the estimates do not depend on."""
        DDx, DHD = cells["DDx"], cells["DHD"]
        ng, T = DDx.shape[0], DDx.shape[1]
        DDinv = np.linalg.pinv(DDx, rcond=1e-12, hermitian=True)
        n_cells = int(np.linalg.matrix_rank(DDx, hermitian=True).sum())

        def bmul(M, X):
            return np.einsum("gst,gtc->gsc", M, X.reshape(ng, T, -1)).reshape(ng * T, -1)
        CZw, CHZ, CX = cells["CZw"][:, live], cells["CHZ"][:, live], cells["CX"]
        Mz = bmul(DDinv, CZw)
        Mx = bmul(DDinv, CX)
        DHDMz = bmul(DHD, Mz)
        ZXt = ZX - Mz.T @ CX[:, :K]
        ZYt = ZY - Mz.T @ CX[:, K]
        S1t = S1 - CHZ.T @ Mz - Mz.T @ CHZ + Mz.T @ DHDMz
        S1t = (S1t + S1t.T) / 2
        XXt = XX - CX.T @ Mx
        ab = dict(ncell=ng * T, DDinv=DDinv, bmul=bmul, Mz=Mz, Mx=Mx, S21=CHZ - DHDMz,
                  DHD=DHD, CZw=CZw, CX=CX, n_cells=n_cells)
        return ab, ZXt, ZYt, S1t, XXt

    def _cell_effects(self, b_full, keep, A, S21, ZXk, ZY, bk):
        """The effects at b in the GMM with the dummies -- their projection
        of y - x'b, less the part their moments' covariance with the
        others' (S21) assigns them -- and the derivative of the concentrated
        residual with respect to b, as x - xshift."""
        ab = self.absorb_info
        Mx, bmul, DDinv = ab["Mx"], ab["bmul"], ab["DDinv"]
        K = Mx.shape[1] - 1
        q = bmul(DDinv, S21 @ (A @ (ZY - ZXk @ bk)))[:, 0]
        delta = Mx[:, K] - Mx[:, :K] @ b_full - q
        w = bmul(DDinv, S21 @ (A @ ZXk))
        xshift = Mx[:, :K].copy()
        xshift[:, keep] -= w
        return np.ascontiguousarray(delta), np.ascontiguousarray(xshift)

    def _absorb_args(self, blk, delta, xshift):
        ab = self.absorb_info
        if not ab:
            z = np.zeros((1, 1))
            return np.zeros(0, np.int64), z, np.zeros(1), z
        return blk["ag"], ab["Mz"], delta, xshift

    # ------------------------------------------------------------------ estimation
