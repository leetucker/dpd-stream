"""The fit with Ahn and Schmidt's nonlinear moments: the first step
(xtdpdgmm's weights) and the second step with the nonlinear solver.
"""

from __future__ import annotations

import numpy as np
from hdfe_stream.core.report import warn

from . import inference as inf


class _NonlinearMoments:
    """Methods of `estimator.DPD`."""

    def _nl_first_step(self, ZXk, ZY, b1, A1raw, keep, res, S1=None):
        """The one-step estimate with Ahn and Schmidt's moments, as xtdpdgmm:
        the initial weight is the linear moments' (Z'HZ)^- and, treated as
        independent of them, the nonlinear moments' 1 / (number of panels)
        times `nl_weight` (the same for every moment, however many panels
        contribute to it) (xtdpdgmm's nl(, weight()); 0 ignores them in the first
        step, which is then the linear one-step estimate). Gauss-Newton from
        the linear one-step estimate."""
        ns = self.nl_sums
        self.mom = inf.Moments(ZXk, ZY, ns["s0"], ns["s1"], ns["s2"], keep)
        j = ZXk.shape[0]
        W1 = np.zeros((j + self.Lnl, j + self.Lnl))
        W1[:j, :j] = A1raw
        W1[j:, j:] = np.eye(self.Lnl) * self.nl_weight / self.n_groups_nl
        if self.nl_iid_h is not None and S1 is not None:
            # nl(iid): the linear moments it adds belong to the linear block, whose
            # weight is (Z'HZ)^- over all of them (xtdpdgmm's gmmiv(L.y, iid)); they
            # are the last of the nonlinear block (indices >= T)
            zhc, chc = self.nl_iid_h
            if self.P is not None:
                zhc = self.P.T @ zhc
            nlin = zhc.shape[1]
            pos = list(range(j)) + list(range(j + self.Lnl - nlin, j + self.Lnl))
            Saug = np.block([[S1, zhc], [zhc.T, chc]])
            W1[:, [i for i in pos]] = 0.0
            W1[[i for i in pos], :] = 0.0
            W1[np.ix_(pos, pos)] = inf.invsym((Saug + Saug.T) / 2)[0]
            rest = list(range(j, j + self.Lnl - nlin))
            W1[np.ix_(rest, rest)] = np.eye(len(rest)) * self.nl_weight / self.n_groups_nl
        self.W1nl = W1
        if self.nl_weight == 0 or self.Lnl == 0:
            return b1
        self.log("one-step GMM with the nonlinear moments (Gauss-Newton)")
        b1n, ok = self.mom.solve(W1, b1)
        res["nl_first_converged"] = bool(ok)
        if not ok:
            warn("the nonlinear one-step GMM did not converge (the Ahn-Schmidt objective need "
                 "not be convex)", self.logger)
        res["b1"] = b1n
        return b1n

    def _two_step_nl(self, info, b1, keep, full, steps, Qk, wttot, res):
        """GMM with Ahn and Schmidt's moments after the first step
        (`_nl_first_step`): the first step's sandwich variance; for one-step
        fits that is the result (with Hansen's test from a two-step estimate,
        as for linear one-step robust fits). Otherwise the weight from the
        first step's residuals, the estimate by Gauss-Newton from the first
        step's, Windmeijer's correction in its general form (Windmeijer 2005,
        section 4), with the first step's sandwich variance."""
        mom, W1 = self.mom, self.W1nl
        S = info["S"]
        G1 = mom.G(b1)
        B1, _ = inf.invsym(G1.T @ W1 @ G1)
        V1robust = B1 @ G1.T @ W1 @ S @ W1 @ G1 @ B1
        V1robust = (V1robust + V1robust.T) / 2
        res["V1"] = V1robust
        A2, _ = inf.invsym(S)
        res["pinv2"] = bool((np.diag(A2) == 0).any())
        b2, ok = mom.solve(A2, b1)
        iterations = 2
        if steps == "iterated":
            for it in range(self.maxiter):
                self.log(f"iterated GMM: step {it + 3}")
                info = self._resid(full(b2), 0)
                S = info["S"]
                A2, _ = inf.invsym(S)
                b_new, ok = mom.solve(A2, b2)
                iterations = it + 3
                done = np.max(np.abs(b_new - b2)) <= self.tol * (1 + np.max(np.abs(b2)))
                b2 = b_new
                if done:
                    break
            else:
                warn(f"iterated GMM did not converge in {self.maxiter} steps", self.logger)
            res["iterations"] = iterations
        if not ok:
            warn("the nonlinear GMM step did not converge (the Ahn-Schmidt objective need not "
                 "be convex); try other lag limits or collapse=True", self.logger)
        res["nl_converged"] = (res.get("nl_first_converged", True) if steps == 1
                               else bool(ok))
        G = mom.G(b2)
        V2, _ = inf.invsym(G.T @ A2 @ G)
        g2 = mom.g(b2)
        A2g = A2 @ g2
        hansen = float(g2 @ A2g)
        j0_nl = A2.shape[0] - int((np.diag(A2) == 0).sum())
        res["j0_nl"] = j0_nl
        res["hansen_df"] = j0_nl - len(keep)
        sig2 = inf.sigma2(Qk, b1 if steps == 1 else b2, self.system, self.h, self.orth, wttot)
        if steps == 1:
            # the one-step estimates; the AR tests came with pass 2
            ar_mode = 1 if self.artests else 0
            return (S, V1robust, A2, b2, V2, None, hansen, info, ar_mode, B1, W1, V1robust, b1,
                    sig2, mom, 1)
        V2robust = None
        if steps == 2 and self.robust:
            self.log("pass 3: Windmeijer correction and Arellano-Bond tests")
            Draw = self._windmeijer(full(b1), A2g, info)[:, keep]
            D = V2 @ (-G).T @ A2 @ Draw
            V2robust = V2 + D @ V1robust @ D.T + D @ V2 + V2 @ D.T
            V2robust = (V2robust + V2robust.T) / 2
        pVr = V2robust if (steps == 2 and self.robust) else V2
        ar_mode = 0
        if self.artests:
            info = self._resid(full(b2), 1, need_S=False)
            ar_mode = 1
        return (S, V1robust, A2, b2, V2, V2robust, hansen, info, ar_mode, V2, A2, pVr, b2,
                sig2, mom, iterations)
