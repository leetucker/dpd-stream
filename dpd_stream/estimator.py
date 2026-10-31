"""The fit: pass 0, then the passes over the blocks, then the algebra.

Passes (each one read of the blocks, panels in parallel):

1. moments that do not depend on the estimates: Z'X, Z'y, sum_i Z_i'HZ_i,
   the cross-products for sigma^2 and collinearity, counts (and the PCA
   moments). The one-step estimate, sigma^2 and the Sargan test follow.
2. at the one-step estimate: the moment covariance S = sum_c g_c g_c'
   (g_c = Z_c'e_c by panel, or by cluster), the robust one-step vcov, the
   two-step estimate and the Hansen test; and, for a one-step fit, the
   Arellano-Bond tests. Iterated GMM repeats this pass.
3. (two-step) at both estimates: the sums of Windmeijer's correction and the
   Arellano-Bond tests at the two-step residuals.

The difference-in-Hansen tests, the MMSC and everything else are computed
from these sums.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numba
import numpy as np
import polars as pl
from hdfe_stream.core.report import warn
from hdfe_stream.core.store import PolarsEngine
from hdfe_stream.core.resources import MemoryBudget, peak_rss_gb, reset_peak_rss, thread_scope
from hdfe_stream.core.utils import as_lazy, single_blas
from hdfe_stream.core.workspace import Run, _base_dir

from . import inference as inf
from . import kernels as kn
from . import layout as lay_mod
from . import store as store_mod
from ._absorb import _Absorb
from ._blocks import _blocks, _chunks
from ._frontend import parse
from ._idtests import _IdTests
from ._nl import _NonlinearMoments
from ._passes import _Passes

WORKDIR_ENV = "DPD_STREAM_WORKDIR"


def _log_fn(verbose, logger, log_level):
    def log(msg):
        if logger is not None:
            logger.log(log_level, msg)
        elif verbose:
            print(f"[dpd {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    return log


def _resolve_vcov(vcov):
    """vcov -> (robust, cluster variables, key)."""
    if vcov is None:
        vcov = "robust"
    if isinstance(vcov, dict):
        (kind, var), = vcov.items()
        vcov = f"{kind}:{var}"
    v = str(vcov)
    low = v.lower()
    if low in ("iid", "unadjusted", "uncorrected"):
        return False, (), "iid"
    if low in ("robust", "hetero", "hc1"):
        return True, (), "robust"
    kind, _, var = v.partition(":")
    if kind.upper() == "CRV1" and var:
        vars_ = tuple(x.strip() for x in var.split("+") if x.strip())
        return True, vars_, "CRV1:" + "+".join(vars_)
    raise ValueError(f"unsupported vcov {vcov!r}: use 'iid', 'robust' (clustered by panel, "
                     "Windmeijer-corrected for two-step), or {'CRV1': 'var'} / "
                     "{'CRV1': 'var1+var2'}")


@dataclass
class _Sums:
    ZX: np.ndarray
    ZY: np.ndarray
    S1: np.ndarray
    Q: np.ndarray
    XX: np.ndarray
    colnz: np.ndarray
    n_obs: int
    n_groups: int
    g_min: int
    g_max: int
    n_diff: int
    wsum: float
    wsum_diff: float = 0.0
    ZZp: np.ndarray = None
    Zsp: np.ndarray = None
    clpres: np.ndarray = None
    cells: dict = None
    nl: dict = None


@dataclass
class _Fit:
    """Everything one fit computes; the result is built from it."""
    b: np.ndarray = None
    V: np.ndarray = None
    names: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


class DPD(_Passes, _Absorb, _IdTests, _NonlinearMoments):
    """One dynamic panel GMM fit (see `dpd_stream` for the options)."""

    def __init__(self, fml, data, panel, *, gmm=(), iv=(), system=False, steps=2,
                 transform="fd", h=None, vcov="robust", small=False, artests=2,
                 collapse=False, constant=True, pca=False, components=0, weights=None,
                 weights_type="aweights", tdelta=None, diffsargan=True, workdir=None,
                 storage="disk", block_mb=64, tol=1e-9, maxiter=100, n_threads=None,
                 idtests="auto", nl=None, nl_weight=1.0, verbose=True, logger=None,
                 log_level=20,
                 polars_engine="cpu", max_memory_gb=None, memory_reserve=None):
        if not (isinstance(panel, (tuple, list)) and len(panel) == 2):
            raise ValueError("panel must be (panel id, time), e.g. panel=('firm', 'year')")
        if transform not in ("fd", "fod"):
            raise ValueError("transform must be 'fd' (first differences) or 'fod' (forward "
                             "orthogonal deviations)")
        if steps not in (1, 2, "iterated"):
            raise ValueError("steps must be 1, 2 or 'iterated'")
        if h is None:
            h = 3
        if h not in (1, 2, 3):
            raise ValueError("h must be 1, 2 or 3 (xtabond2's h())")
        if storage not in ("disk", "memory"):
            raise ValueError("storage must be 'disk' or 'memory'")
        if weights_type not in ("aweights", "pweights", "fweights"):
            raise ValueError("weights_type must be 'aweights', 'pweights' or 'fweights'")
        if artests < 0:
            raise ValueError("artests must be >= 0")
        self.fml, self.data, self.panel = fml, data, tuple(panel)
        self.gmm, self.iv = list(gmm or ()), list(iv or ())
        self.system, self.steps, self.orth = bool(system), steps, transform == "fod"
        self.h, self.vcov, self.small, self.artests = h, vcov, bool(small), int(artests)
        self.collapse, self.constant = bool(collapse), bool(constant)
        self.pca, self.components = bool(pca), int(components)
        self.weights, self.weights_type, self.tdelta = weights, weights_type, tdelta
        self.diffsargan = diffsargan
        self.workdir, self.storage, self.block_mb = workdir, storage, block_mb
        self.tol, self.maxiter = tol, maxiter
        if idtests not in ("auto", True, False):
            raise ValueError("idtests must be 'auto', True or False")
        self.idtests = idtests
        if nl not in (None, "noserial", "iid"):
            raise ValueError("nl must be None, 'noserial' (Ahn and Schmidt's moments under no "
                             "serial correlation) or 'iid' (and homoskedasticity)")
        if not nl_weight >= 0:
            raise ValueError("nl_weight must be >= 0")
        self.nl, self.nl_weight = nl, float(nl_weight)
        self.nlkind = {None: 0, "noserial": 1, "iid": 2}[nl]
        self.budget = MemoryBudget(max_memory_gb, memory_reserve)
        self._threads = n_threads                   # held by fit(), then restored
        self.n_threads = n_threads or numba.get_num_threads()
        self.log = _log_fn(verbose, logger, log_level)
        self.logger = logger
        self.engine = PolarsEngine(polars_engine)

    # ------------------------------------------------------------------ driver
    def fit(self):
        reset_peak_rss()
        with thread_scope(self._threads):
            self.n_threads = self._threads or numba.get_num_threads()   # the accumulators' count
            out = self._fit()
        rss = peak_rss_gb()
        out.info["peak_rss_gb"] = None if rss is None else round(rss, 3)
        return out

    def _copies(self, per_copy_bytes, want):
        """How many per-thread copies of an accumulator of `per_copy_bytes` fit in
        half the memory headroom: `want` (one per thread) if they do, fewer
        otherwise, and at least one. Fewer copies is fewer threads on that
        pass, not a different answer."""
        fit = int(0.5 * self.budget.headroom_gb() * 1e9 // max(per_copy_bytes, 1))
        return max(1, min(want, fit))

    def profile(self):
        """The memory and disk this fit would need (a `Profile`), from one scan
        of the data; nothing is fitted. See `dpd_stream.profile`."""
        from .profile import profile
        return profile(self)

    def _prepare(self):
        """Parse the formula and settle the options against the data's schema;
        returns the LazyFrame and whether this is a one-step non-robust fit.
        Nothing is read but the levels of categorical variables."""
        lf = as_lazy(self.data)
        schema = lf.collect_schema()

        def levels(v):
            return self.engine.collect(lf.select(pl.col(v).drop_nulls().unique().sort()),
                                       f"levels of {v}")[v].to_list()
        spec = parse(self.fml, schema, levels, gmm=self.gmm, iv=self.iv, system=self.system,
                     panel=self.panel)
        if self.collapse:
            from dataclasses import replace
            spec.gmm = [(replace(g, collapse=True), ts) for g, ts in spec.gmm]
        self.absorb_g = None
        if spec.absorb:
            self.absorb_g = self._check_absorb(spec.absorb)
        if not spec.gmm and not spec.iv and not self.system:
            raise ValueError("no instruments: add gmm() and/or iv() terms")
        self.spec = spec
        robust, cvars, key = _resolve_vcov(self.vcov)
        if self.nl:
            if spec.absorb:
                raise NotImplementedError(
                    f"nl={self.nl!r} with an absorbed effect ('| {spec.absorb[0]}'): Ahn and "
                    "Schmidt's moments are nonlinear in the absorbed effects, so they cannot "
                    "be partialled out; '| year' (time effects, estimated as dummies) works")
            for opt, bad in (("clustering", bool(cvars)), ("weights", bool(self.weights)),
                             ("pca", self.pca)):
                if bad:
                    raise NotImplementedError(f"nl={self.nl!r} with {opt} is not available")
            if self.steps == 1 and not robust:
                raise ValueError("a one-step fit with nonlinear moments has only the sandwich "
                                 "(robust) variance: use vcov='robust'")
        if self.weights and self.weights_type == "pweights" and self.steps == 1:
            robust = True
        self.robust, self.cvars, self.vkey = robust, cvars, key
        onestepnonrobust = self.steps == 1 and not robust
        return lf, onestepnonrobust

    def _fit(self):
        t0 = time.time()
        lf, onestepnonrobust = self._prepare()
        spec, cvars = self.spec, self.cvars

        base = _base_dir(self.workdir or os.environ.get(WORKDIR_ENV))
        run = Run(base, auto_cleanup=True)
        try:
            self.log("pass 0: reading the source")
            st = store_mod.build(lf, spec.bases, self.panel, tdelta=self.tdelta,
                                 cluster=cvars, weights=self.weights,
                                 fweights=self.weights_type == "fweights",
                                 absorb_g=self.absorb_g, engine=self.engine,
                                 workdir=run.path, in_memory=self.storage == "memory",
                                 block_mb=self.block_mb, log=self.log)
            self.store = st
            lay = lay_mod.build(spec, st.T, self.system, self.orth, self.constant)
            self.lay = lay
            for note in lay.notes:
                self.log(note)
            self.log(f"{st.n_rows:,} rows, {st.n_panels:,} panels, {st.T} periods; "
                     f"{lay.j:,} instrument columns before dropping empty ones")
            with single_blas():
                out = self._estimate(onestepnonrobust)
        finally:
            run.cleanup()
        out.info["seconds"] = time.time() - t0
        return out

    def _arrays(self):
        return self.lay.arrays()

    def _kargs(self, blk, cb, pstart):
        L = self.L
        B = blk["X"]
        W = blk.get("w")
        use_w = W is not None
        if W is None:
            W = np.ones(len(blk["t"]))
        else:
            W = W * self.wscale
        return (cb, pstart, blk["t"], B, W, use_w, self.lay.T, self.system, self.orth,
                L["term_base"], L["term_k"], L["term_diff"], L["ty"], L["tx"], L["cmp_terms"],
                L["iv_term"], L["iv_eq"], L["iv_pass"], L["iv_mz"],
                L["g_nb"], L["g_t0"], L["g_terms"], L["g_eqrow"], L["g_fd"], L["g_coll"],
                L["g_orth"], L["g_a"], L["g_b"], L["g_lp"], L["lag_lo"], L["lag_hi"],
                L["lag_col"], L["g_extra"], L["g_xlag"], L["g_xlo"], L["g_xhi"], L["g_xcol"],
                L["g_xsh"], self.maxnz)

    def _pass1(self):
        lay, st = self.lay, self.store
        jf, K = lay.j, lay.K
        nch = self.n_threads
        # S1, and Z'Z (the identification tests and PCA need it) of the same size
        per = jf * jf * 8 * (2 if self.idtests is not False else 1)
        if not self.budget.fits(per / 1e9):
            warn(f"{jf:,} instrument columns: even one set of moment matrices needs "
                 f"{per / 1e9:.1f} GB and {self.budget.headroom_gb():.1f} GB is free; consider "
                 "collapse=True, tighter lag limits, idtests=False or a larger max_memory_gb",
                 self.logger)
        fewer = self._copies(per, nch)
        if fewer < nch:
            self.log(f"pass 1: {fewer} of {nch} per-thread moment accumulators fit in memory "
                     f"({per / 1e9:.2f} GB each); the pass runs on {fewer} threads")
            nch = fewer
        H = lay_mod.h_matrix(self.h, self.orth, self.orth, self.system, lay.T)
        hp, hi, hv = lay_mod.csr(H)
        self.H = H
        ZX = np.zeros((nch, jf, K))
        ZY = np.zeros((nch, jf))
        S1 = np.zeros((nch, jf, jf))
        Q = np.zeros((nch, 2, K + 1, K + 1))
        XX = np.zeros((nch, 2, K + 1, K + 1))
        colnz = np.zeros((nch, jf), np.bool_)
        cnt = np.zeros((nch, 9))
        jg0 = lay.j_iv
        jg = lay.j_gmm if self.pca else 0
        # the identification tests need Z'Z of every column (which then
        # serves PCA too)
        self.want_zz = self.idtests is not False
        if self.want_zz:
            jg, jg0 = jf, -1
        ZZp = np.zeros((nch, jg, jg))
        Zsp = np.zeros((nch, jg))
        ncl = max((c.n_codes for c in st.clusterings if c.col >= 0), default=0)
        ncol = sum(1 for c in st.clusterings if c.col >= 0)
        clpres = np.zeros((max(ncol, 1), max(ncl, 1)), np.bool_)
        self.maxnz = lay.j_iv + lay.max_gmm_nnz
        cells = None
        if self.absorb_g is not None:
            ng = st.n_absorb_groups
            ncell = ng * lay.T
            if not self.budget.fits(ncell * jf * 16 / 1e9):
                raise MemoryError(f"{ncell:,} absorbed cells x {jf:,} instruments: the cell "
                                  f"sums need {ncell * jf * 16 / 1e9:.1f} GB and "
                                  f"{self.budget.headroom_gb():.1f} GB is free "
                                  f"({self.budget}); raise max_memory_gb or absorb a coarser effect")
            nchc = self._copies(ncell * jf * 16, nch)
            cells = dict(DDx=np.zeros((nchc, ng, lay.T, lay.T)),
                         DHD=np.zeros((nchc, ng, lay.T, lay.T)),
                         CZw=np.zeros((nchc, ncell, jf)), CHZ=np.zeros((nchc, ncell, jf)),
                         CX=np.zeros((nchc, ncell, K + 1)), nch=nchc)
        mins = []
        T = lay.T
        g_xstop = np.array([g.xstop for g in lay.gmm], np.int64)
        nlag = max([g.xstop - g.xlag + 1 for g in lay.gmm if g.extra and g.collapse] or [1])
        nbmax = max([len(g.terms) for g in lay.gmm] or [1])
        xnz = np.zeros((nch, max(len(lay.gmm), 1), nbmax, nlag), np.bool_)
        Tn = T * (2 if self.nlkind == 2 else 1)
        nls0 = np.zeros((nch, Tn))
        nls1 = np.zeros((nch, Tn, K))
        nls2 = np.zeros((nch, Tn, K, K))
        nlcnt = np.zeros((nch, Tn))
        nlpan = np.zeros(nch)
        # nl(iid): Z'HC and C'HC of the linear moments it adds (see `nl_pass1`)
        iid_h = self.nlkind == 2 and not self.orth
        zhc = np.zeros((nch, jf, Tn // 2) if iid_h else (nch, 1, 1))
        chc = np.zeros((nch, Tn // 2, Tn // 2) if iid_h else (nch, 1, 1))
        for blk, pstart in _blocks(st):
            cb = _chunks(pstart, nch)
            CL = blk.get("cl", np.zeros((len(blk["t"]), 0), np.int64))
            cnt[:, 2] = 0
            cnt[:, 3] = 0
            kn.pass1(*self._kargs(blk, cb, pstart), hp, hi, hv, CL, clpres,
                     self.pca or self.want_zz, jg0,
                     ZX, ZY, S1, Q, XX, colnz, cnt, ZZp, Zsp, xnz, g_xstop)
            if self.nl:
                kn.nl_pass1(*self._kargs(blk, cb, pstart), self.collapse, self.nlkind, nls0,
                            nls1, nls2, nlcnt, nlpan, hp, hi, hv, zhc, chc)
            if cells is not None:
                kn.pass1_cells(*self._kargs(blk, _chunks(pstart, cells["nch"]), pstart),
                               hp, hi, hv, blk["ag"], cells["DDx"], cells["DHD"], cells["CZw"],
                               cells["CHZ"], cells["CX"])
            used = cnt[:len(cb) - 1]
            nonempty = used[:, 0] > 0
            if np.any(nonempty):
                mins.append((used[nonempty, 2].min(), used[nonempty, 3].max()))
            cnt[:, 2] = 0
            cnt[:, 3] = 0
        tot = cnt.sum(0)
        g_min = int(min(m[0] for m in mins)) if mins else 0
        g_max = int(max(m[1] for m in mins)) if mins else 0
        self._pca_weight_total = float(tot[7])
        self._xnz = xnz.any(0)
        return _Sums(ZX=ZX.sum(0), ZY=ZY.sum(0), S1=S1.sum(0), Q=Q.sum(0), XX=XX.sum(0),
                     colnz=colnz.any(0), n_obs=int(tot[0]), n_groups=int(tot[1]),
                     g_min=g_min, g_max=g_max, n_diff=int(tot[4]), wsum=float(tot[5]),
                     wsum_diff=float(tot[8]),
                     ZZp=ZZp.sum(0) if (self.pca or self.want_zz) else None,
                     Zsp=Zsp.sum(0) if self.pca else None,
                     clpres=clpres,
                     cells=None if cells is None else {k: v.sum(0) for k, v in cells.items()
                                                       if k != "nch"},
                     nl=None if not self.nl else dict(s0=nls0.sum(0), s1=nls1.sum(0),
                                                      s2=nls2.sum(0), n=nlcnt.sum(0),
                                                      npan=float(nlpan.sum()),
                                                      zhc=zhc.sum(0) if iid_h else None,
                                                      chc=chc.sum(0) if iid_h else None))

    def _shift_extra(self):
        """xtabond2's search for system GMM's collapsed levels instruments:
        a variable's column whose first lag (the difference at lag a - 1) is
        zero throughout the sample moves to deeper lags until it has values,
        up to lag b - 1, with one budget of moves for the group's variables
        in order. (xtabond2 moves the column a row at a time across the
        whole data set; this moves it within each panel.) Returns whether
        any moved."""
        moved = False
        for gi, g in enumerate(self.lay.gmm):
            if not (g.extra and g.collapse):
                continue
            lag, stop = 0, g.xstop - g.xlag
            xsh = []
            for v in range(len(g.terms)):
                sh = 0
                while lag < stop and not self._xnz[gi, v, sh]:
                    lag += 1
                    sh += 1
                xsh.append(sh)
            if any(xsh):
                g.xsh = xsh
                moved = True
                self.lay.notes.append(f"{g.name}: levels instruments moved to deeper lags, "
                                      f"as xtabond2 does ({xsh})")
        return moved

    def _cluster_plan(self, sums):
        """Classify the clusterings and count each one's clusters."""
        plan = []
        st = self.store
        col = 0
        for c in st.clusterings:
            if c.col >= 0:
                c.n = int(sums.clpres[col].sum())
                col += 1
            elif c.is_panel:
                c.n = sums.n_groups
            else:                               # by observation
                c.n = sums.n_obs
            plan.append(c)
        return plan

    def _estimate(self, onestepnonrobust):
        lay = self.lay
        self.L = self._arrays()
        self.wscale = 1.0
        self.Lnl, self.nlmap = 0, np.full(1, -1, np.int64)
        self.log("pass 1: moments")
        sums = self._pass1()
        if self._shift_extra():
            self.L = self._arrays()
            self.log("pass 1 again, with the levels instruments xtabond2 moves to deeper lags")
            sums = self._pass1()
        K = lay.K
        system, h, orth = self.system, self.h, self.orth
        consopt = system and self.constant
        if sums.n_obs == 0:
            raise ValueError("no observations: every row lacks a regressor, the outcome, "
                             "an instrument or the lags they need")
        # weights: xtabond2 rescales them to sum to the number of observations
        n_obs = sums.n_obs
        n_obs_eff = n_obs
        wttot = sums.n_diff if (system and h > 1) else n_obs
        c = 1.0
        if self.weights:
            if self.weights_type == "fweights":
                # xtabond2: the observations are the weights' sum; they are
                # rescaled to the count, except in a one-step non-robust fit
                # (or system GMM with h > 1, which keeps them as they are)
                n_obs = sums.wsum
                if onestepnonrobust:
                    n_obs_eff = n_obs
                if system and h > 1:
                    wttot = sums.wsum_diff
                else:
                    c = n_obs_eff / sums.wsum
                    wttot = n_obs_eff
            else:
                c = n_obs / sums.wsum
        self.wscale = c
        ZX, ZY, S1, Q = sums.ZX * c, sums.ZY * c, sums.S1 * c, sums.Q * c
        # drop the instrument columns that are empty in the sample (the
        # generalized inverse would drop them; the others keep their order)
        live = np.flatnonzero(sums.colnz)
        cmap = np.full(lay.j, -1, np.int64)
        cmap[live] = np.arange(len(live))
        self.live = live
        ZX, ZY, S1 = ZX[live], ZY[live], S1[np.ix_(live, live)]
        XXall = sums.XX[0] + sums.XX[1]
        self.absorb_info = ab = None
        if sums.cells is not None:
            # the cells' sums on the scale of the normalized weights
            cells = {kk: v * c for kk, v in sums.cells.items()}
            ab, ZX, ZY, S1, XXall = self._partial(cells, live, ZX, ZY, S1, XXall, K)
            self.absorb_info = ab
        # collinear regressors
        keep = inf.rmcoll(XXall[:K, :K], consopt)
        self.keep = keep
        dropped = [lay.xnames[k] for k in range(K) if k not in keep]
        for d in dropped:
            self.log(f"{d} dropped due to collinearity")
        ZXk = ZX[:, keep]
        Qk = np.empty((2, len(keep) + 1, len(keep) + 1))
        sel = list(keep) + [K]
        for e in range(2):
            Qk[e] = Q[e][np.ix_(sel, sel)]
        k = len(keep)

        # PCA of the GMM-style instruments (xtabond2's pca)
        P = None
        pca_info = None
        jiv_live = int((live < lay.j_iv).sum())
        ZY_live = ZY
        if self.pca and lay.j_gmm:
            P, pca_info = self._pca(sums, live, jiv_live, k)
            ZXk, ZY, S1 = P.T @ ZXk, P.T @ ZY, P.T @ S1 @ P
        self.cmap = cmap
        self.jc = len(live)
        self.Lnl, self.nlmap, self.nl_sums = 0, np.full(1, -1, np.int64), None
        if self.nl:
            ns = sums.nl
            live_nl = np.flatnonzero(ns["n"] > 0)
            if self.collapse:
                live_nl = live_nl[(live_nl == 0) | (live_nl == lay.T)]
            self.nlmap = np.full(len(ns["n"]), -1, np.int64)
            self.nlmap[live_nl] = np.arange(len(live_nl))
            self.Lnl = len(live_nl)
            self.n_groups_nl = ns["npan"]
            lin_nl = live_nl[live_nl >= lay.T] - lay.T      # nl(iid)'s added linear moments
            self.nl_iid_h = None
            if ns["zhc"] is not None and len(lin_nl):
                self.nl_iid_h = (ns["zhc"][np.ix_(live, lin_nl)] * c,
                                 ns["chc"][np.ix_(lin_nl, lin_nl)] * c)
            self.nl_sums = dict(s0=ns["s0"][live_nl], s1=ns["s1"][live_nl],
                                s2=ns["s2"][live_nl], n=ns["n"][live_nl])
            if self.Lnl == 0:
                warn("no panel has the periods the Ahn-Schmidt moments need", self.logger)
        self.P = P
        j = ZXk.shape[0]
        self.jc_final = j
        if j < k:
            raise ValueError(f"equation not identified: {k} regressors, {j} instruments")

        # one-step
        A1raw, _ = inf.invsym(S1)
        b1, V1raw = inf.gmm(ZXk, ZY, A1raw)
        self.clusters = self._cluster_plan(sums)
        info_ab = None
        if ab:
            # with the dummies' effects: sigma^2 and everything else come from
            # their residuals, in pass 2
            kf = np.zeros(K)
            kf[keep] = b1
            delta1, xshift1 = self._cell_effects(kf, keep, A1raw, ab["S21"], ZXk, ZY, b1)
            ar1 = (2 if onestepnonrobust else (1 if self.steps == 1 else 0)) \
                if self.artests else 0
            self.log("pass 2: residuals at the one-step estimates")
            info_ab = self._resid(kf, ar1, need_S=not onestepnonrobust, delta=delta1,
                                  xshift=xshift1, cross=self.steps != 1 or ar1 == 1)
            sig2 = info_ab["SSE"] / (2 - orth) / wttot
        else:
            sig2 = inf.sigma2(Qk, b1, system, h, orth, wttot)
        A1 = A1raw / sig2
        V1 = V1raw * sig2
        j0 = j - int((np.diag(A1) == 0).sum())
        if j0 > sums.n_groups:
            warn(f"{j0} instruments for {sums.n_groups} groups: the number of instruments "
                 "may be large relative to the number of groups (Roodman 2009)", self.logger)
        Ze1 = ZY - ZXk @ b1
        sargan = float(Ze1 @ A1 @ Ze1)
        sargan_df = j0 - k
        n_clusters = {"+".join(cc.vars): cc.n for cc in self.clusters if len(cc.vars) == 1}
        min_clust = min((cc.n for cc in self.clusters), default=None)

        def full(bk):
            out = np.zeros(K)
            out[keep] = bk
            return out

        def proj(M):
            return M if P is None else P.T @ M

        steps = self.steps
        res = dict(b1=b1, V1=V1, sig2_1=sig2, j=j, j0=j0, sargan=sargan, sargan_df=sargan_df,
                   n_obs=n_obs, n_groups=sums.n_groups, g_min=sums.g_min, g_max=sums.g_max,
                   dropped=dropped, keep=keep, pca=pca_info, n_clusters=n_clusters,
                   pinv1=bool((np.diag(A1raw) == 0).any()))
        hansen = None
        V1robust = V2 = V2robust = b2 = A2 = None
        iterations = 1
        mom = None
        # with an absorbed effect: where the moments' covariance was taken
        info_S = point_S = None
        if ab:
            point_S = (full(b1), delta1)
        if onestepnonrobust:
            S = S1 * sig2
            ar_mode = 2 if self.artests else 0
            pV, pA, pVr, bfin = V1, A1, V1, b1
            if ar_mode:
                if info_ab is not None:
                    info = info_ab
                else:
                    self.log("pass 2: Arellano-Bond tests")
                    info = self._resid(full(b1), ar_mode, need_S=False)
        else:
            ar_mode = 1 if (steps == 1 and self.artests) else 0
            if info_ab is not None:
                info = info_ab
            else:
                if self.nl:
                    b1 = self._nl_first_step(ZXk, ZY, b1, A1raw, keep, res, S1)
                self.log("pass 2: moment covariance at the one-step estimates")
                info = self._resid(full(b1), ar_mode)
            if self.nl:
                (S, V1robust, A2, b2, V2, V2robust, hansen, info, ar_mode, pV, pA, pVr,
                 bfin, sig2, mom, iterations) = self._two_step_nl(
                    info, b1, keep, full, steps, Qk, wttot, res)
            else:
                S = proj_S = self._proj_S(info["S"])
                VXZA = V1 @ ZXk.T @ A1
                V1robust = VXZA @ proj_S @ VXZA.T
                A2, _ = inf.invsym(proj_S)
                if (np.diag(A2) == 0).any():
                    warn("two-step estimated covariance matrix of moments is singular; using a "
                         "generalized inverse" + ("; difference-in-Hansen statistics may be "
                                                  "negative" if self.diffsargan else ""),
                         self.logger)
                res["pinv2"] = bool((np.diag(A2) == 0).any())
                b2, V2 = inf.gmm(ZXk, ZY, A2)
                if steps == "iterated":
                    b_prev = b2
                    for it in range(self.maxiter):
                        self.log(f"iterated GMM: step {it + 2}")
                        if ab:
                            # the effects at the previous estimates, from the
                            # weight and cross moments that gave them
                            dl, xs = self._cell_effects(full(b_prev), keep, A2, info["cross"],
                                                        ZXk, ZY, b_prev)
                            info = self._resid(full(b_prev), 0, delta=dl, xshift=xs,
                                               cross=True)
                            point_S = (full(b_prev), dl)
                        else:
                            info = self._resid(full(b_prev), 0)
                        S = self._proj_S(info["S"])
                        A2, _ = inf.invsym(S)
                        b2, V2 = inf.gmm(ZXk, ZY, A2)
                        iterations = it + 3
                        if np.max(np.abs(b2 - b_prev)) <= self.tol * (1 + np.max(np.abs(b_prev))):
                            break
                        b_prev = b2
                    else:
                        warn(f"iterated GMM did not converge in {self.maxiter} steps", self.logger)
                    res["iterations"] = iterations
                if steps != 1 and not ab:
                    sig2 = inf.sigma2(Qk, b2, system, h, orth, wttot)
                Ze2 = ZY - ZXk @ b2
                A2Ze = A2 @ Ze2
                hansen = float(Ze2 @ A2Ze)
                d2 = x2 = None
                if ab and steps != 1:
                    d2, x2 = self._cell_effects(full(b2), keep, A2, info["cross"], ZXk, ZY, b2)
                if steps == 2 and self.robust and not ab:
                    self.log("pass 3: Windmeijer correction")
                    v_full = self._unproj(A2Ze)
                    Draw = proj(self._windmeijer(full(b1), v_full, info)[:, keep])
                    V2robust = inf.windmeijer(V2, ZXk, A2, Draw, V1robust)
                if steps == 1:
                    pV, pA, pVr, bfin = V1, A1, V1robust, b1
                else:
                    pV, pA, bfin = V2, A2, b2
                    pVr = (V2robust if (steps == 2 and self.robust) else
                           (V2 if steps == "iterated" else (V2robust if self.robust else V2)))
                    info_S = info
                    if self.artests or ab:
                        self.log("pass 3: Arellano-Bond tests")
                        info = self._resid(full(b2), 1 if self.artests else 0, need_S=False,
                                           delta=d2, xshift=x2)
                        ar_mode = 1 if self.artests else 0
                        if ab:
                            sig2 = info["SSE"] / (2 - orth) / wttot
        ar_extra = None
        if ab:
            kind = ("1iid" if onestepnonrobust else "1rob" if steps == 1
                    else "2rob" if (steps == 2 and self.robust) else "2")
            if kind == "1rob":
                info_S = info
            ctx = dict(keep=keep, ZXk=ZXk, A1raw=A1raw, sig2=sig2,
                       s2ar=sig2 * (wttot / (wttot - k) if self.small else 1.0),
                       ar=info["ar"] if (self.artests and ar_mode) else None,
                       point_S=point_S, b1=full(b1), d1=delta1, xs1=xshift1)
            if kind != "1iid":
                ctx.update(G1=VXZA, S=S, cross=info_S.get("cross"))
            if kind[0] == "2":
                ctx.update(A2=A2, V2=V2, v=A2Ze, xs2=x2, info_S=info_S)
            V2c, ar_extra = self._cells_inference(kind, ctx)
            if V2c is not None:
                V2robust = pVr = V2c
        b = bfin
        if steps == 1:
            V = V1robust if self.robust else V1
        elif steps == 2:
            V = V2robust if self.robust else V2
        else:
            V = V2
        V = (V + V.T) / 2
        V_unadj = V.copy()

        # Arellano-Bond tests
        ar = []
        if self.artests and ar_mode:
            ar_sum, wHw, tmp, zhw = info["ar"][:4]
            if ar_mode == 2:
                # one-step non-robust: sigma^2 H in place of the outer products
                # (with the small-sample sigma^2, as xtabond2)
                s2ar = sig2 * (wttot / (wttot - k) if self.small else 1.0)
                wHw, zhw = wHw * s2ar, zhw * s2ar
            m2VZXA = -2 * pV @ (ZXk.T if not self.Lnl else -mom.G(b).T) @ pA
            for m in range(self.artests):
                ZHw = proj(zhw[m][:, None])[:, 0] if P is not None else zhw[m]
                z, p = inf.ar_stat(ar_sum[m], wHw[m], tmp[m][keep], ZHw, m2VZXA, pVr,
                                   0.0 if ar_extra is None else ar_extra[m])
                if ar_sum[m] == 0:
                    z, p = np.nan, np.nan
                ar.append(dict(order=m + 1, z=z, p=p))

        # small-sample adjustments
        df_r = None
        vfac = {"iid1": 1.0, "other": 1.0}
        # the absorbed cells count as parameters, as their dummies would
        k_eff = k + (ab["n_cells"] if ab else 0)
        if self.small:
            tmp_f = wttot / (wttot - k_eff)
            vfac["iid1"] = tmp_f
            if self.clusters:
                vfac["other"] = (n_obs - 1) / (n_obs - k_eff)
            else:
                vfac["other"] = ((n_obs - 1) / (n_obs - k_eff) * sums.n_groups
                                 / (sums.n_groups - 1))
            V = V * (vfac["iid1"] if onestepnonrobust else vfac["other"])
            sig2 = sig2 * tmp_f
        if ab:
            # the model test is of the reported coefficients only
            XXlev = XXall[np.ix_(keep, keep)]
        else:
            XXlev = (sums.XX[1] if system else sums.XX[0] + sums.XX[1])[np.ix_(keep, keep)]
        df_m = int(np.linalg.matrix_rank(XXlev)) - consopt if XXlev.size else 0
        if self.small:
            df_r = ((n_obs_eff - df_m) if onestepnonrobust
                    else (min_clust if self.clusters else sums.n_groups)) - consopt
        # absorbed time effects ('| year') are estimated as dummies, and not
        # reported: the model test and the results are of the others
        tnames = {t.name for t in self.spec.time_dummies}
        shown = [i for i, kk in enumerate(keep) if lay.xnames[kk] not in tnames]
        n_hidden = k - len(shown)
        if n_hidden:
            XXs = XXlev[np.ix_(shown, shown)]
            df_ms = int(np.linalg.matrix_rank(XXs)) - consopt if XXs.size else 0
            wald = inf.wald_all(b[shown], V[np.ix_(shown, shown)], df_ms, self.small, df_r)
        else:
            wald = inf.wald_all(b, V, df_m, self.small, df_r)

        dsar = []
        if self.diffsargan and P is None:
            groups = []
            for name, cols in lay_mod.diff_sargan_groups(lay):
                live_cols = {int(cmap[c]) for c in cols if cmap[c] >= 0}
                groups.append((name, live_cols))
            full_stat = hansen if hansen is not None else sargan
            Vfull = V2 if V2 is not None else V1
            if self.Lnl:
                groups.append(("nonlinear moments (Ahn-Schmidt)",
                               set(range(j, j + self.Lnl))))
                A1d = np.concatenate([np.diag(A1), np.ones(self.Lnl)])
                dsar = inf.diff_sargan_general(groups, res["j0_nl"], k, S, mom, A1d, Vfull,
                                               res["hansen_df"], full_stat, b)
            else:
                dsar = inf.diff_sargan(groups, j0, k, S, ZXk, ZY, np.diag(A1), Vfull,
                                       sargan_df, full_stat)
        res.update(b=b, V=V, V_unadj=V_unadj, b2=b2, V2=V2, V2robust=V2robust,
                   V1robust=V1robust, sig2=sig2, hansen=hansen, ar=ar, diffsargan=dsar,
                   df_m=df_m, df_r=df_r, wald=wald, names=[lay.xnames[i] for i in keep],
                   onestep_vcovs=({"robust": V1robust} if self.nl else
                                  {"iid": V1, **({"robust": V1robust}
                                                 if V1robust is not None else {})}),
                   steps=steps, robust=self.robust, wttot=wttot, wscale=c, vfac=vfac,
                   absorbed=(dict(term=self.spec.absorb[0], effects=ab["n_cells"],
                                  method="cells") if ab else None))
        if n_hidden:
            res["absorbed"] = dict(term=self.spec.time_effects, effects=n_hidden,
                                   method="dummies")
        res["coef_all"] = {lay.xnames[kk]: float(b[i]) for i, kk in enumerate(keep)}
        if hansen is not None:
            res["mmsc"] = inf.mmsc(hansen, res.get("j0_nl", j0), k, n_obs)
        res["idtests"] = None
        if self.want_zz:
            try:
                res["idtests"] = self._idtests(sums, ZX, ZY_live, keep, live, b, V)
            except (np.linalg.LinAlgError, ValueError) as err:
                warn(f"identification tests not computed: {err}", self.logger)
            if res["idtests"] and tnames:
                res["idtests"]["exogenous"] = [x for x in res["idtests"]["exogenous"]
                                               if x not in tnames]
        if n_hidden:
            self._hide(res, shown, tnames)
        return _Fit(b=res["b"], V=res["V"], names=res["names"], info=res)

    @staticmethod
    def _hide(res, shown, tnames):
        """Drop the absorbed time effects from the reported estimates."""
        for key in ("b", "b1", "b2"):
            if res.get(key) is not None:
                res[key] = res[key][shown]
        for key in ("V", "V_unadj", "V1", "V1robust", "V2", "V2robust"):
            if res.get(key) is not None:
                res[key] = res[key][np.ix_(shown, shown)]
        res["onestep_vcovs"] = {kk: v[np.ix_(shown, shown)]
                                for kk, v in res["onestep_vcovs"].items()}
        res["names"] = [res["names"][i] for i in shown]
        res["dropped"] = [d for d in res["dropped"] if d not in tnames]

    def _proj_S(self, S):
        return S if self.P is None else self.P.T @ S @ self.P

    def _unproj(self, v):
        return v if self.P is None else self.P @ v

    def _pca(self, sums, live, jiv_live, k):
        """xtabond2's pca: the GMM-style instruments replaced by their
        principal components (of their correlation matrix in the sample),
        as many as have eigenvalues of 1 or more (at least k - #IV)."""
        lay = self.lay
        g_live = live[live >= lay.j_iv] - (0 if self.want_zz else lay.j_iv)
        ZZ = sums.ZZp[np.ix_(g_live, g_live)]
        zs = sums.Zsp[g_live]
        n = self._pca_weight_total
        mu = zs / n
        C = ZZ - n * np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(C), 0, None))
        sd[sd == 0] = 1
        Rm = C / np.outer(sd, sd)
        Ri, _ = inf.invsym(Rm)
        dS = np.sqrt(np.clip(np.diag(Ri), 0, None))
        dS[dS == 0] = 1
        Pm = Ri / np.outer(dS, dS)
        lo = np.tril_indices_from(Rm, -1)
        kmo = 1 / (1 + (Pm[lo] ** 2).sum() / (Rm[lo] ** 2).sum())
        evals, evecs = np.linalg.eigh(Rm)
        order = np.argsort(-evals, kind="stable")
        evals, evecs = evals[order], evecs[:, order]
        evals[np.abs(evals) < 1e-12] = 0.0
        if self.components:
            jg = min(self.components, len(g_live))
        else:
            jg = max(int((evals > .999999).sum()), k - jiv_live)
        P = np.zeros((len(live), jiv_live + jg))
        P[:jiv_live, :jiv_live] = np.eye(jiv_live)
        P[jiv_live:, jiv_live:] = evecs[:, :jg]
        return P, dict(components=jg, eigenvalues=evals, kmo=float(kmo),
                       r2=float(evals[:jg].sum() / len(evals)))
