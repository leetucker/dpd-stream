"""The instrument matrix's layout: which column each instrument is.

The columns follow xtabond2 exactly, because with more instruments than
the data can identify (common in small samples) which ones its
generalized inverse drops depends on their order:

- IV-style columns first, then GMM-style ones;
- within each kind, the groups in reverse order of specification (each new
  group goes to the head of a list), the constant (system GMM) being the
  last IV-style column;
- within a GMM group, for each lag from a to b, one column per (period,
  variable) -- or per variable when collapsed -- then, for a system's
  ``eq="both"`` groups, the levels equation's differenced instruments.

The layout is computed on the balanced time grid of T periods (as xtabond2
fills the data out to N x T rows). Columns that turn out to be empty in the
sample are dropped after pass 1; they would be dropped by the generalized
inverse anyway, without changing the order of the others.

Everything numba needs is in flat integer arrays (`Layout.arrays()`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._frontend import Iv, Spec, Term

_EQ = {"level": 0, "diff": 1, "both": 2}


@dataclass
class GmmGroup:
    """A GMM-style group as xtabond2 holds it (a `split` group is two)."""

    name: str
    terms: list                 # indices into Layout.terms
    equation: int               # 0 levels, 1 transformed, 2 both
    passthru: bool
    collapse: bool
    orthogonal: bool
    a: int
    b: int
    full_eq: int = 0            # the full set is for the levels equation
    full_diffed: int = 0        # ... and made of differences
    extra: bool = False         # system eq(both): the levels' extra instruments
    lags: list = field(default_factory=list)       # [(lag, lo, hi, col0)], lo/hi 1-based
    col0: int = 0
    ncols: int = 0
    xlag: int = 0
    xlo: int = 0
    xhi: int = 0
    xcol0: int = 0
    xcols: int = 0
    xstop: int = 0
    xsh: list = field(default_factory=list)        # per variable: lags shifted deeper


@dataclass
class Layout:
    T: int
    system: bool
    orthogonal: bool
    terms: list                 # [Term], all the series a panel needs
    y: int
    x: list                     # term per regressor (the constant last, if any)
    xnames: list
    iv_cols: list               # [(term, eq, passthru, mz, name)]
    iv_groups: list             # [(name, [columns])], for difference-in-Hansen
    gmm: list                   # [GmmGroup]
    complete_terms: list        # non-mz IV-style terms checked for completeness
    j_iv: int = 0
    j_gmm: int = 0
    max_gmm_nnz: int = 0
    notes: list = field(default_factory=list)

    @property
    def j(self):
        return self.j_iv + self.j_gmm

    @property
    def K(self):
        return len(self.x)

    def arrays(self):
        """The layout as the flat arrays the kernels take."""
        tb = np.array([t.base for t in self.terms], np.int64)
        tk = np.array([t.k for t in self.terms], np.int64)
        td = np.array([t.diff for t in self.terms], np.bool_)
        iv_term = np.array([c[0] for c in self.iv_cols], np.int64)
        iv_eq = np.array([c[1] for c in self.iv_cols], np.int64)
        iv_pass = np.array([c[2] for c in self.iv_cols], np.bool_)
        iv_mz = np.array([c[3] for c in self.iv_cols], np.bool_)
        g = self.gmm
        g_terms, g_t0, lag_lo, lag_hi, lag_col, g_lp = [], [], [], [], [], []
        for grp in g:
            g_t0.append(len(g_terms))
            g_terms += grp.terms
            g_lp.append(len(lag_lo))
            for _, lo, hi, c0 in grp.lags:
                lag_lo.append(lo)
                lag_hi.append(hi)
                lag_col.append(c0)
        i64 = lambda v: np.array(v, np.int64)  # noqa: E731
        nbmax = max((len(x.terms) for x in g), default=1)
        return dict(
            term_base=tb, term_k=tk, term_diff=td,
            ty=np.int64(self.y), tx=i64(self.x), cmp_terms=i64(self.complete_terms),
            iv_term=iv_term, iv_eq=iv_eq, iv_pass=iv_pass, iv_mz=iv_mz,
            g_nb=i64([len(x.terms) for x in g]), g_t0=i64(g_t0), g_terms=i64(g_terms),
            g_eqrow=i64([x.full_eq for x in g]), g_fd=i64([x.full_diffed for x in g]),
            g_coll=i64([x.collapse for x in g]), g_orth=i64([x.orthogonal for x in g]),
            g_a=i64([x.a for x in g]), g_b=i64([x.b for x in g]), g_lp=i64(g_lp),
            lag_lo=i64(lag_lo), lag_hi=i64(lag_hi), lag_col=i64(lag_col),
            g_extra=i64([x.extra for x in g]), g_xlag=i64([x.xlag for x in g]),
            g_xlo=i64([x.xlo for x in g]), g_xhi=i64([x.xhi for x in g]),
            g_xcol=i64([x.xcol0 for x in g]),
            g_xsh=np.array([(list(x.xsh) + [0] * max(nbmax, 1))[:max(nbmax, 1)]
                            for x in g] or np.zeros((0, 1)), np.int64).reshape(len(g), -1),
        )

    def column_names(self):
        """A name for every column, xtabond2-style (for `instruments()`)."""
        names = [f"{nm}" + {0: " (levels)", 1: " (transformed)", 2: ""}[eq]
                 for (_, eq, _, _, nm) in self.iv_cols]
        names += [""] * self.j_gmm
        T = self.T
        for grp in self.gmm:
            tn = [self.terms[t].name for t in grp.terms]
            nb = len(tn)
            eqn = "levels" if grp.full_eq else "transformed"
            for lag, lo, hi, c0 in grp.lags:
                pre = "D." if grp.full_diffed else ("BOD." if grp.orthogonal else "")
                if grp.collapse:
                    for v in range(nb):
                        names[self.j_iv + c0 + v] = f"{pre}L{lag}.{tn[v]} ({eqn}, collapsed)"
                else:
                    for s in range(lo, hi + 1):
                        for v in range(nb):
                            names[self.j_iv + c0 + (s - lo) * nb + v] = \
                                f"{pre}L{lag}.{tn[v]} ({eqn}, t={s - 1 + lag})"
            if grp.extra:
                if grp.collapse:
                    for v in range(nb):
                        names[self.j_iv + grp.xcol0 + v] = f"D.L{grp.xlag}.{tn[v]} " \
                                                           "(levels, collapsed)"
                else:
                    for s in range(grp.xlo, grp.xhi + 1):
                        for v in range(nb):
                            names[self.j_iv + grp.xcol0 + (s - grp.xlo) * nb + v] = \
                                f"D.L{grp.xlag}.{tn[v]} (levels, t={s - 1 + grp.xlag})"
        del T
        return names


def _insts_per_base(g: GmmGroup, T: int) -> int:
    """Clip the lag limits to what the grid allows and count the
    instruments per variable (xtabond2's rule)."""
    lag_max = T - 1 - (g.full_eq and g.full_diffed)
    fwd_max = T - 2 + g.full_eq
    g.a = max(-fwd_max, g.a)
    g.b = min(lag_max, g.b)
    return lag_max, fwd_max


def build(spec: Spec, T: int, system: bool, orthogonal: bool, constant: bool) -> Layout:
    """The layout of `spec`'s instruments on a grid of `T` periods."""
    terms: list[Term] = []
    index: dict = {}

    def term(t: Term) -> int:
        key = (t.base, t.k, t.diff)
        if key not in index:
            index[key] = len(terms)
            terms.append(t)
        return index[key]

    y = term(spec.y)
    xterms = list(spec.x) + list(spec.time_dummies)
    x = [term(t) for t in xterms]
    xnames = [t.name for t in xterms]
    use_const = system and constant
    if use_const:
        x.append(term(Term("Intercept", -1)))
        xnames.append("Intercept")
    if not x:
        raise ValueError("no regressors")

    notes = []
    # IV-style groups: given order reversed, time dummies (iv(yr*)) given
    # first, the constant last
    iv_specs = []
    tobj = None
    if spec.time_dummies:
        tobj = Iv(tuple(t.name for t in spec.time_dummies))
        iv_specs.append((tobj, list(spec.time_dummies)))
    iv_specs += list(spec.iv)
    groups_rev = []
    for obj, ts in iv_specs:
        eqn = _EQ[obj.eq]
        if not (system or eqn):
            notes.append(f"{obj}: instruments for the levels equation only, ignored in "
                         "difference GMM")
            continue
        if obj.passthru and eqn == 2 and system:
            raise ValueError(f"{obj}: passthru is not valid with eq='both' in system GMM")
        groups_rev.insert(0, (obj, ts))
    iv_cols, iv_groups, complete_terms = [], [], []
    for obj, ts in groups_rev:
        start = len(iv_cols)
        for t in ts:
            ti = term(t)
            iv_cols.append((ti, _EQ[obj.eq], obj.passthru, obj.mz, t.name))
            if not obj.mz:
                complete_terms.append(ti)
        name = f"time effects ({spec.time_effects})" if obj is tobj else str(obj)
        iv_groups.append((name, list(range(start, len(iv_cols)))))
    if use_const:
        iv_cols.append((term(Term("Intercept", -1)), 0, False, False, "Intercept"))
    j_iv = len(iv_cols)

    # GMM-style groups
    gmm_rev = []
    for obj, ts in spec.gmm:
        eqn = _EQ[obj.eq]
        split = obj.split
        if split and eqn != 2:
            notes.append(f"{obj}: split has no effect with eq='{obj.eq}'")
            split = False
        if split and not system:
            notes.append(f"{obj}: split has no effect in difference GMM")
            split = False
        if not (system or eqn):
            notes.append(f"{obj}: instruments for the levels equation only, ignored in "
                         "difference GMM")
            continue
        if system and obj.passthru and eqn == 2:
            raise ValueError(f"{obj}: passthru is not valid with eq='both' in system GMM")
        a, b = obj.lags
        b = 10 ** 9 if b is None else b
        ti = [term(t) for t in ts]
        base = dict(terms=ti, passthru=obj.passthru, collapse=obj.collapse,
                    orthogonal=obj.orthogonal)
        if split:
            pieces = [GmmGroup(name=str(obj) + " [levels]", equation=0, a=a - 1, b=a - 1,
                               **base),
                      GmmGroup(name=str(obj) + " [transformed]", equation=1, a=a, b=b, **base)]
        else:
            pieces = [GmmGroup(name=str(obj), equation=eqn, a=a, b=b, **base)]
        for p in pieces:
            if p.orthogonal and not orthogonal and p.equation:
                notes.append(f"{obj}: backward orthogonal deviations are usually valid only "
                             "with transform='fod'")
        gmm_rev = pieces + gmm_rev

    col = 0
    max_nnz = 0
    for g in gmm_rev:
        g.extra = g.equation == 2 and system
        g.full_eq = int(not g.equation)
        g.full_diffed = int(not (g.equation or g.passthru))
        nb = len(g.terms)
        _insts_per_base(g, T)
        if g.a > g.b:
            raise ValueError(f"{g.name}: no lags available on a grid of {T} periods")
        g.col0 = col
        if g.a + g.full_eq > 0:
            lo, hi = 1 + g.full_diffed, T - g.a
        else:
            lo, hi = 2 - g.full_eq - g.a, T
        for lag in range(g.a, g.b + 1):
            g.lags.append((lag, lo, hi, col))
            col += nb if g.collapse else max(0, hi - lo + 1) * nb
            if lag >= 0:
                hi -= 1
            if lag < 0 or (lag == 0 and not g.full_diffed):
                lo += 1
        nnz = (g.b - g.a + 1) * nb
        if g.extra:
            if g.a > 0:
                g.xlag, g.xstop = g.a - 1, g.b - 1
            elif g.b > 0:
                g.xlag, g.xstop = 0, g.b - 1
            else:
                raise ValueError(f"{g.name}: GMM-style groups of leads only (both lag limits "
                                 "<= 0) are not supported for system GMM's levels equation")
            g.xlo, g.xhi = (1, T - g.xlag)
            g.xcol0 = col
            g.xcols = nb if g.collapse else max(0, g.xhi - g.xlo + 1) * nb
            col += g.xcols
            nnz += nb
        g.ncols = col - g.col0
        max_nnz += nnz
    return Layout(T=T, system=system, orthogonal=orthogonal, terms=terms, y=y, x=x,
                  xnames=xnames, iv_cols=iv_cols, iv_groups=iv_groups, gmm=gmm_rev,
                  complete_terms=complete_terms, j_iv=j_iv, j_gmm=col,
                  max_gmm_nnz=max_nnz, notes=notes)


def h_matrix(h: int, left: bool, right: bool, system: bool, T: int) -> np.ndarray:
    """xtabond2's H: the prior covariance of the (stacked) errors, per
    panel, rows transformed equation then levels (Roodman 2009, eq. 26)."""
    def xform(orth):
        M = np.zeros((T, T))
        if orth:
            for c in range(1, T):
                M[c - 1, c] = 1.0
                M[c:, c] = -1.0 / (T - c)
            n = np.sqrt((M ** 2).sum(0))
            n[n == 0] = 1
            return M / n
        for c in range(1, T):
            M[c, c] = 1.0
            M[c - 1, c] = -1.0
        return M
    Ml = np.eye(T) if h == 1 else xform(left)
    Mr = np.eye(T) if h == 1 else xform(right)
    if h == 3 or not system:
        if system:
            Ml = np.hstack([Ml, np.eye(T)])
            Mr = np.hstack([Mr, np.eye(T)])
        H = Ml.T @ Mr
    else:
        H = np.zeros((2 * T, 2 * T))
        H[:T, :T] = Ml.T @ Mr
        H[T:, T:] = np.eye(T)
    H[np.abs(H) < 1e-12] = 0.0
    return H


def csr(H: np.ndarray):
    """H's nonzeros by row: (indptr, indices, values)."""
    ptr, idx, val = [0], [], []
    for r in range(H.shape[0]):
        nz = np.flatnonzero(H[r])
        idx += nz.tolist()
        val += H[r, nz].tolist()
        ptr.append(len(idx))
    return (np.array(ptr, np.int64), np.array(idx, np.int64), np.array(val, np.float64))


def diff_sargan_groups(lay: Layout):
    """The instrument groups of the difference-in-Sargan/Hansen tests:
    [(name, set of columns tested)], xtabond2's: each IV-style group (not
    the constant), each GMM-style group, and in system GMM "GMM instruments
    for levels" (every GMM-style levels instrument)."""
    out = [(name, set(cols)) for name, cols in lay.iv_groups]
    for g in lay.gmm:
        out.append((g.name, set(range(lay.j_iv + g.col0, lay.j_iv + g.col0 + g.ncols))))
    if lay.system:
        lev = set()
        for g in lay.gmm:
            if g.full_eq:
                lev |= set(range(lay.j_iv + g.col0, lay.j_iv + g.col0 + g.ncols))
            elif g.extra:
                lev |= set(range(lay.j_iv + g.xcol0, lay.j_iv + g.xcol0 + g.xcols))
        out.append(("GMM instruments for levels", lev))
    return out
