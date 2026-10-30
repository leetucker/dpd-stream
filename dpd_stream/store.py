"""Pass 0: the source -> blocks of whole panels, sorted by time.

The source is read once. Each "base" column (every distinct expression the
model's terms are made of) is evaluated row by row by Polars, the time
variable becomes an index on the global grid (0..T-1, in steps of
`tdelta`), cluster variables become integer codes, and the rows are
written, sorted by panel and time, as blocks of whole panels in a
`hdfe_stream.core.store.BlockStore` (on disk, or in memory). Lags,
differences and instruments are not materialized: the kernels build them
from these columns, panel by panel, in every pass.

Rows with a missing panel or time are dropped; every other row is kept,
whatever is missing in it, because a missing value is information (it
breaks a lag), as in xtabond2, which works on the full N x T grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import polars as pl
from hdfe_stream.core.store import BlockStore, PolarsEngine, block_rows, iter_groups, write_sample

_T = "__dpd_t"
_ROW = "__dpd_row"


@dataclass
class Clustering:
    """One term of a (multi-way) clustering: a combination of the cluster
    variables, with its inclusion-exclusion sign."""

    vars: tuple
    sign: int
    nested: bool = False        # constant within panels
    by_obs: bool = False        # each cluster is one observation
    is_panel: bool = False      # the panel variable itself
    n_codes: int = 0
    col: int = -1               # its code column in the store, if any
    offset: int = 0             # general clusterings: offset of its codes in Acc
    n: int = 0                  # clusters in the sample (after pass 1)


@dataclass
class Store:
    blocks: BlockStore
    T: int
    tmin: int
    tdelta: int
    n_panels: int
    n_rows: int
    nbase: int
    clusterings: list = field(default_factory=list)
    weights: str | None = None
    n_absorb_groups: int = 0      # levels of g in an absorbed g^time term


def _combos(vars_):
    """The nonempty combinations of the cluster variables, with signs
    (+ odd, - even), as xtabond2's inclusion-exclusion."""
    out = []
    for r in range(1, len(vars_) + 1):
        for c in combinations(vars_, r):
            out.append((c, 1 if r % 2 else -1))
    return out


def build(lf: pl.LazyFrame, bases, panel, *, tdelta=None, cluster=(), weights=None,
          fweights=False, absorb_g=None, engine: PolarsEngine, workdir, in_memory=False,
          block_mb=64, log=None) -> Store:
    """Pass 0. `bases` is `Spec.bases`, [(name, expr)]; `panel` is
    (id, time); `cluster` the cluster variables."""
    pid, tvar = panel
    schema = lf.collect_schema()
    for c in (pid, tvar, *cluster, *( [weights] if weights else [])):
        if c not in schema:
            raise ValueError(f"column {c!r} not in the data")
    if not schema[tvar].is_integer():
        if schema[tvar].is_float():
            pass
        elif isinstance(schema[tvar], (pl.Date,)):
            lf = lf.with_columns(pl.col(tvar).dt.year().alias(tvar))
        else:
            raise ValueError(f"time variable {tvar!r} must be numeric (integer periods), "
                             f"got {schema[tvar]}")
    lf = lf.filter(pl.col(pid).is_not_null() & pl.col(tvar).is_not_null())
    info = engine.collect(lf.select(
        pl.len().alias("n"), pl.col(tvar).min().alias("tmin"), pl.col(tvar).max().alias("tmax"),
        pl.struct(pid, tvar).n_unique().alias("nu"), pl.col(pid).n_unique().alias("np"),
        (pl.col(tvar) != pl.col(tvar).round(0)).any().alias("frac")), "panel structure")
    n = int(info["n"][0])
    if n == 0:
        raise ValueError("no observations with a panel and a time")
    if info["frac"][0]:
        raise ValueError(f"time variable {tvar!r} must hold whole numbers")
    if int(info["nu"][0]) != n:
        raise ValueError(f"repeated time values within panel: ({pid}, {tvar}) does not "
                         "identify the observations")
    tmin, tmax = int(info["tmin"][0]), int(info["tmax"][0])
    if tdelta is None:
        tdelta = 1
    tdelta = int(tdelta)
    if tdelta < 1:
        raise ValueError("tdelta must be a positive integer")
    if tdelta > 1:
        bad = engine.collect(lf.select(((pl.col(tvar).cast(pl.Int64) - tmin) % tdelta != 0)
                                       .any()), "time steps")
        if bad.item():
            raise ValueError(f"time values are not on a grid of step {tdelta} from {tmin}")
    T = (tmax - tmin) // tdelta + 1
    n_panels = int(info["np"][0])
    if weights and fweights:
        varies = engine.collect(lf.group_by(pid).agg(pl.col(weights).drop_nulls().n_unique()
                                                     .alias("u")).select(pl.col("u").max()),
                                "frequency weights")
        if (varies.item() or 0) > 1:
            raise ValueError("frequency weights must be constant within each panel "
                             "(as xtabond2 requires)")

    # an absorbed g^time term: g's codes, which must be constant within panels
    n_ag = 0
    if absorb_g is not None:
        if absorb_g:
            for c in absorb_g:
                if c not in schema:
                    raise ValueError(f"column {c!r} (absorbed fixed effect) not in the data")
            nulls = engine.collect(lf.select(pl.any_horizontal(
                [pl.col(c).is_null() for c in absorb_g]).sum()), "absorbed effect")
            if nulls.item():
                raise ValueError(f"{'^'.join(absorb_g)} is missing in {nulls.item():,} rows")
            key = pl.struct(list(absorb_g)) if len(absorb_g) > 1 else pl.col(absorb_g[0])
            chk = engine.collect(lf.select(
                pl.struct([pid, *[c for c in absorb_g if c != pid]]).n_unique().alias("pairs"),
                key.n_unique().alias("ng")), "absorbed effect")
            if int(chk["pairs"][0]) != n_panels:
                raise ValueError(
                    f"absorbed fixed effect {'^'.join(absorb_g)}^{tvar}: {'^'.join(absorb_g)} "
                    "must be constant within each panel (after first differencing, its "
                    "interaction with time is then a cell effect, which is what is absorbed)")
            n_ag = int(chk["ng"][0])
            lf = lf.with_columns((key.rank("dense").cast(pl.Int64) - 1).alias("__dpd_ag"))
        else:
            n_ag = 1
            lf = lf.with_columns(pl.lit(0, pl.Int64).alias("__dpd_ag"))

    # clusterings: every combination of the cluster variables
    clusterings = []
    if cluster:
        cl_cols = []
        for k, (vs, sign) in enumerate(_combos(list(cluster))):
            c = Clustering(vars=vs, sign=sign)
            c.is_panel = vs == (pid,)
            key = pl.struct(list(vs)) if len(vs) > 1 else pl.col(vs[0])
            stats = engine.collect(lf.select(
                key.n_unique().alias("codes"),
                pl.struct([pid, *[v for v in vs if v != pid]]).n_unique().alias("pairs"),
                pl.col(pid).n_unique().alias("panels")), f"clustering by {'+'.join(vs)}")
            c.n_codes = int(stats["codes"][0])
            c.nested = int(stats["pairs"][0]) == int(stats["panels"][0])
            c.by_obs = not c.nested and c.n_codes == n
            if not (c.is_panel or c.by_obs):
                name = f"__dpd_cl{k}"
                if len(vs) == 1 and schema[vs[0]].is_integer():
                    # codes by rank, so they are 0..n-1
                    expr = pl.col(vs[0]).rank("dense").cast(pl.Int64) - 1
                else:
                    expr = pl.struct(list(vs)).rank("dense").cast(pl.Int64) - 1
                lf = lf.with_columns(expr.alias(name))
                c.col = len(cl_cols)
                cl_cols.append(name)
            clusterings.append(c)
    else:
        cl_cols = []

    base_names = [f"__dpd_b{i}" for i in range(len(bases))]
    lf2 = lf.with_columns(
        ((pl.col(tvar).cast(pl.Int64) - tmin) // tdelta).alias(_T),
        *[expr.cast(pl.Float64).alias(nm) for nm, (_, expr) in zip(base_names, bases)])
    if weights:
        lf2 = lf2.with_columns(pl.col(weights).cast(pl.Float64).alias("__dpd_w"))
    cols = [pid, _T, *base_names, *cl_cols, *(["__dpd_w"] if weights else []),
            *(["__dpd_ag"] if absorb_g is not None else [])]
    workdir = workdir
    sample = workdir / "sample.parquet"
    width = len(base_names) + len(cl_cols) + 3
    rows = block_rows(block_mb, width)
    write_sample(engine, lf2, pl.lit(True), cols, sample, rows, _ROW)
    store = BlockStore(workdir / "blocks", in_memory=in_memory)
    code = 0
    for frame in iter_groups(engine, sample, [pid], rows, workdir, n_panels,
                             sort_by=[pid, _T], log=log):
        frame = frame.sort([pid, _T])       # whole panels: sorted by time within each
        ids = frame[pid]
        starts = (ids != ids.shift(1)).fill_null(True).to_numpy()
        pcode = np.cumsum(starts).astype(np.int64) - 1 + code
        code = int(pcode[-1]) + 1
        B = (frame.select(base_names).to_numpy().astype(np.float64)
             if base_names else np.empty((frame.height, 0)))
        arrays = {"X": B, "t": frame[_T].to_numpy().astype(np.int64), "pid": pcode}
        if cl_cols:
            arrays["cl"] = np.ascontiguousarray(
                frame.select(cl_cols).fill_null(-1).to_numpy().astype(np.int64))
        if weights:
            arrays["w"] = frame["__dpd_w"].fill_null(np.nan).to_numpy().astype(np.float64)
        if absorb_g is not None:
            arrays["ag"] = frame["__dpd_ag"].to_numpy().astype(np.int64)
        store.add(arrays)
    sample.unlink(missing_ok=True)
    return Store(blocks=store, T=T, tmin=tmin, tdelta=tdelta, n_panels=code, n_rows=n,
                 nbase=len(bases), clusterings=clusterings, weights=weights,
                 n_absorb_groups=n_ag)
