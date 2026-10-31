"""`profile=True`: the memory and disk a fit needs, from one scan.

dpd-stream's memory is set by the number of instrument columns jf, which the
formula and the number of periods T determine (`layout.build` needs no data),
by the thread count (every accumulator has one copy per thread), and by the
clusters. So the profile is the most exact of the three packages': the scan
finds N, the panels, T and the cluster counts; the layout gives jf; the
arrays are those `_pass1`, `_resid` and `_windmeijer` allocate.

jf is an upper bound on the columns used: those empty in the sample are
dropped after pass 1.
"""

from __future__ import annotations

import polars as pl
from hdfe_stream.core.profile import (
    GB, Component, Profile, as_components, baseline_gb, scan_source,
)

from . import layout as lay_mod


def profile(est) -> Profile:
    """The `Profile` of `est`, a `DPD` that has not been fitted."""
    import numba
    lf, _ = est._prepare()
    spec = est.spec
    pid, tvar = est.panel
    schema = lf.collect_schema()
    tcol = pl.col(tvar)
    if isinstance(schema[tvar], pl.Date):
        tcol = tcol.dt.year()
    lf = lf.with_columns(tcol.alias(tvar))
    absorb = list(est.absorb_g or ())
    robust, cvars, _ = est.robust, est.cvars, est.vkey
    distinct = {pid: [pid]}
    for c in cvars:
        distinct[c] = [c]
    if absorb:
        distinct["__absorb"] = absorb
    stats = scan_source(lf, present=[pl.col(pid), pl.col(tvar)], distinct=distinct,
                        minmax=[tvar])
    N = max(stats.n_rows, 1)
    tmin, tmax = stats.minmax[tvar]
    tdelta = int(est.tdelta or 1)
    T = (int(tmax) - int(tmin)) // tdelta + 1
    n_panels = max(stats.distinct.get(pid, 1), 1)
    lay = lay_mod.build(spec, T, est.system, est.orth, est.constant)
    jf, K = lay.j, lay.K
    nch = est.n_threads or numba.get_num_threads()
    pl_threads = pl.thread_pool_size()
    nbase = len(spec.bases)
    ncl = sum(1 for c in cvars)
    width = nbase + ncl + 3                                   # store.py's block width
    block_rows = max(int(est.block_mb * 1e6 / (8 * width)), 1)
    comps: list[Component] = []
    warns: list[str] = []
    add = comps.append
    want_zz = est.idtests is not False
    sq = jf * jf * 8
    # pass 1's per-thread accumulators: as many as fit in half the headroom (est._copies),
    # exactly the rule the fit uses; fewer copies is fewer threads on that pass
    nch1 = est._copies(sq * (2 if want_zz else 1), nch)
    if nch1 < nch:
        warns.append(f"pass 1: only {nch1} of {nch} per-thread moment accumulators fit in the "
                     "memory left; that pass runs on fewer threads (slower, same answer)")

    # numba kernels, Polars' engines and scipy, loaded by the first fit in a process
    # (measured: a 2,000-row fit adds 0.4 GB to a process that has only imported
    # the package; fits vary by 0.2 GB with which kernels compile). A process
    # that has already fit holds them in the baseline, so this over-counts there.
    add(Component("kernels and library state (first fit)", 0.6e9, 1.0e9,
                  formula="measured floor of a small fit"))
    add(Component("pass 0: aggregates and copies", N * width * 8 * 1.5, N * width * 8 * 4,
                  phases=("pass0",), formula=f"1.5-4 x N x {width} x 8 (panel checks, sample copy)"))

    # ---- pass 0: scan, narrow sample file, regroup whole panels into blocks
    src = nbase + 2 + ncl
    add(Component("pass 0: Polars sink buffers", 0.25 * pl_threads * 500_000 * src * 8,
                  pl_threads * 500_000 * src * 8, phases=("pass0",),
                  formula=f"{pl_threads} threads x ~one row group x {src} source columns x 8"))
    add(Component("pass 0: regrouping blocks", 2 * est.block_mb * 1e6, 4 * est.block_mb * 1e6,
                  phases=("pass0",), formula="~2-4 x block_mb"))
    store_bytes = N * width * 8
    add(Component("row blocks (store)", store_bytes, store_bytes,
                  where="ram" if est.storage == "memory" else "disk",
                  phases=("all",) if est.storage == "memory" else ("pass1",),
                  formula=f"N x {width} columns x 8 (storage={est.storage!r})"))
    add(Component("narrow sample file", 0.7 * N * src * 8, N * src * 8, where="disk",
                  formula=f"N x {src} source columns x 8, compressed"))
    # a block being processed: the kernels' rows, weights, instruments built per panel
    add(Component("one block in the kernels", est.block_mb * 1e6 * 3, est.block_mb * 1e6 * 5,
                  phases=("pass1", "pass2", "pass3"), formula="~3-5 x block_mb"))

    # ---- pass 1: per-thread moment accumulators (the dominant terms)
    add(Component("S1: Z'HZ per thread", nch * sq, nch * sq, phases=("pass1",),
                  formula=f"nthreads={nch} x jf={jf:,}^2 x 8"))
    if want_zz:
        add(Component("Z'Z per thread (identification tests, PCA)", nch1 * sq, nch1 * sq,
                      phases=("pass1",), formula=f"{nch1} accumulators x jf^2 x 8 "
                                                 f"(idtests={est.idtests!r})"))
    add(Component("Z'X and Z'y per thread", nch1 * jf * (K + 1) * 8, nch1 * jf * (K + 1) * 8,
                  phases=("pass1",), formula=f"{nch1} x jf x (K={K}+1) x 8"))
    # reduced matrices and the algebra on them
    add(Component("weight matrices and their inverses", 6 * sq, 12 * sq,
                  phases=("pass1", "pass2", "pass3"),
                  formula=f"6-12 x jf^2 x 8 (Z'HZ, A1, A2, S, generalized inverses)"))
    if want_zz:
        add(Component("Z'Z (reduced)", sq, 3 * sq, phases=("pass1", "pass2", "pass3"),
                      formula="jf^2 x 8, and its eigendecomposition for PCA"))

    # ---- pass 2: scores per block, clusters
    avg_T = max(N / n_panels, 1.0)
    add(Component("pass 2: per-panel moments of a block",
                  block_rows / avg_T * jf * 8, block_rows * jf * 8, phases=("pass2",),
                  formula=f"panels in a block x jf x 8 (block_rows={block_rows:,})"))
    for c in cvars:
        Gc = stats.distinct.get(c, n_panels)
        per = Gc * jf * 8
        copies = est._copies(per, nch)
        add(Component(f"cluster sums: {c}", per, per * copies, phases=("pass2",),
                      formula=f"G={Gc:,} x jf x 8 (x{copies} per-thread copies at most)"))
        if not est.budget.fits(per / GB):
            warns.append(f"clustering by {c}: {per / GB:.1f} GB of cluster sums and "
                         f"{est.budget.headroom_gb():.1f} GB is free; the fit will raise "
                         "MemoryError (raise max_memory_gb)")

    # ---- pass 3: Windmeijer's derivative, two-step robust
    if est.steps == 2 and robust:
        Gw = max([stats.distinct.get(c, n_panels) for c in cvars] or [n_panels])
        add(Component("Windmeijer: D per thread", nch * jf * K * 8, nch * jf * K * 8,
                      phases=("pass3",), formula="nthreads x jf x K x 8"))
        add(Component("Windmeijer: cluster scores", nch * Gw * K * 8 * 0.1, nch * Gw * K * 8,
                      phases=("pass3",), formula="up to nthreads x G x K x 8"))

    # ---- absorbed g^year cells
    if absorb:
        ng = max(stats.distinct.get("__absorb", 1), 1)
        ncell = ng * T
        nchc = est._copies(ncell * jf * 16, nch)
        cells = nchc * (2 * ncell * jf + ncell * (K + 1) + 2 * ng * T * T) * 8
        add(Component("absorbed g^year cell sums", cells, cells, phases=("pass1", "pass2", "pass3"),
                      formula=f"{nchc} copies x (2 x ncell x jf + ...) x 8; ncell={ncell:,}"))
        if not est.budget.fits(ncell * jf * 16 / GB):
            warns.append(f"{ncell:,} absorbed cells x {jf:,} instruments: the cell sums need "
                         f"{ncell * jf * 16 / GB:.1f} GB and {est.budget.headroom_gb():.1f} GB "
                         "is free; the fit will raise MemoryError (raise max_memory_gb)")
    if est.nl:
        Tn = T * (2 if est.nlkind == 2 else 1)
        add(Component("nonlinear moments", nch * Tn * K * K * 8, nch * Tn * K * K * 8 * 2,
                      phases=("pass1", "pass2"), formula="nthreads x Tn x K^2 x 8"))

    if not est.budget.fits(sq * (2 if want_zz else 1) / GB):
        warns.append(f"{jf:,} instrument columns: even one set of moment matrices needs "
                     f"{sq * (2 if want_zz else 1) / GB:.1f} GB and "
                     f"{est.budget.headroom_gb():.1f} GB is free; consider collapse=True, "
                     "tighter lag limits, idtests=False or a larger max_memory_gb")
    return Profile(
        model=f"dpd_stream {est.fml}", components=as_components(comps), scan=stats,
        threads=int(nch), polars_threads=int(pl_threads), baseline_gb=baseline_gb(),
        memory_limit_gb=est.budget.limit_gb, memory_reserve=est.budget.reserve,
        warnings=warns,
        notes={"jf": jf, "K": K, "T": T, "panels": n_panels, "rows": N,
               "instruments_are_an_upper_bound": "columns empty in the sample are dropped"})
