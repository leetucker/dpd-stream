# Resource limits and profiling

(quickstart)=
## Quickstart for cluster jobs

Two habits cover most of it.

**1. Profile your fits before you run them.** `profile=True` reads the data once and returns
the estimated memory and disk instead of fitting, so you can request the right resources:

```python
from dpd_stream import dpd_stream

p = dpd_stream("y ~ l(y, 1) + x | year | gmm(y, 2:.) + iv(x)", "data/*.parquet",
               panel=("id", "year"), profile=True)
print(p.recommend())      # ncpus=8  mem=14gb  disk=40gb
print(p.recommend().pbs)  # select=1:ncpus=8:mem=14gb
```

**2. If your script imports Polars, run `stream_limits.set_limits()` *before* the import.**

```python
import stream_limits
stream_limits.set_limits()     # caps threads and Polars' memory at the job's allocation

import polars as pl            # only now
from dpd_stream import dpd_stream
```

Polars fixes its thread pool and its memory budget when it is imported, and nothing can change
them afterwards, so a limit set after the import cannot reach it (you get a warning that says
so). On a cluster, importing dpd-stream applies the limits for you; the line above matters when
something imports Polars *before* dpd-stream does, such as your own `import polars` at the top of the
script.

The rest of this page is the short version of
[hdfe-stream's Resource limits and profiling](https://leetucker.github.io/hdfe-stream/resources.html), the full reference.

## What the profile models

dpd-stream's memory is set by the number of instrument columns `jf`, which the formula and the
number of periods fix before any data is read (`jf` is an upper bound: columns empty in the sample
are dropped after pass 1), and by the thread count: `S1` and, with the identification tests on
(`idtests`, the default `"auto"`), `Z'Z` each take `n_threads x jf^2 x 8` bytes. `collapse=True`,
tighter lag limits, `idtests=False` and fewer threads all shrink it, and when the accumulators
would not fit in the memory left, pass 1 uses fewer of them (fewer threads, same answer).

## Thread and memory limits

Polars, numba, BLAS and pyarrow each default to every core of the node. dpd-stream caps all of
them at a scheduler's allocation automatically when it finds one, and gives Polars 2.x a memory
budget past which it spills to disk. This is done by the standalone
[`stream-limits`](https://leetucker.github.io/stream-limits/) package; its documentation covers
[thread limits](https://leetucker.github.io/stream-limits/threads.html),
[memory limits](https://leetucker.github.io/stream-limits/memory.html) and
[schedulers and containers](https://leetucker.github.io/stream-limits/schedulers.html) (PBS, Slurm,
cgroups, and a `qstat` fallback for sites that set no memory variable). `python -m stream_limits`
shows what was found, from inside a job. The `n_threads=` option holds numba and BLAS to that many
threads for one fit.

## Choices that follow the memory left

Choices that trade speed for memory follow what is *left* of the job's memory: the `max_memory_gb`
option sets the limit and `memory_reserve` the share held back (default 0.15), and what cannot
shrink is refused up front with the numbers. Every fit records `diagnostics["peak_rss_gb"]`. The
full explanation, with the profile's accuracy and what each package does with the headroom, is in
[hdfe-stream's documentation](https://leetucker.github.io/hdfe-stream/resources.html).
