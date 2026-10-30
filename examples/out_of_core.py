"""Out of core: a Parquet panel larger than memory.

    python examples/out_of_core.py [n_panels]

The data are read once, into blocks of whole panels on disk; each later
pass reads them in turn. Memory holds a block and the moment matrices,
whatever the size of the data. Residuals come back as a lazy Polars frame,
sunk to Parquet without being collected.
"""

import sys
import time
from pathlib import Path

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
path = OUT / "big.parquet"
if not path.exists():
    simulate_dynamic(n, 10, rho=(0.5,), seed=8).write_parquet(path)

t = time.time()
r = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)", path,
               panel=("id", "year"), storage="disk", block_mb=64, verbose=True)
print(f"{r.n_obs:,} observations, {r.n_instruments} instruments, "
      f"{time.time() - t:.1f} s")
r.resid("fd").sink_parquet(OUT / "resid.parquet")
