"""Absorbing high-dimensional period effects in difference GMM.

    python examples/absorbed.py

With region-by-year shocks, year dummies are not enough; a dummy for every
region-year cell is thousands of regressors and instruments. Absorbing
`region^year` gives the same estimates as those dummies would.
"""

import numpy as np
import polars as pl

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

rng = np.random.default_rng(5)
df = simulate_dynamic(50_000, 8, rho=(0.5,), time_effects=False, seed=5)
n_regions = 400
region = rng.integers(0, n_regions, df["id"].max() + 1)
shock = rng.normal(0, 1, (n_regions, 8 + 1))
df = df.with_columns(pl.Series("region", region[df["id"].to_numpy()]))
yr = (df["year"] - df["year"].min()).to_numpy()
df = df.with_columns(pl.col("y") + shock[df["region"].to_numpy(), yr])  # region-year shocks

P = ("id", "year")
inst = "gmm(y, 2:4) + gmm(x, 1:3) + iv(w)"
naive = dpd_stream(f"y ~ l(y, 1) + x + w | year | {inst}", df, panel=P, verbose=False)
cells = dpd_stream(f"y ~ l(y, 1) + x + w | region^year | {inst}", df, panel=P, verbose=False)
print("year effects only:  ", naive.beta.round(4))
print("region-year cells:  ", cells.beta.round(4), cells.absorbed)
