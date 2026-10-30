import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from dpd_stream.simulate import simulate_dynamic

DATA = Path(__file__).parent / "data"


def lag_by_time(df: pl.DataFrame, var: str, k: int, panel="id", time="year") -> np.ndarray:
    """`var` lagged k periods by time value (NaN where that period is
    missing), aligned with `df`'s rows; a negative k is a lead."""
    src = df.select(panel, (pl.col(time) + k).alias(time), pl.col(var).alias("__v"))
    out = df.select(panel, time).join(src, on=[panel, time], how="left", maintain_order="left")
    return out["__v"].fill_null(np.nan).to_numpy().astype(float)


@pytest.fixture(scope="session")
def panel_data():
    """Unbalanced, with gaps, clusters and weights. (More clusters than
    instruments: with fewer, the two-step moment covariance is singular and
    which columns a generalized inverse keeps is up to roundoff.)"""
    return simulate_dynamic(150, 7, rho=(0.5,), unbalanced=True, p_gap=0.08, n_clusters=60,
                            weights=True, heteroskedastic=True, seed=11)


@pytest.fixture(scope="session")
def sim_panel():
    """A simulated unbalanced panel (200 firms, 3 to 9 years; from
    `simulate_dynamic`, written once to tests/data/sim_panel.csv so that the
    other libraries' cached results apply to exactly these values)."""
    return pl.read_csv(DATA / "sim_panel.csv")


@pytest.fixture(scope="session")
def abdata():
    """Arellano and Bond's (1991) UK firms, Stata's abdata (n, w, k, ys are
    the logs of employment, the wage, capital and industry output), with
    year dummies yr1976..yr1984. The data are not distributed with
    dpd-stream: the Stata bundle exports them (copy its data/abdata.csv to
    tests/data/abdata_stata.csv), or point DPD_STREAM_ABDATA at a CSV of
    them. Without them, the tests that use them skip."""
    path = Path(os.environ.get("DPD_STREAM_ABDATA") or DATA / "abdata_stata.csv")
    if not path.exists():
        pytest.skip("Arellano and Bond's data are not here (see the abdata fixture)")
    d = pl.read_csv(path)
    if "firm" not in d.columns:
        d = d.rename({"id": "firm"})
    d = d.with_columns(pl.col("firm").cast(pl.Int64), pl.col("year").cast(pl.Int64))
    years = sorted(d["year"].unique())
    return d.with_columns([(pl.col("year") == y).cast(pl.Float64).alias(f"yr{y}")
                           for y in years])
