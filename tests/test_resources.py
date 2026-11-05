"""Memory tests: the streamed fits stay inside a memory limit on data that
would not fit in it.

Run with ``pytest -m resource`` (the default run leaves them out: they take
about two minutes). Each fit runs in a child process under stream-limits'
``resource_runner``: a watchdog on every platform, or a kernel limit when
``STREAM_LIMITS_TEST_CGROUP_PARENT`` names a cgroup set up for it.

The panel is 3 million firms over 8 years, 24 million rows, 0.96 GB in
memory, written as six Parquet files and read as a glob. With
``max_memory_gb=1.0`` difference and system GMM hold about 1.1 GB of private
memory (not counting memory-mapped scratch files, which the kernel can
reclaim), most of it buffers whose sizes are capped: the same on 9.6 million
rows. The limit, 1.6 GB, leaves 0.5 GB for platform differences; a fit that
held the panel in memory would need at least 2 GB. Measured on Linux (WSL2,
2 threads): difference GMM 1.09 GB, system GMM 1.07 GB.

The limit is on private memory, which stream-limits measures on Linux and
Windows. On macOS it measures resident memory, which counts the mapped
scratch files, so there the limit is a looser safety net.
On Windows, private memory is committed memory: a buffer or an allocator
segment counts in full once allocated, touched or not (an hdfe-stream logit
fit had 1.61 GB committed while 1.14 GB was resident). So Windows also gets
the looser limit, until its own CI figures set a tighter one.
"""

from __future__ import annotations

import json
import sys

import polars as pl
import pytest

from dpd_stream.simulate import simulate_dynamic

pytestmark = pytest.mark.resource

PARTS, PANELS_PER_PART, T = 6, 500_000, 8       # 24 million rows
BUDGET_GB = 1.0                                 # the fit's max_memory_gb
LIMIT_GB = {"darwin": 2.6, "win32": 2.6}.get(sys.platform, 1.6)
FML = "y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)"


@pytest.fixture(scope="module")
def big_panel(tmp_path_factory):
    """Simulated in parts (the whole panel at once needs 7 GB to simulate),
    each part's firms numbered after the last part's."""
    folder = tmp_path_factory.mktemp("big")
    size = 0
    for i in range(PARTS):
        d = simulate_dynamic(PANELS_PER_PART, T, seed=100 + i)
        d = d.with_columns(pl.col("id") + i * PANELS_PER_PART)
        d.write_parquet(folder / f"part{i}.parquet")
        size += d.estimated_size()
    assert size > 0.9e9
    return folder / "*.parquet"


def _fit_in_child(resource_runner, system: bool, path, workdir) -> dict:
    code = f"""
import json
from dpd_stream import dpd_stream
r = dpd_stream({FML!r}, {str(path)!r}, panel=("id", "year"), system={system},
               workdir={str(workdir)!r}, verbose=False, max_memory_gb={BUDGET_GB})
print(json.dumps(dict(n_obs=int(r.n_obs), coef={{k: float(v) for k, v in r.coef().items()}})))
"""
    result = resource_runner([sys.executable, "-c", code], memory_gb=LIMIT_GB,
                             threads=2, timeout=900)
    report = {k: result.as_dict()[k] for k in ("status", "mode", "memory_measure",
                                               "peak_private_bytes", "peak_rss_bytes",
                                               "elapsed_s")}
    print(json.dumps(report))
    assert result.status == "ok", (report, result.stderr[-3000:])
    if result.kernel:
        assert result.kernel["events"].get("oom_kill", 0) == 0
    return json.loads(result.stdout.strip().splitlines()[-1])


# The checks against the simulated coefficients (simulate_dynamic's defaults:
# rho 0.5, beta 1) are sanity checks only: agreement with xtabond2 is the rest
# of the suite's job.

def test_difference_gmm_streams_inside_the_limit(resource_runner, big_panel, tmp_path):
    out = _fit_in_child(resource_runner, False, big_panel, tmp_path / "work")
    assert out["n_obs"] == PARTS * PANELS_PER_PART * (T - 2)
    assert abs(out["coef"]["l(y, 1)"] - 0.5) < 0.02
    assert abs(out["coef"]["x"] - 1.0) < 0.02


def test_system_gmm_streams_inside_the_limit(resource_runner, big_panel, tmp_path):
    out = _fit_in_child(resource_runner, True, big_panel, tmp_path / "work")
    assert abs(out["coef"]["l(y, 1)"] - 0.5) < 0.02
    assert abs(out["coef"]["x"] - 1.0) < 0.02
