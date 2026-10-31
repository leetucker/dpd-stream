"""`profile=True`: the estimate is returned instead of a fit, the instrument
count is the layout's, and the upper bound holds against a real fit."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import stream_limits
from hdfe_stream import Profile

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

# These tests compare the profile with a measured peak, so they need a peak-memory reading (Linux,
# macOS and, from stream-limits 0.1.1, Windows). The model was calibrated on Linux and checked on
# macOS; on Windows a failure here is information about the model, not a broken test.
needs_peak_rss = pytest.mark.skipif(
    stream_limits.peak_rss_gb() is None, reason="peak memory cannot be read on this platform")

F = "y ~ l(y, 1) + x + w | year | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)"
P = ("id", "year")


@pytest.fixture(scope="module")
def df():
    return simulate_dynamic(3_000, 8, seed=1)


def test_returns_a_profile_without_fitting(df, tmp_path):
    p = dpd_stream(F, df, panel=P, profile=True, workdir=tmp_path, verbose=False)
    assert isinstance(p, Profile)
    assert list(tmp_path.iterdir()) == []
    assert p.scan.n_rows == len(df) and p.notes["T"] == 8
    assert p.memory_gb[0] <= p.memory_gb[1]


def test_instrument_count_is_the_layouts_and_bounds_the_fits(df):
    p = dpd_stream(F, df, panel=P, profile=True, verbose=False)
    r = dpd_stream(F, df, panel=P, verbose=False)
    assert p.notes["jf"] >= r.diagnostics["j_columns"] >= p.notes["jf"] - 5


def test_the_variants_profile(df):
    for kw in ({}, {"system": True}, {"collapse": True}, {"steps": 1}, {"steps": "iterated"},
               {"vcov": {"CRV1": "id"}}, {"transform": "fod"}, {"pca": True}, {"idtests": False}):
        p = dpd_stream(F, df, panel=P, profile=True, verbose=False, **kw)
        assert p.memory_gb[1] > 0, kw


def test_threads_and_idtests_move_the_big_terms(df):
    a = dpd_stream(F, df, panel=P, profile=True, verbose=False, n_threads=1)
    b = dpd_stream(F, df, panel=P, profile=True, verbose=False, n_threads=4)
    c = dpd_stream(F, df, panel=P, profile=True, verbose=False, n_threads=4, idtests=False)

    def s1(p):
        return sum(x.upper for x in p.components if x.name.startswith(("S1", "Z'Z per thread")))
    assert s1(b) == pytest.approx(4 * s1(a))
    assert s1(c) < s1(b)
    assert a.threads == 1 and b.threads == 4


def test_absorbed_cells_and_clusters(df):
    d = df.with_columns(region=(df["id"] % 5))
    p = dpd_stream("y ~ l(y, 1) + x + w | region^year | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)",
                   d, panel=P, profile=True, verbose=False)
    assert any(c.name.startswith("absorbed") for c in p.components)
    q = dpd_stream(F, d, panel=P, profile=True, verbose=False, vcov={"CRV1": "region"})
    assert any(c.name.startswith("cluster sums") for c in q.components)


MEASURE = r'''
import json, sys
from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic
fml, kw = sys.argv[1], json.loads(sys.argv[2])
df = simulate_dynamic(20000, 10, seed=2)
p = dpd_stream(fml, df, panel=("id", "year"), profile=True, verbose=False, **kw)
r = dpd_stream(fml, df, panel=("id", "year"), verbose=False, **kw)
print(json.dumps({"mem": p.memory_gb, "rss": r.diagnostics["peak_rss_gb"]}))
'''


@needs_peak_rss
@pytest.mark.parametrize("kw", [{}, {"system": True}, {"collapse": True},
                                {"vcov": {"CRV1": "id"}}])
def test_the_upper_bound_holds(kw):
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", MEASURE, F, json.dumps(kw)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-500:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["rss"] <= r["mem"][1]
    assert r["rss"] >= 0.4 * r["mem"][0]
