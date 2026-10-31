"""Memory guards follow the headroom (limit, less a reserve, less what the
process holds), and `max_memory_gb` sets the limit: per-thread accumulators
drop to fewer copies, and an allocation that cannot shrink is refused up front."""

from __future__ import annotations

import numpy as np
import pytest

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

F = "y ~ l(y, 1) + x + w | year | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)"
P = ("id", "year")


@pytest.fixture(scope="module")
def df():
    d = simulate_dynamic(3_000, 8, seed=1)
    return d.with_columns(region=(d["id"] % 5), grp=((d["id"] * 7 + d["year"]) % 200))


def test_copies_follow_the_headroom(df):
    from dpd_stream.estimator import DPD
    big = DPD(F, df, P, max_memory_gb=500, verbose=False, n_threads=4)
    tiny = DPD(F, df, P, max_memory_gb=0.01, verbose=False, n_threads=4)
    assert big._copies(10**6, 4) == 4
    assert tiny._copies(10**6, 4) == 1                 # nothing left: one copy, never zero


def test_fewer_accumulators_give_the_same_answer(df):
    wide = dpd_stream(F, df, panel=P, verbose=False, max_memory_gb=500, n_threads=4)
    with pytest.warns(UserWarning, match="even one set|free"):
        thin = dpd_stream(F, df, panel=P, verbose=False, max_memory_gb=0.01, n_threads=4)
    assert np.allclose(wide.coef().to_numpy(), thin.coef().to_numpy(), rtol=1e-8, atol=1e-10) \
        if hasattr(wide.coef(), "to_numpy") else True
    assert np.allclose(np.asarray(wide.beta), np.asarray(thin.beta), rtol=1e-8, atol=1e-10)


def test_cluster_sums_that_cannot_shrink_are_refused(df):
    ok = dpd_stream(F, df, panel=P, verbose=False, vcov={"CRV1": "grp"}, max_memory_gb=500)
    assert ok.beta.shape[0] > 0
    with pytest.raises(MemoryError, match="max_memory_gb"):
        dpd_stream(F, df, panel=P, verbose=False, vcov={"CRV1": "grp"}, max_memory_gb=0.01)


def test_absorbed_cells_are_refused_when_they_do_not_fit(df):
    f = "y ~ l(y, 1) + x + w | region^year | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)"
    ok = dpd_stream(f, df, panel=P, verbose=False, max_memory_gb=500)
    assert ok.se[0] > 0
    with pytest.raises(MemoryError, match="max_memory_gb"):
        dpd_stream(f, df, panel=P, verbose=False, max_memory_gb=0.01)


def test_profile_reports_the_reduced_copies(df):
    p = dpd_stream(F, df, panel=P, profile=True, verbose=False, max_memory_gb=0.01, n_threads=4)
    assert any("fit in the memory left" in w or "even one" in w for w in p.warnings)
    assert p.memory_limit_gb == 0.01
