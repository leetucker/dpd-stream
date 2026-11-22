"""The bootstrap's recentered refits against momentfit (R; Chausse), a general GMM
package, given each unit's rows of y, X and Z (benchmarks/external/
momentfit_recenter_check.R, cached in tests/data/momentfit_recenter.json):

- the shift, the original fit's moment sum at its estimate;
- the draw's recentered one- and two-step (or iterated) estimates;
- its Hansen statistic, from momentfit's specTest;
- Windmeijer's corrected variance, with the derivative of the two-step estimate with
  respect to the first-step one taken by finite differences, not by formula.

The test skips without the cached results.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from test_recenter import FML, _resample

from dpd_stream.estimator import DPD
from dpd_stream.simulate import simulate_dynamic

REF = Path(__file__).parent / "data" / "momentfit_recenter.json"

CASES = {
    "diff 2step": dict(system=False, seed=1),
    "sys 2step": dict(system=True, seed=2),
    "sys fod 2step": dict(system=True, orthogonal=True, seed=3),
    "sys iterated": dict(system=True, steps="iterated", seed=4),
    "diff cluster": dict(system=False, cluster="cluster", seed=6),
}


def panel_frame():
    """tests/conftest.py's panel_data."""
    df = simulate_dynamic(150, 7, rho=(0.5,), unbalanced=True, p_gap=0.08, n_clusters=60,
                          weights=True, heteroskedastic=True, seed=11)
    return df.select("id", "year", "y", "x", "w", "cluster")


def resample(df, seed, cluster=None):
    return _resample(df, seed, key=cluster or "id")


def _options(c):
    kw = dict(system=c["system"], steps=c.get("steps", 2))
    if c.get("orthogonal"):
        kw["transform"] = "fod"
    if c.get("cluster"):
        kw["vcov"] = {"CRV1": c["cluster"]}
    return kw


@pytest.fixture(scope="module")
def reference():
    if not REF.exists():
        pytest.skip("momentfit's results are not here (see benchmarks/external/"
                    "momentfit_recenter_check.R)")
    return json.loads(REF.read_text())


@pytest.mark.parametrize("case", list(CASES))
def test_against_momentfit(reference, case):
    c, ref = CASES[case], reference[case]
    kw = _options(c)
    df = panel_frame()
    est = DPD(FML, df, ("id", "year"), verbose=False, **kw)
    f = est.fit()
    np.testing.assert_allclose(f.b, ref["original_b"], rtol=1e-7, atol=1e-9)
    live = f.info["live_mask"]
    np.testing.assert_allclose(f.info["moment_sum"][live], ref["shift"], rtol=1e-6,
                               atol=1e-7)
    draw, n = resample(df, c["seed"], c.get("cluster"))
    fd = DPD(FML, draw, ("id", "year"), verbose=False, estimates_only=True, draw_se=True,
             moment_shift=(f.info["moment_sum"], n), fixed_columns=live,
             fixed_xsh=f.info["xsh"], tdelta=est.store.tdelta, **kw).fit()
    # momentfit's numerical optimizer, and the same estimates in closed form
    np.testing.assert_allclose(fd.info["b1"], ref["b1"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fd.b, ref["b"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fd.info["b1"], ref["b1_closed"], rtol=1e-7, atol=1e-9)
    if c.get("steps", 2) == 2:
        np.testing.assert_allclose(fd.b, ref["b2_closed"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(fd.info["hansen"], ref["hansen"], rtol=1e-6)
    # Windmeijer by finite differences (two-step); uncorrected at convergence (iterated)
    np.testing.assert_allclose(fd.V, np.array(ref["V"]), rtol=1e-5, atol=1e-10)

