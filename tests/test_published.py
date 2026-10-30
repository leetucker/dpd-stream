"""Published and other libraries' results.

- Roodman (2009), "How to do xtabond2", Stata Journal 9(1), pp. 111-118:
  xtabond2's output, to the digits printed, on Arellano and Bond's (1991)
  data. Those are not distributed with dpd-stream; the tests run on
  Stata's abdata when it is there (see the `abdata` fixture), and skip
  otherwise. Stata's abdata is float, so estimates agree to about 1e-6
  relative. The paper's system GMM example predates two changes in xtabond2
  (3.7.2 is what dpd-stream follows): sigma^2 for the Sargan test is now
  divided by the transformed equation's observations (751), not the levels'
  (891), and the model's Wald test now includes the constant. Those two
  numbers are checked with the old conventions undone.
- R plm 2.6.7's pgmm, two-step difference GMM with time effects
  (``effect = "twoways"``) and vcovHC's Windmeijer correction, on the
  simulated panel (tests/data/sim_panel.csv); values from R 4.5.3.
"""

import numpy as np
import pytest

from dpd_stream import dpd_stream

YR = " + ".join(f"yr{y}" for y in range(1976, 1985))
AB = ("n ~ l(n, 1:2) + w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + " + YR)
IVS = "iv(w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + " + YR + ")"


def close(a, b, digits, rel=1e-5):
    """Agreement to the printed digits (rounding), allowing for Stata's
    float data (about 1e-5 relative in a standard error, 3e-7 absolute)."""
    assert abs(a - b) <= 0.6 * 10 ** -digits + rel * abs(b) + 3e-7, (a, b)


def test_ab_onestep_robust(abdata):
    """Roodman (2009) p. 111: Arellano-Bond Table 4 (a1)."""
    r = dpd_stream(AB + " | gmm(l(n, 1)) + " + IVS, abdata, panel=("firm", "year"),
                   steps=1, vcov="robust", verbose=False)
    want = {"l(n, 1)": (.6862261, .1445943), "l(n, 2)": (-.0853582, .0560155),
            "w": (-.6078208, .1782055), "l(w, 1)": (.3926237, .1679931),
            "k": (.3568456, .0590203), "l(k, 1)": (-.0580012, .0731797),
            "ys": (.6085073, .1725313), "l(ys, 2)": (.1057969, .1412021),
            "yr1983": (-.0136171, .0188263)}
    b, se = dict(zip(r.coefnames, r.beta)), dict(zip(r.coefnames, r.se))
    for k, (bb, ss) in want.items():
        close(b[k], bb, 7)
        close(se[k], ss, 7)
    assert set(r.collin_vars) == {"yr1976", "yr1977", "yr1984"}
    assert (r.n_obs, r.n_groups, r.n_instruments) == (611, 140, 41)
    assert (r.obs_per_group["min"], r.obs_per_group["max"]) == (4, 6)
    close(r.model_test["statistic"], 1727.45, 2)
    assert r.model_test["df1"] == 16
    close(r.ar[0]["z"], -3.60, 2)
    close(r.ar[1]["z"], -0.52, 2)
    close(r.sargan["statistic"], 67.59, 2)
    assert r.sargan["df"] == 25
    close(r.hansen["statistic"], 31.38, 2)
    d, = r.diff_hansen
    close(d["excluded"], 12.01, 2)
    assert (d["excluded_df"], d["difference_df"]) == (11, 14)
    close(d["difference"], 19.37, 2)


def test_ab_gmm_wk_small(abdata):
    """Roodman (2009) p. 113: w and k instrumented GMM-style, small."""
    r = dpd_stream(AB + " | gmm(l(n, 1) + l(w, 1) + l(k, 1)) + iv(ys + l(ys, 1:2) + " + YR + ")",
                   abdata, panel=("firm", "year"), steps=1, small=True, verbose=False)
    b, se = dict(zip(r.coefnames, r.beta)), dict(zip(r.coefnames, r.se))
    close(b["l(n, 1)"], .8179867, 7)
    close(b["w"], -.6816685, 7)
    close(b["yr1983"], -.0069458, 7)
    # the paper's standard errors are 2e-5 (relative) below these: its
    # 2009 version's small-sample factor, apparently (ex. 1, without small,
    # agrees to 1e-6)
    np.testing.assert_allclose(se["l(n, 1)"], .0859761, rtol=5e-5)
    assert r.n_instruments == 90
    m = r.model_test
    assert (m["df1"], m["df2"], m["distribution"]) == (16, 140, "F")
    close(m["statistic"], 85.30, 2, rel=1e-4)       # with the SEs' 2e-5
    close(r.ar[0]["z"], -5.39, 2)
    close(r.ar[1]["z"], -0.78, 2)
    close(r.sargan["statistic"], 120.62, 2)
    close(r.hansen["statistic"], 73.72, 2)
    d, = r.diff_hansen
    close(d["excluded"], 56.99, 2)
    close(d["difference"], 16.72, 2)


def test_bb_system_onestep(abdata):
    """Roodman (2009) p. 118: Blundell-Bond system GMM, one-step robust."""
    r = dpd_stream("n ~ l(n, 1) + w + l(w, 1) + k + l(k, 1) + " + YR
                   + " | gmm(l(n, 1) + l(w, 1) + l(k, 1)) + iv(" + YR + ", eq='level')",
                   abdata, panel=("firm", "year"), system=True, steps=1, small=True,
                   verbose=False)
    b, se = dict(zip(r.coefnames, r.beta)), dict(zip(r.coefnames, r.se))
    close(b["l(n, 1)"], .9356053, 7)
    close(se["l(n, 1)"], .026569, 6)
    close(b["k"], .4839299, 7)
    close(se["k"], .0544281, 7)
    close(b["Intercept"], .5522011, 7)
    close(se["Intercept"], .1971607, 7)
    assert set(r.collin_vars) == {"yr1976", "yr1984"}
    assert (r.n_obs, r.n_groups, r.n_instruments) == (891, 140, 113)
    assert r.model_test["df1"] == 12 and r.model_test["df2"] == 139
    # the paper's F excludes the constant (old xtabond2)
    k = len(r.beta) - 1
    F_old = r.beta[:k] @ np.linalg.solve(r.vcov[:k, :k], r.beta[:k]) / 12
    close(F_old, 1154.36, 2)
    close(r.ar[0]["z"], -5.46, 2)
    close(r.ar[1]["z"], -0.25, 2)
    # the paper's sigma^2 divides by the 891 levels observations (old xtabond2)
    close(r.sargan["statistic"] * 891 / 751, 186.90, 2)
    close(r.hansen["statistic"], 110.70, 2)
    lev = [d for d in r.diff_hansen if d["group"] == "GMM instruments for levels"][0]
    close(lev["excluded"], 84.33, 2)
    assert (lev["excluded_df"], lev["difference_df"]) == (79, 21)
    close(lev["difference"], 26.37, 2)


PLM = dict(b=[0.35600971508871, -0.089368971613501, 0.64174760546723, 0.52458408028671],
           se_windmeijer=[0.086425674131009, 0.04526900313098, 0.059991829607553,
                          0.050787542014259],
           se_plain=[0.052403153958092, 0.035554592203792, 0.047118452532028,
                     0.037320073087512],
           sargan=21.68641772, ar1=-3.547395513, ar2=0.3189873646)


@pytest.mark.parametrize("vcov", ["robust", "iid"])
def test_plm_twostep(sim_panel, vcov):
    """pgmm(y ~ lag(y, 1:2) + x + w | lag(y, 2:99), effect = "twoways",
    model = "twosteps"): the regressors outside the GMM part, and the time
    effects, are their own instruments."""
    r = dpd_stream("y ~ l(y, 1:2) + x + w | year | gmm(y, 2:.) + iv(x + w)", sim_panel,
                   panel=("id", "year"), vcov=vcov, verbose=False)
    np.testing.assert_allclose(r.beta, PLM["b"], rtol=1e-9)
    se = PLM["se_windmeijer" if vcov == "robust" else "se_plain"]
    np.testing.assert_allclose(r.se, se, rtol=1e-8)
    np.testing.assert_allclose(r.hansen["statistic"], PLM["sargan"], rtol=1e-8)
    if vcov == "robust":
        np.testing.assert_allclose([r.ar[0]["z"], r.ar[1]["z"]], [PLM["ar1"], PLM["ar2"]],
                                   rtol=1e-8)
