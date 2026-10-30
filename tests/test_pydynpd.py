"""pydynpd (xtabond2's Python port), on the simulated panel
(tests/data/sim_panel.csv).

Values from pydynpd 0.2.2 (benchmarks/external/.pydynpd, numpy < 2.3):
``regression.abond(cmd, df, ["id", "year"])`` with the commands below
(two-step, Windmeijer-corrected, the default).
"""

import numpy as np
import pytest

from dpd_stream import dpd_stream

PYDYNPD = {
    "y L(1:2).y x w | gmm(y, 2:4) gmm(x, 1:3) iv(w) | nolevel": dict(
        system=False, collapse=False,
        b=[0.4628251956520219, -0.1092119490209836, 0.7930231360738565, 0.5198630217324547],
        se=[0.06163489783450746, 0.04793840759897628, 0.12037373719934459,
            0.0436432005454096],
        hansen=26.26720746162459, ar=[-4.21695776412733, 0.8647690434695043]),
    "y L(1:2).y x w | gmm(y, 2:4) gmm(x, 1:3) iv(w)": dict(
        system=True, collapse=False,
        b=[0.5910305480932195, 0.014098326188186232, 1.003737286402274, 0.6094241467939825,
           0.057942044368995804],
        se=[0.040507320440093184, 0.043956539698887194, 0.0772764377378461,
            0.04845767558844088, 0.062451274264447076],
        hansen=39.80370795141832, ar=[-4.742867750152612, 0.11007490554043806]),
    "y L(1:2).y x w | gmm(y, 2:4) gmm(x, 1:3) iv(w) | collapse": dict(
        system=True, collapse=True,
        b=[0.5181724542100766, -0.05847555680158073, 0.8900885880403577, 0.6730845404279971,
           -0.08587127227566976],
        se=[0.057081153066102716, 0.06036990354430777, 0.11143652538641094,
            0.05340030134308378, 0.11372575813335851],
        hansen=5.6583094594664285, ar=[-4.783399553252455, 0.5275698119080693]),
}


@pytest.mark.parametrize("cmd", list(PYDYNPD))
def test_pydynpd(sim_panel, cmd):
    want = PYDYNPD[cmd]
    r = dpd_stream("y ~ l(y, 1:2) + x + w | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)", sim_panel,
                   panel=("id", "year"), system=want["system"], collapse=want["collapse"],
                   verbose=False)
    np.testing.assert_allclose(r.beta, want["b"], rtol=1e-9)
    np.testing.assert_allclose(r.se, want["se"], rtol=1e-8)
    np.testing.assert_allclose(r.hansen["statistic"], want["hansen"], rtol=1e-9)
    np.testing.assert_allclose([a["z"] for a in r.ar], want["ar"], rtol=1e-8)
