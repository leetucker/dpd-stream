"""Choosing instruments: lag limits, collapsing, forward orthogonal
deviations, principal components, and the tests that guide the choice.

    python examples/instruments.py

Too many instruments overfit the endogenous variables and weaken the
Hansen test (Roodman 2009b); this compares instrument sets on one panel.
"""

import polars as pl

from dpd_stream import Gmm, Iv, dpd_stream
from dpd_stream.simulate import simulate_dynamic

df = simulate_dynamic(5_000, 10, rho=(0.6,), seed=2)
P = ("id", "year")
base = "y ~ l(y, 1) + x + w | year"

specs = {
    "all lags": "gmm(y, 2:.) + gmm(x, 1:.) + iv(w)",
    "lags 2-3": "gmm(y, 2:3) + gmm(x, 1:2) + iv(w)",
    "collapsed": "gmm(y, 2:., collapse=True) + gmm(x, 1:., collapse=True) + iv(w)",
}
rows = []
for name, inst in specs.items():
    r = dpd_stream(f"{base} | {inst}", df, panel=P, verbose=False)
    rows.append(dict(spec=name, rho=r.beta[0], se=r.se[0], instruments=r.n_instruments,
                     hansen_p=r.hansen["p"], ar2_p=r.ar_test(2)["p"],
                     mmsc_bic=r.mmsc["bic"]))
print(pl.DataFrame(rows))

# ---- forward orthogonal deviations keep more observations with gaps
gappy = simulate_dynamic(5_000, 10, rho=(0.6,), p_gap=0.1, seed=3)
for tr in ("fd", "fod"):
    r = dpd_stream(f"{base} | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)", gappy, panel=P,
                   transform=tr, verbose=False)
    print(f"{tr}: {r.n_obs:,} observations, rho = {r.beta[0]:.4f} ({r.se[0]:.4f})")

# ---- principal components of the GMM-style instruments (xtabond2's pca)
r = dpd_stream(f"{base} | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)", df, panel=P, pca=True,
               verbose=False)
print(f"pca: {r.pca['components']} principal components (KMO {r.pca['kmo']:.3f}); "
      f"rho = {r.beta[0]:.4f} ({r.se[0]:.4f})")

# ---- the same specification as objects, e.g. built in a loop
for b in (3, 4, 5):
    r = dpd_stream(base, df, panel=P, gmm=[Gmm("y", lags=(2, b)), Gmm("x", lags=(1, b - 1))],
                   iv=[Iv("w")], verbose=False)
    print(f"lags 2-{b}: rho = {r.beta[0]:.4f}, Hansen p = {r.hansen['p']:.3f}, "
          f"difference-in-Hansen p (x's instruments) = "
          f"{[d for d in r.diff_hansen if 'x' in d['group']][0]['difference_p']:.3f}")
