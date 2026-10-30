"""Ahn and Schmidt's nonlinear moment conditions, for persistent series.

    python examples/nonlinear.py

When y is close to a random walk, its lagged levels are weak instruments
for its differences, and difference GMM is imprecise. Ahn and Schmidt's
moments, E[u_iT (u_it - u_i,t-1)] = 0 under no serial correlation, add
information without system GMM's stationarity assumption.
"""

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

# non-stationary initial conditions: system GMM's levels moments fail here
df = simulate_dynamic(5_000, 7, rho=(0.9,), stationary=False, seed=6)
P = ("id", "year")
F = "y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)"
for name, kw in [("difference GMM", {}),
                 ("difference GMM + Ahn-Schmidt", dict(nl="noserial")),
                 ("system GMM", dict(system=True))]:
    r = dpd_stream(F if "system" not in name else F.replace("iv(w)", "iv(w, eq='diff')"),
                   df, panel=P, verbose=False, **kw)
    print(f"{name:30s} rho = {r.beta[0]:.4f} ({r.se[0]:.4f})   Hansen p = {r.hansen['p']:.3f}")
