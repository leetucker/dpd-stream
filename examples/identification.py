"""Underidentification and weak-instrument tests.

    python examples/identification.py

As rho approaches 1, lagged levels stop predicting differences: difference
GMM is weakly identified, and the Kleibergen-Paap and Sanderson-Windmeijer
tests say so.
"""

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

P = ("id", "year")
for rho in (0.3, 0.8, 0.98):
    df = simulate_dynamic(3_000, 7, rho=(rho,), seed=7)
    r = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4, collapse=True) "
                   "+ gmm(x, 1:3, collapse=True) + iv(w)", df, panel=P, verbose=False)
    it = r.idtests
    sw = ", ".join(f"{s['variable']}: p = {s['p']:.3f}" for s in it["sanderson_windmeijer"])
    print(f"rho = {rho}: KP rk LM p = {it['kp_lm']['p']:.4f}, KP Wald F = "
          f"{it['kp_wald']['F']:.1f}; Sanderson-Windmeijer {sw}")

# With x strictly exogenous, the lagged dependent variable is the only
# endogenous regressor: the effective F, and the Anderson-Rubin confidence
# set for rho, which stays valid however weak the instruments
for rho in (0.3, 0.8, 0.98):
    df = simulate_dynamic(3_000, 7, rho=(rho,), x_type="exogenous", seed=7)
    r = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4, collapse=True) + iv(x + w)",
                   df, panel=P, verbose=False)
    it = r.idtests
    cs = " U ".join(f"[{lo:.3f}, {hi:.3f}]"
                    for lo, hi in it["anderson_rubin"]["confidence_set"]["intervals"])
    lo, hi = r.confint().row(0)[1:3]
    print(f"rho = {rho}: rho^ = {r.beta[0]:.3f}, Wald 95% [{lo:.3f}, {hi:.3f}], "
          f"Anderson-Rubin 95% {cs or 'empty'}; effective F = {it['effective_f']['F']:.1f} "
          f"(critical value {it['effective_f']['critical_value']:.1f})")
