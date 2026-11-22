"""Bootstrap inference: resample whole panels and refit. Studentized intervals,
the bootstrap Hansen test, a long-run effect from the draws, clusters.

    python examples/bootstrap.py
"""

import numpy as np

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

# 250 panels of 8 periods, and every available lag as an instrument: many
# instruments for the panels, where the analytic errors are least reliable
df = simulate_dynamic(250, 8, rho=(0.5,), n_clusters=50, heteroskedastic=True,
                      time_effects=False, seed=7)
P = ("id", "year")
F = "y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)"

fit = dpd_stream(F, df, panel=P, verbose=False)
print("instruments:", fit.n_instruments, " panels:", fit.n_groups)
print("Windmeijer-corrected:", fit.se.round(4))

boot = fit.refit_bootstrap(reps=499, seed=1, n_workers=4)
print("bootstrap:           ", boot.standard_errors().round(4))
print(boot.tidy(method="studentized"))

# the Hansen test with a bootstrap p-value, from the recentered draws
print(boot.hansen())

# a function of the coefficients: the long-run effect of x, b_x / (1 - b_ly)
d = boot.draws()
long_run = (d["x"] / (1 - d["l(y, 1)"])).to_numpy()
print("long-run effect of x:", round(fit.long_run("x")["estimate"], 4),
      "95% percentile interval:", np.quantile(long_run, [0.025, 0.975]).round(4))

# iterated GMM has no Windmeijer correction: the bootstrap fills the gap
it = dpd_stream(F, df, panel=P, steps="iterated", verbose=False)
it_boot = it.refit_bootstrap(reps=499, seed=1, n_workers=4)
print("iterated, uncorrected:", it.se.round(4))
print("iterated, bootstrap:  ", it_boot.standard_errors().round(4))

# clustered fits resample whole clusters, each copy a cluster of its own
cl = dpd_stream(F, df, panel=P, vcov={"CRV1": "cluster"}, verbose=False)
cl_boot = cl.refit_bootstrap(reps=499, seed=1, n_workers=4)
print(cl_boot.diagnostics["resampled"], cl_boot.diagnostics["clusters"], "clusters")
print("clustered, bootstrap:", cl_boot.standard_errors().round(4))
