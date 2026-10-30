"""Standard errors: one-step and two-step, Windmeijer's correction,
clustering (one- and two-way), small-sample statistics.

    python examples/inference.py
"""

from dpd_stream import dpd_stream
from dpd_stream.simulate import simulate_dynamic

df = simulate_dynamic(3_000, 8, rho=(0.5,), n_clusters=60, heteroskedastic=True, seed=4)
P = ("id", "year")
F = "y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)"

two = dpd_stream(F, df, panel=P, verbose=False)
print("two-step, Windmeijer-corrected:", two.se.round(4))
print("two-step, uncorrected:        ", two.with_vcov("iid").se.round(4))
print("one-step, robust:             ", two.onestep.se.round(4))

# clustering defines the two-step weight matrix too (as xtabond2's cluster())
cl = dpd_stream(F, df, panel=P, vcov={"CRV1": "cluster"}, verbose=False)
print("clustered by cluster:", cl.se.round(4), cl.n_clusters)
# two-way clustering: with few clusters in one dimension (8 years here) the
# two-step weight matrix it defines is singular, so cluster a one-step fit,
# whose weight does not depend on it
tw = dpd_stream(F, df, panel=P, steps=1, vcov={"CRV1": "cluster+year"}, verbose=False)
print("one-step, two-way (cluster, year):", tw.se[:3].round(4), "(l(y, 1), x, w)")

sm = dpd_stream(F, df, panel=P, small=True, verbose=False)
print(sm.tidy())
print(sm.wald_test(keep="x|w"))
