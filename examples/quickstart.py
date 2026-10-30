"""Quickstart: difference and system GMM on a simulated dynamic panel.

    python examples/quickstart.py

The panel is y_it = 0.5 y_i,t-1 + x_it + 0.5 w_it + alpha_i + lambda_t + e_it,
with x predetermined (it responds to past shocks) and w strictly exogenous
but correlated with the fixed effect. OLS overstates the lagged dependent
variable's coefficient and within-groups understates it; GMM recovers it.
"""

from pathlib import Path

from dpd_stream import dpd_stream, etable
from dpd_stream.simulate import simulate_dynamic, truth_dynamic

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

df = simulate_dynamic(20_000, 8, rho=(0.5,), seed=1)
df.write_parquet(OUT / "panel.parquet")
print(df.head(3))
print("truth:", truth_dynamic(rho=(0.5,)))

# ---- difference GMM (Arellano-Bond): two-step, Windmeijer-corrected SEs.
# The year effects are absorbed; y's lags 2-4 instrument the differenced
# equation, x's lags 1-3 (x is predetermined), w is its own instrument.
diff = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)",
                  OUT / "panel.parquet", panel=("id", "year"), verbose=False)
diff.summary()

# ---- system GMM (Blundell-Bond): the levels equation is added, instrumented
# by differences. w is correlated with the fixed effect, so it instruments
# the differenced equation only; the year effects become dummies.
sys_ = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4, collapse=True) "
                  "+ gmm(x, 1:3, collapse=True) + iv(w, eq='diff')",
                  OUT / "panel.parquet", panel=("id", "year"), system=True, verbose=False)
sys_.summary()

# ---- one table; the one-step estimates come with every two-step fit
print(etable([diff.onestep, diff, sys_], keep=["l\\(y", "^x$", "^w$"], type="df"))
print("long-run effect of x:", sys_.long_run("x"))
