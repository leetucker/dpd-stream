# Quickstart

{download}`Download the script: quickstart.py <../../examples/quickstart.py>`

The panel has 20,000 units and 8 periods, with $\rho = 0.5$, $\beta = 1$
and $\gamma = 0.5$ (see [the simulated data](index.md#the-simulated-data)):

```python
from dpd_stream import dpd_stream, etable
from dpd_stream.simulate import simulate_dynamic

df = simulate_dynamic(20_000, 8, rho=(0.5,), seed=1)
df.write_parquet("panel.parquet")
```

## Difference GMM

The formula has three parts: the model, time effects, and instruments.

- **Model:** `l(y, 1)` is y's first lag, taken by the time variable's value.
- **Time effects:** `| year` absorbs a period effect. It is estimated as a
  dummy per year, each its own instrument (as xtabond2's `yr*` in the
  regressors and in `iv()`), so every statistic is the dummies' exactly, but
  the dummies are not reported. For reported dummies, write them among the
  regressors and instruments yourself.
- **Instruments:** `gmm(y, 2:4)` instruments the differenced equation with
  y's lags 2 to 4, one column per lag and period, as xtabond2's
  `gmm(y, lag(2 4))`. x is predetermined, so its lags 1 to 3 are valid
  instruments; w is its own instrument.

```python
diff = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)",
                  "panel.parquet", panel=("id", "year"))
diff.summary()
```

```text
Dynamic panel-data estimation, two-step difference GMM (dpd_stream)   vcov: robust (Windmeijer-corrected)
observations: 120,000   groups: 20,000   obs per group: min 6, avg 6.00, max 6   instruments: 39
absorbed fixed effect: year (6 effects, as dummies)
Wald test of all coefficients: chi2(3) = 53969.59, p = 0.0000
Arellano-Bond test for AR(1) in first differences: z = -84.99, p = 0.0000
Arellano-Bond test for AR(2) in first differences: z = -0.29, p = 0.7696
Sargan test of overid. restrictions: chi2(30) = 22.63, p = 0.8300  (not robust)
Hansen test of overid. restrictions: chi2(30) = 22.69, p = 0.8278  (robust)
Difference-in-Hansen tests of exogeneity of instrument subsets:
  iv(w)
    excluding group: chi2(29) = 22.69, p = 0.7905   difference: chi2(1) = 0.00, p = 0.9918
  ...
Underidentification (Kleibergen-Paap rk LM, robust): chi2(31) = 7428.63, p = 0.0000   weak identification (KP rk Wald F): 372.21
  Sanderson-Windmeijer, l(y, 1): chi2(31) = 7197.96, p = 0.0000
  Sanderson-Windmeijer, x: chi2(31) = 8242.24, p = 0.0000
┌─────────────┬──────────┬────────────┬────────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value    ┆ Pr(>|t|) │
╞═════════════╪══════════╪════════════╪════════════╪══════════╡
│ l(y, 1)     ┆ 0.501335 ┆ 0.003398   ┆ 147.518876 ┆ 0.0      │
│ x           ┆ 0.995548 ┆ 0.005092   ┆ 195.507327 ┆ 0.0      │
│ w           ┆ 0.496752 ┆ 0.003504   ┆ 141.777769 ┆ 0.0      │
└─────────────┴──────────┴────────────┴────────────┴──────────┘
```

The report is xtabond2's, and so are the numbers. How to read the tests:

- **AR(1)** rejects, as it must: first differences are serially correlated
  by construction.
- **AR(2)** does not reject, which supports lags 2 and deeper of y as
  instruments.
- **The Hansen test** does not reject the overidentifying restrictions.
- **The difference-in-Hansen tests** check each instrument group in turn.
- **The identification tests** (Kleibergen-Paap, Sanderson-Windmeijer)
  reject underidentification, and the weak-identification F is large; see
  [identification tests](identification.md).

## System GMM

System GMM adds the levels equation, instrumented by lagged differences.
`collapse=True` keeps the instrument count small (16 here). w is correlated
with the fixed effect, so it instruments the differenced equation only
(`eq='diff'`).

```python
sys_ = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4, collapse=True) "
                  "+ gmm(x, 1:3, collapse=True) + iv(w, eq='diff')",
                  "panel.parquet", panel=("id", "year"), system=True)
```

```text
Dynamic panel-data estimation, two-step system GMM (dpd_stream)   vcov: robust (Windmeijer-corrected)
observations: 140,000   groups: 20,000   obs per group: min 7, avg 7.00, max 7   instruments: 16
...
Hansen test of overid. restrictions: chi2(6) = 2.60, p = 0.8576  (robust)
Difference-in-Hansen tests of exogeneity of instrument subsets:
  ...
  GMM instruments for levels
    excluding group: chi2(4) = 1.44, p = 0.8371   difference: chi2(2) = 1.15, p = 0.5613
Underidentification (Kleibergen-Paap rk LM, robust): chi2(7) = 12370.65, p = 0.0000   weak identification (KP rk Wald F): 4385.58
│ l(y, 1)     ┆ 0.497039  ┆ 0.002883   ┆ ...
│ x           ┆ 0.989955  ┆ 0.004088   ┆ ...
│ w           ┆ 0.496112  ┆ 0.003444   ┆ ...
```

## Reporting

Results have hdfe-stream's reporting methods:

- `tidy()`, `confint()`, `wald_test()`, `summary_json()`;
- `etable`, through pyfixest;
- `onestep`, which every two-step fit carries: the one-step estimates, at
  no extra cost.

```python
etable([diff.onestep, diff, sys_], keep=["l\\(y", "^x$", "^w$"])
sys_.long_run("x")           # beta / (1 - rho), delta-method SE
```

```text
{'variable': 'x', 'estimate': 1.968, 'se': 0.0157, 'z': 125.4, 'p': 0.0,
 'terms': ['x'], 'ldv_terms': ['l(y, 1)']}
```

:::{dropdown} The script
```{literalinclude} ../../examples/quickstart.py
:language: python
```
:::
