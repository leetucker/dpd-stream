# Identification tests

{download}`Download the script: identification.py <../../examples/identification.py>`

**What the tests ask.** GMM estimates are only as good as the instruments'
power to predict the endogenous regressors. These are the regressors that
are not their own instruments: here, the lagged dependent variable and the
predetermined x. dpd-stream reports the tests of Windmeijer (2025) that
Stata's `underid` reports with `kp sw`. They are:

- robust to the fit's clustering, by panel unless `vcov=` says otherwise;
- computed in the weighted regressions, with weights;
- computed with the principal components as the instruments, with PCA;
- net of the exogenous regressors (those that are their own instruments, the
  constant among them), and of any absorbed effect.

The algebra is hdfe-stream's `core.ivtests`, which feols_stream's 2SLS uses
too.

| Test | Asks | Rejection means |
|---|---|---|
| **Kleibergen–Paap rk LM** | is the instruments' first-stage coefficient matrix short of full rank? | the model is identified |
| **Kleibergen–Paap rk Wald F** | (weak-identification statistic, with ivreg2's small-sample factor) | small values mean weak instruments |
| **Sanderson–Windmeijer**, one per endogenous regressor | is this regressor identified given the others? | that regressor is identified |

**No Stock–Yogo critical values.** They are derived for homoskedastic
errors and few instruments, so they do not apply here.

```python
for rho in (0.3, 0.8, 0.98):
    df = simulate_dynamic(3_000, 7, rho=(rho,), seed=7)
    r = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4, collapse=True) "
                   "+ gmm(x, 1:3, collapse=True) + iv(w)", df, panel=("id", "year"))
    r.idtests
```

```text
rho = 0.3: KP rk LM p = 0.0000, KP Wald F = 311.4; Sanderson-Windmeijer l(y, 1): p = 0.000, x: p = 0.000
rho = 0.8: KP rk LM p = 0.0000, KP Wald F = 74.8; Sanderson-Windmeijer l(y, 1): p = 0.000, x: p = 0.000
rho = 0.98: KP rk LM p = 0.0095, KP Wald F = 2.5; Sanderson-Windmeijer l(y, 1): p = 0.010, x: p = 0.009
```

As ρ approaches 1, lagged levels stop predicting differences, and the
statistics show it.

## One endogenous regressor: effective F and Anderson–Rubin

With one endogenous regressor (here x is strictly exogenous, so only y's lag
is), `r.idtests` also has:

- **`effective_f`:** Montiel Olea and Pflueger's (2013) effective F, robust
  to heteroskedasticity and clustering, with the critical value that bounds
  2SLS's worst-case bias at 10% of OLS's (their simplified test);
- **`anderson_rubin`:** the Anderson–Rubin (1949) test, and its 95%
  confidence set for the coefficient. Both stay valid however weak the
  instruments are.

Every fit also reports each endogenous regressor's first-stage partial R²
and Shea's partial R² (`partial_r2`), and the Anderson–Rubin test that the
endogenous regressors' coefficients are all zero.

```python
df = simulate_dynamic(3_000, 7, rho=(rho,), x_type="exogenous", seed=7)
r = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4, collapse=True) + iv(x + w)",
               df, panel=("id", "year"))
r.idtests["anderson_rubin"]["confidence_set"]
```

```text
rho = 0.3: rho^ = 0.294, Wald 95% [0.268, 0.321], Anderson-Rubin 95% [0.259, 0.330]; effective F = 575.1 (critical value 18.1)
rho = 0.8: rho^ = 0.782, Wald 95% [0.723, 0.842], Anderson-Rubin 95% [0.704, 0.862]; effective F = 56.2 (critical value 18.6)
rho = 0.98: rho^ = 0.914, Wald 95% [0.684, 1.144], Anderson-Rubin 95% [-inf, -0.689] U [0.169, 2.782] U [6.003, inf]; effective F = 3.8 (critical value 19.4)
```

With strong instruments the two intervals agree. At ρ = 0.98 the effective F
is far below its critical value. The Wald interval still looks tidy, but the
Anderson–Rubin set is unbounded: the data cannot pin ρ down.

**No endogeneity test here.** In the transformed equation, the lagged
dependent variable is endogenous by construction. Whether another regressor
is exogenous is what the difference-in-Hansen test of its own IV-style group
checks (put it in `iv()` and read that group's test).

:::{admonition} These are not Stata's `underid` numbers
:class: note
The statistics follow ivreg2: Kleibergen and Paap's rk by the SVD algebra
(closed form, the same as Anderson's LM under homoskedasticity) and
Sanderson and Windmeijer's by 2SLS. After xtabond2, Stata's `underid` computes
something else:

- **Kleibergen–Paap:** `ranktest`'s `nosvd` form, a J statistic at the LIML
  estimates. (`ranktest` has no SVD form that takes xtabond2's H matrix,
  which is why `underid` uses this one.)
- **Sanderson–Windmeijer:** a two-step GMM that keeps the exogenous
  regressors among the moments, with a first step weighted by $Z'HZ$.

The tests are asymptotically equivalent under the null, but not equal in
finite samples. For a simulated panel the Kleibergen–Paap LM is 33.6 here and
33.0 from `underid`; in two other fits the two differ by 7% and 10%. Neither
form is known to be consistently more conservative. Do not compare these
numbers digit for digit with `underid`'s output.
:::

**Cost.** The tests cost a pass, and the Kleibergen–Paap covariance grows
with the square of (excluded instruments × endogenous regressors). By
default (`idtests="auto"`) they are computed when that product is at most
500. `idtests=True` forces them, and `collapse=True` makes them cheap.

:::{dropdown} The script
```{literalinclude} ../../examples/identification.py
:language: python
```
:::
