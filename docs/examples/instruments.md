# Choosing instruments

{download}`Download the script: instruments.py <../../examples/instruments.py>`

**The risk of many instruments.** Every lag of every instrumenting variable,
in every period, is a column. With 10 periods, `gmm(y, 2:.)` alone is 36
columns. Too many instruments overfit the endogenous variables, and the
Hansen test loses its power to reject: its p-values drift toward 1 (Roodman
2009b).

**What to compare.** The usual remedies are to:

- limit the lags;
- collapse the instruments (one per lag);
- replace them by their principal components.

Then look at what changes: the instrument count, the estimates, the Hansen
and AR(2) tests, and the model and moment selection criteria (MMSC; lower
is better).

```python
base = "y ~ l(y, 1) + x + w | year"
specs = {
    "all lags": "gmm(y, 2:.) + gmm(x, 1:.) + iv(w)",
    "lags 2-3": "gmm(y, 2:3) + gmm(x, 1:2) + iv(w)",
    "collapsed": "gmm(y, 2:., collapse=True) + gmm(x, 1:., collapse=True) + iv(w)",
}
for name, inst in specs.items():
    r = dpd_stream(f"{base} | {inst}", df, panel=("id", "year"))
    ...
```

```text
┌───────────┬──────────┬──────────┬─────────────┬──────────┬──────────┬─────────────┐
│ spec      ┆ rho      ┆ se       ┆ instruments ┆ hansen_p ┆ ar2_p    ┆ mmsc_bic    │
╞═══════════╪══════════╪══════════╪═════════════╪══════════╪══════════╪═════════════╡
│ all lags  ┆ 0.595532 ┆ 0.004602 ┆ 89          ┆ 0.486069 ┆ 0.540762 ┆ -748.768287 │
│ lags 2-3  ┆ 0.593107 ┆ 0.005802 ┆ 40          ┆ 0.743126 ┆ 0.546    ┆ -283.593002 │
│ collapsed ┆ 0.591788 ┆ 0.005324 ┆ 26          ┆ 0.96834  ┆ 0.5477   ┆ -152.372356 │
└───────────┴──────────┴──────────┴─────────────┴──────────┴──────────┴─────────────┘
```

(The true ρ is 0.6.)

## Forward orthogonal deviations

**The problem with gaps.** First differences lose two observations at every
gap in a panel: the one missing, and the next, which has no difference.
Forward orthogonal deviations (`transform="fod"`, xtabond2's `orthogonal`)
subtract the mean of all later observations, so they lose only one.

```text
fd: 29,519 observations, rho = 0.6017 (0.0081)
fod: 31,902 observations, rho = 0.6013 (0.0058)
```

## Principal components

`pca=True` replaces the GMM-style instruments by the principal components of
their correlation matrix. It keeps those with eigenvalues of 1 or more, as
xtabond2 does; `components=` fixes the number instead. The
Kaiser–Meyer–Olkin statistic measures how well the instruments suit the
reduction.

```text
pca: 11 principal components (KMO 0.904); rho = 0.5752 (0.0198)
```

## Instruments as objects

The formula's `gmm()` and `iv()` terms can also be passed as `Gmm` and `Iv`
objects. That is convenient in a loop:

```python
from dpd_stream import Gmm, Iv

for b in (3, 4, 5):
    r = dpd_stream("y ~ l(y, 1) + x + w | year", df, panel=("id", "year"),
                   gmm=[Gmm("y", lags=(2, b)), Gmm("x", lags=(1, b - 1))], iv=[Iv("w")])
```

```text
lags 2-3: rho = 0.5931, Hansen p = 0.743, difference-in-Hansen p (x's instruments) = 0.818
lags 2-4: rho = 0.5934, Hansen p = 0.355, difference-in-Hansen p (x's instruments) = 0.689
lags 2-5: rho = 0.5931, Hansen p = 0.410, difference-in-Hansen p (x's instruments) = 0.581
```

## The suboptions

| Suboption | Effect |
|---|---|
| `gmm(x, a:b)` | lags a to b; `.` for no upper limit; `1:.` by default, as xtabond2 |
| `collapse=True` | one instrument per lag and variable (or `collapse=True` on the call, for every group) |
| `eq='diff'` / `'level'` / `'both'` | which equation the instruments are for, in system GMM; `'level'` gives differenced instruments for the levels equation |
| `split=True` | system GMM: the levels instruments get their own difference-in-Hansen test |
| `passthru=True` | levels-equation instruments are not differenced |
| `orthogonal=True` | backward orthogonal deviations of the variable, to pair with `transform="fod"` |
| `iv(x, eq=..., passthru=True, mz=True)` | IV-style: one column per variable, transformed for the transformed equation unless `passthru`; `mz`: missing becomes 0 instead of dropping the observation |

:::{dropdown} The script
```{literalinclude} ../../examples/instruments.py
:language: python
```
:::
