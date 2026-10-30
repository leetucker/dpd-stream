# Arellano and Bond's employment equations

{download}`Download the script: arellano_bond.py <../../examples/arellano_bond.py>`

Arellano and Bond (1991) estimated a dynamic labor demand equation for 140
UK firms over 1976–1984. These are Stata's `abdata`, which the script
downloads from Stata Press (they are not distributed with dpd-stream;
reading them needs pandas):

```python
import io
import urllib.request

import pandas as pd
import polars as pl

URL = "https://www.stata-press.com/data/r18/abdata.dta"
req = urllib.request.Request(URL, headers={"User-Agent": "dpd-stream example"})
with urllib.request.urlopen(req) as resp:
    raw = io.BytesIO(resp.read())
d = (pl.from_pandas(pd.read_stata(raw)).rename({"id": "firm"})
     .with_columns(pl.col("firm").cast(pl.Int64), pl.col("year").cast(pl.Int64)))
```

Roodman (2009) uses the same data and specifications to introduce
xtabond2. dpd-stream reproduces his tables to the digits printed (pp.
111–118).

The variables are logs:

| Variable | Log of |
|---|---|
| `n` | employment |
| `w` | the wage |
| `k` | capital |
| `ys` | industry output |

## Difference GMM, one step (Table 4, column a1)

This is xtabond2's
`xtabond2 n L.n L2.n w L.w L(0/2).(k ys) yr*, gmm(L.n) iv(w L.w L(0/2).(k ys) yr*) noleveleq robust`:

```python
YR = " + ".join(f"yr{y}" for y in range(1976, 1985))
ab = dpd_stream("n ~ l(n, 1:2) + w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + " + YR
                + " | gmm(l(n, 1)) + iv(w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + "
                + YR + ")", d, panel=("firm", "year"), steps=1)
```

```text
Dynamic panel-data estimation, one-step difference GMM (dpd_stream)   vcov: robust
observations: 611   groups: 140   obs per group: min 4, avg 4.36, max 6   instruments: 41
Wald test of all coefficients: chi2(16) = 1727.45, p = 0.0000
Arellano-Bond test for AR(1) in first differences: z = -3.60, p = 0.0003
Arellano-Bond test for AR(2) in first differences: z = -0.52, p = 0.6058
Sargan test of overid. restrictions: chi2(25) = 67.59, p = 0.0000  (not robust)
Hansen test of overid. restrictions: chi2(25) = 31.38, p = 0.1767  (robust)
Difference-in-Hansen tests of exogeneity of instrument subsets:
  iv(w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + yr1976 + ... + yr1984)
    excluding group: chi2(11) = 12.01, p = 0.3632   difference: chi2(14) = 19.37, p = 0.1511
Underidentification (Kleibergen-Paap rk LM, robust): chi2(26) = 32.66, p = 0.1723   weak identification (KP rk Wald F): 2.24
  ...
dropped as collinear: ['yr1976', 'yr1977', 'yr1984']
│ l(n, 1)     ┆ 0.686226  ┆ 0.144594   ┆ ...
│ l(n, 2)     ┆ -0.085358 ┆ 0.056016   ┆ ...
│ w           ┆ -0.607821 ┆ 0.178205   ┆ ...
│ l(w, 1)     ┆ 0.392624  ┆ 0.167993   ┆ ...
│ k           ┆ 0.356846  ┆ 0.05902    ┆ ...
```

Roodman's output (p. 111) has the same numbers:

- L1.n: .6862261 (SE .1445943);
- the AR tests −3.60 and −0.52;
- Sargan 67.59 and Hansen 31.38;
- excluding the group, 12.01; the difference, 19.37.

The year dummies are dropped as collinear as in Stata (1976, 1977 and 1984).

## Two steps (column a2)

`gmm(n, 2:.)` is the same instrument set as `gmm(l(n, 1))`, that is, lags 2
and deeper of n. The two-step fit has Windmeijer-corrected standard errors:

```text
│ l(n, 1)     ┆ 0.628709  ┆ 0.193414   ┆ ...
│ l(n, 2)     ┆ -0.065188 ┆ 0.04505    ┆ ...
│ w           ┆ -0.52576  ┆ 0.154611   ┆ ...
```

These are the article's column (a2). On R plm's copy of the data
(`EmplUK`, in double precision, where Stata's is float), they also match
plm's `pgmm` example to every digit: the estimates, its `vcovHC` errors
(Windmeijer's), the Sargan/Hansen statistic and the AR tests.

## System GMM (Blundell and Bond)

Wages and capital are endogenous, instrumented GMM-style, and the year
dummies instrument the levels equation. This is xtabond2's
`gmmstyle(L.(n w k)) ivstyle(yr*, equation(level)) robust small`:

```python
bb = dpd_stream("n ~ l(n, 1) + w + l(w, 1) + k + l(k, 1) + " + YR
                + " | gmm(l(n, 1) + l(w, 1) + l(k, 1)) + iv(" + YR + ", eq='level')",
                d, panel=("firm", "year"), system=True, steps=1, small=True)
```

```text
Dynamic panel-data estimation, one-step system GMM (dpd_stream)   vcov: robust
observations: 891   groups: 140   obs per group: min 6, avg 6.36, max 8   instruments: 113
Arellano-Bond test for AR(1) in first differences: z = -5.46, p = 0.0000
Arellano-Bond test for AR(2) in first differences: z = -0.25, p = 0.8038
Hansen test of overid. restrictions: chi2(100) = 110.70, p = 0.2183  (robust)
  GMM instruments for levels
    excluding group: chi2(79) = 84.33, p = 0.3200   difference: chi2(21) = 26.37, p = 0.1927
│ l(n, 1)     ┆ 0.935605  ┆ 0.026569   ┆ ...
│ Intercept   ┆ 0.552201  ┆ 0.197162   ┆ ...
```

**Two numbers differ from the article,** and both because of changes in
xtabond2 since 2009. dpd-stream follows the current version, 3.7.2:

- **The Sargan statistic** (157.54; the article has 186.90). σ² now divides
  by the 751 observations of the differenced equation, not the 891 of the
  levels equation.
- **The model F** (6359; the article has 1154). The test now includes the
  constant.

With the old conventions undone, both agree. `tests/test_published.py`
checks this.

**Instrument proliferation.** 113 instruments for 140 groups is the
proliferation Roodman (2009b) warns about. dpd-stream warns when the
instruments outnumber the groups, as xtabond2 does.

:::{dropdown} The script
```{literalinclude} ../../examples/arellano_bond.py
:language: python
```
:::
