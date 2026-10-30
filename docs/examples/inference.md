# Standard errors

{download}`Download the script: inference.py <../../examples/inference.py>`

**Why the two-step errors need a correction.** Two-step GMM is
asymptotically efficient, but its conventional standard errors are badly
biased downward in finite samples. Windmeijer (2005) derived a correction
for the uncertainty in the first-step estimate that the weight matrix uses.
It is the default, `vcov="robust"`: clustered by panel, and
Windmeijer-corrected for two-step. `with_vcov("iid")` gives the uncorrected
two-step errors, and every two-step fit carries its one-step results.

```python
two = dpd_stream(F, df, panel=("id", "year"))
two.se                    # Windmeijer-corrected
two.with_vcov("iid").se   # uncorrected
two.onestep.se            # one-step, robust
```

```text
two-step, Windmeijer-corrected: [0.0095 0.0145 0.0095]
two-step, uncorrected:         [0.0092 0.014  0.0093]
one-step, robust:              [0.0093 0.0143 0.0095]
```

## Clustering

`vcov={"CRV1": "cluster"}` clusters by any variable. As with xtabond2's
`cluster()`, the clusters also define the two-step weight matrix, so they
change the two-step estimates, not only the standard errors.

Several variables, as in `{"CRV1": "firm+year"}`, cluster by every
combination with inclusion and exclusion (Cameron, Gelbach and Miller
2011).

```text
clustered by cluster: [0.0123 0.0193 0.0084] {'cluster': 60}
one-step, two-way (cluster, year): [0.0332 0.0469 0.0117] (l(y, 1), x, w)
```

**Two-way clustering with few clusters.** With few clusters in one dimension
(years, typically), multi-way clustering is fragile, in xtabond2 as here:

- Its part of the moment covariance has the rank of its cluster count. So a
  two-step weight matrix built on it is singular, and the estimates
  unreliable.
- The inclusion–exclusion variance can be negative for coefficients that
  vary only in that dimension, here the (absorbed) year effects.

Clustering a one-step fit avoids the first problem.

## Small-sample statistics

`small=True` gives t and F statistics, with xtabond2's degrees of freedom
and small-sample factors:

```text
│ l(y, 1)     ┆ 0.49296   ┆ 0.009475   ┆ 52.027339 ┆ 0.0         │
...
{'statistic': 3315.6, 'pvalue': 0.0, 'df1': 2, 'df2': 3000, 'distribution': 'F'}
```

:::{dropdown} The script
```{literalinclude} ../../examples/inference.py
:language: python
```
:::
