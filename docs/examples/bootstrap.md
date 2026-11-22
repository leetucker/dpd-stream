# Bootstrap inference

{download}`Download the script: bootstrap.py <../../examples/bootstrap.py>`

`fit.refit_bootstrap()` resamples whole panels with replacement and refits the
model on each draw, with the fit's own options. It gives bootstrap standard
errors and intervals, and a bootstrap p-value for the Hansen test.

**When to use it.** For a two-step fit, Windmeijer's correction (the default,
`vcov="robust"`) is usually enough. The bootstrap is worth its cost when:

- **the fit is iterated GMM** (`steps="iterated"`), whose reported standard
  errors are uncorrected and too small;
- **the instruments are many relative to the panels**, where the corrected
  errors can still be too small, and iterated GMM's are much too small;
- **you need a function of the coefficients**, such as a long-run effect,
  without the delta method.

```python
fit = dpd_stream(F, df, panel=("id", "year"))
boot = fit.refit_bootstrap(reps=499, seed=1, n_workers=4)
boot.standard_errors()
boot.tidy(method="studentized")
boot.hansen()
```

```text
instruments: 49  panels: 250
Windmeijer-corrected: [0.0264 0.0385 0.0359]
bootstrap:            [0.0239 0.0363 0.0341]
┌─────────────┬──────────┬──────────┬──────────┬────────────┬─────────────┬───────────┬────────────┐
│ Coefficient ┆ Estimate ┆ Lower    ┆ Upper    ┆ Std. Error ┆ Bootstrap   ┆ Bias      ┆ Successful │
│             ┆          ┆          ┆          ┆            ┆ Mean        ┆           ┆ Reps       │
╞═════════════╪══════════╪══════════╪══════════╪════════════╪═════════════╪═══════════╪════════════╡
│ l(y, 1)     ┆ 0.482572 ┆ 0.442094 ┆ 0.55366  ┆ 0.023903   ┆ 0.46926     ┆ -0.013312 ┆ 499        │
│ x           ┆ 1.003878 ┆ 0.932552 ┆ 1.104169 ┆ 0.036297   ┆ 0.999819    ┆ -0.00406  ┆ 499        │
│ w           ┆ 0.448145 ┆ 0.378549 ┆ 0.525602 ┆ 0.034092   ┆ 0.445905    ┆ -0.002241 ┆ 499        │
└─────────────┴──────────┴──────────┴──────────┴────────────┴─────────────┴───────────┴────────────┘
{'statistic': 45.09, 'df': 46, 'p': 0.5105, 'bootstrap_p': 0.782, 'successful_reps': 499}
```

## What is resampled

Each draw takes as many panels as the data have, with replacement. Each copy of
a panel gets an ID of its own, so its lags and instruments are built exactly
as in the data. The refit keeps the original fit's instrument columns, so every
draw estimates the same model.

With `vcov={"CRV1": "g"}` and `g` constant within panels (a region, an
industry), whole clusters are drawn instead, and each copy is a cluster of its
own. This is Stata's
`bootstrap, cluster(g) idcluster(newg): xtabond2 ...`.

## Recentering, and which interval to report

By default each draw's moments are **recentered**: the moment conditions of the
draw, less their value in the data at the fit's estimate (Hall and Horowitz
1996). An overidentified model's moments are not exactly zero at its own
estimate, so without recentering the draws come from a population in which the
model is wrong. The draws' estimates, weight matrices, standard errors and
Hansen statistics all use the recentered moments. This is what makes the
following valid:

- **Studentized intervals** (`method="studentized"`, or `"symmetric"`). Each
  draw's t-statistic, (b* − b) / se*, uses the draw's own standard error, and
  its quantiles replace the normal ones. These are the intervals to report
  (see [Accuracy](#accuracy)). They need
  each draw's standard errors (`draw_standard_errors=True`, the default).

  `"studentized"` is equal-tailed: each bound comes from one tail of t\*.
  `"symmetric"` uses the quantile of |t\*|, so the interval is symmetric
  around the estimate. Hall and Horowitz (1996) find that the bootstrap
  improves on first-order asymptotics more for symmetric tests and intervals
  based on GMM estimators than for one-sided ones: through O(n⁻¹), against
  O(n⁻¹ᐟ²) (Horowitz 2001, section 4.1.2). Their result is for dependent data
  resampled in blocks. In this page's Monte Carlo study, coverage of nominal
  95% intervals was 95–98% for the symmetric intervals and 93–96% for the
  equal-tailed ones.
- **The bootstrap Hansen test** (`hansen()`). Its p-value is the share of
  the draws' statistics at least the fit's, (1 + #{J* ≥ J}) / (1 + B), so
  with B draws it is never below 1 / (1 + B).

:::{note}
With many instruments relative to the panels, the bootstrap Hansen test does
not perform as well as the studentized intervals. The studentized intervals
cover close to 95% even with many instruments. The bootstrap Hansen test does
not repair the Hansen test's weakness there: it rejects even less often than
the asymptotic test, both when the instruments are valid and when one is
invalid. With few instruments, the bootstrap and asymptotic tests agree. See
[Accuracy](#accuracy).
:::

The percentile, basic and normal intervals and the bootstrap standard errors
are available whether or not the draws are recentered. `recenter=False` gives
the ordinary pairs bootstrap, Stata's `bootstrap` prefix, which has no Hansen
test.

**When the draws are not recentered.** Recentering is not available with
`weights`, an absorbed `| g^year` effect, `nl()` or `pca`. With
`recenter="auto"` (the default) such a fit is bootstrapped without it, and
`boot.diagnostics["recenter_reason"]` says why; `recenter=True` raises an
error instead. Report percentile intervals for these.

## Functions of the coefficients

`boot.draws()` holds every draw's estimates, so any function of them has a
percentile interval:

```python
d = boot.draws()
long_run = (d["x"] / (1 - d["l(y, 1)"])).to_numpy()
np.quantile(long_run, [0.025, 0.975])
```

```text
long-run effect of x: 1.9401 95% percentile interval: [1.6441 2.1313]
```

## Iterated GMM

Windmeijer's correction applies to two-step estimates only, so an iterated
fit reports the uncorrected variance. The bootstrap corrects it:

```text
iterated, uncorrected: [0.0201 0.0306 0.0288]
iterated, bootstrap:   [0.0255 0.0391 0.0359]
```

## Clusters

```text
clusters (cluster) 49 clusters
clustered, bootstrap: [0.0281 0.0598 0.0437]
```

Two-way clustering (`{"CRV1": "g1+g2"}`) and clustering by a variable that
varies within panels (such as the year) have no scheme that resamples whole
panels, and raise an error.

(accuracy)=
## Accuracy

`benchmarks/bootstrap_coverage.py` simulates 300 samples of 200 panels of 8
periods, with valid moment conditions. It fits difference GMM with every
available lag as an instrument, which gives 49 instruments for 200 panels.
Each fit is bootstrapped with 199 draws.

**Intervals.** The share of 95% intervals that cover the true coefficient:

| Fit | Coefficient | Analytic | Percentile | Studentized | Symmetric |
|---|---|---|---|---|---|
| two-step | l(y, 1) | 0.920 | 0.833 | 0.953 | 0.957 |
| | x | 0.930 | 0.890 | 0.933 | 0.963 |
| | w | 0.923 | 0.883 | 0.933 | 0.947 |
| iterated | l(y, 1) | 0.807 | 0.843 | 0.963 | 0.970 |
| | x | 0.817 | 0.890 | 0.963 | 0.983 |
| | w | 0.803 | 0.880 | 0.950 | 0.967 |

"Analytic" is Windmeijer-corrected for two-step, and uncorrected for iterated
GMM. The studentized intervals come closest to 95%, and they repair iterated
GMM's intervals, which cover only about 80% of the time. The percentile
intervals cover too little. The estimates are biased (the lag's coefficient by
about half a standard deviation, downward), and the draws are biased the same way
around the estimate. So percentile intervals sit further from the truth, while
studentized ones, b − t*·se, correct for the bias. With 300 samples,
a coverage rate has a standard error of about 0.013.

**The Hansen test.** The share of samples in which the test rejects at 5%:

| Instruments | Moments | Asymptotic | Bootstrap |
|---|---|---|---|
| 49 | valid | 0.020 | 0.003 |
| 49 | x's current value added (invalid) | 0.383 | 0.047 |
| 14 (`collapse=True`) | valid | 0.057 | 0.043 |
| 14 (`collapse=True`) | x's current value added (invalid) | 0.720 | 0.687 |

With many instruments the asymptotic Hansen statistic is too small: it
averages about 40 here, against 46 degrees of freedom. That is why it rarely
rejects (Roodman 2009). The bootstrap does not reproduce this: the draws'
statistics average about 52. Each draw holds only about 126 distinct panels,
which makes its estimated weight matrix less precise and its statistic
larger. The bootstrap test is then even more conservative than the
asymptotic one, and it loses its power. With few instruments, the two tests
agree, in size and in power. So the bootstrap Hansen test does not improve on
the asymptotic one in either case. When the Hansen test matters, reduce the
instruments (`collapse`, lag limits).

**With few instruments** (`collapse=True`, 14 instruments), every interval,
analytic or bootstrap, covers 93% to 97% of the time. Then the bootstrap is
needed only for functions of the coefficients.

## Cost, and failed draws

A bootstrap takes about `reps` fits. For a two-step fit, each draw's standard
errors take one more pass over the draw (Windmeijer's correction).
`draw_standard_errors=False` skips it, and then the studentized intervals are
not available. Refits skip every statistic they don't need: the
Arellano–Bond, difference-in-Hansen and identification tests.

`n_workers=` runs the refits in parallel processes and shares out
`n_threads` and `max_memory_gb` among them. Each draw is written to disk, in
bounded memory, and only one draw per worker exists at a time. The same seed
gives the same draws, whatever the number of workers.

A draw fails when it lacks a period of the data, which changes the
instruments, or drops a different regressor as collinear. A failed draw is
recorded in `boot.failures()` and not redrawn. Inference on the rest then
needs `allow_incomplete=True`, and is conditional on the draws that succeeded.

## References

- Hall, P., and J. L. Horowitz (1996). Bootstrap critical values for tests
  based on generalized-method-of-moments estimators. *Econometrica* 64(4),
  891–916.
- Horowitz, J. L. (2001). The bootstrap. In J. J. Heckman and E. Leamer
  (eds.), *Handbook of Econometrics*, vol. 5, ch. 52, 3159–3228. Elsevier.
- Roodman, D. (2009). A note on the theme of too many instruments. *Oxford
  Bulletin of Economics and Statistics* 71(1), 135–158.
- Windmeijer, F. (2005). A finite sample correction for the variance of
  linear efficient two-step GMM estimators. *Journal of Econometrics* 126(1),
  25–51.

:::{dropdown} The script
```{literalinclude} ../../examples/bootstrap.py
:language: python
```
:::
