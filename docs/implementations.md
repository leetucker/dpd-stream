# Differences between implementations

dpd-stream follows Stata's xtabond2 where conventions differ (xtabond2
3.7.2). This page lists the differences we found when comparing it with other
implementations, at the versions below, so that a difference in someone's
output can be traced. "Differs" is not "wrong": most are different conventions
for the same model, and where one is a limitation or a likely bug, we say so.

| Implementation | Version | Compared on |
|---|---|---|
| xtabond2 | 3.7.2 | about fifty specifications; the reference |
| xtdpdgmm | | nonlinear moments, system GMM, forward orthogonal deviations |
| ivreg2 / `underid` | | identification tests |
| pydynpd | 0.2.2 | difference and system GMM |
| plm `pgmm` | 2.6.7 | difference and system GMM, Windmeijer, AR tests |
| pdynmc | 0.9.13 | difference and system GMM, nonlinear moments |
| systemgmmkit | 1.0.5 | collapsed difference and system GMM |

## Where they agree

- **xtabond2** agrees with dpd-stream to 1e-5 or better in coefficients,
  standard errors and every test, on the specifications that both run.
- **pydynpd** agrees to rounding in difference and system GMM, collapsed or
  not.
- **plm and pdynmc** agree to rounding in difference GMM (plm's Windmeijer
  errors and AR tests too, on the Arellano–Bond data).
- **systemgmmkit** agrees to rounding in first-differenced difference GMM,
  and in system GMM once its conventions below are matched, on balanced panels
  and on unbalanced ones without gaps.

## Conventions that change the estimates

**Which equation an IV-style instrument serves.** In system GMM, xtabond2
(and dpd-stream by default) uses an IV-style instrument such as the exogenous
$w$ in both equations: $E[\Delta w_{it}\,\Delta\varepsilon_{it}] = 0$ and
$E[w_{it}(\alpha_i + \varepsilon_{it})] = 0$. systemgmmkit uses it in the levels
equation only (`iv(w, eq='level')` is the same model here). xtdpdgmm's
`iv(w)` in a differenced equation uses $w$ itself, not $\Delta w$ (`iv(w,
difference)` is xtabond2's). dpd-stream takes `iv(w, passthru=True)` for
xtdpdgmm's.

**Time effects in the levels equation.** In system GMM, dpd-stream's year
effects are xtabond2's: a dummy per year, each an IV-style instrument for both
equations (`iv(yr*)`), checked against xtabond2 at 1e-5 or better (Arellano
and Bond's data, and a simulated panel with `| year`). pydynpd's `timedumm`
agrees with it to rounding. plm and pdynmc give system-GMM estimates 0.1 to
0.14 from ours on the benchmark panel, which we attribute to how they enter the
year effects in the levels equation, though we did not trace it. systemgmmkit
with time effects also differs, from placing the dummies' instruments
differently. Difference GMM agrees in all of them.

**Lags by period or by row.** xtabond2 and dpd-stream take lags by the value
of the time variable, so a gap makes the lag missing. systemgmmkit takes them
by position in the panel, so across a gap it has more observations (752
against 734 in one test panel) and different estimates. The two agree on
unbalanced panels without gaps.

**Forward orthogonal deviations.** The deviation is
$\varepsilon^*_{it} = c_t(\varepsilon_{it} - \bar\varepsilon_{i,t+1:T})$. With
$c_t^2 = (T-t)/(T-t+1)$ the transformed errors have equal variances, which
is xtabond2's choice and dpd-stream's. xtdpdgmm uses $c_t^2$ in place of
$c_t$, so it weights the rows differently, and the estimates differ whenever
instruments are shared across periods (collapsed ones, IV-style ones; with
one instrument per period the scale cancels). We reproduced xtdpdgmm's fit
to six digits only with that scale. systemgmmkit's orthogonal-deviation fits
have different instrument sets (12 against 14 in one difference-GMM test)
and different estimates, and its Arellano–Bond AR(1) statistic is near zero
where the others' is large and negative.

**Ahn–Schmidt moments.** dpd-stream and xtdpdgmm agree to six digits, in
difference and system GMM, balanced or not, once the first-step weights match
(the nonlinear block's weight is $1/N_g$ times `nl_weight`, and `nl(iid)`'s
added linear moments sit in the $Z'HZ$ block). pdynmc defines the same
moments but needs a third difference, so its set is one moment shorter; its
one-step fit matches ours with that moment dropped. Its two-step estimate
differs from ours by up to 0.6 standard errors; it is the minimizer of its
own objective, and we did not resolve why that objective differs.

**Collapsed levels instruments.** xtabond2 searches for a collapsed
levels-instrument column that is empty and moves it to deeper lags, across
panel boundaries; dpd-stream does it within each panel.

## Conventions that change only standard errors and statistics

**Small-sample factor.** systemgmmkit multiplies its variance by xtabond2's
`small` factor, $\frac{G}{G-1}\cdot\frac{N-1}{N-K}$; dpd-stream's
`small=True` gives the same standard errors.

**Nonlinear-fit standard errors.** xtdpdgmm's reported variance for
Ahn–Schmidt fits differs by up to 3% (balanced two-step; more unbalanced) from
the exact one for its own moments. Its stored weight matrix and Hansen
statistic equal ours, and our Jacobian satisfies the first-order condition at
its estimates, so we keep ours.

**Statistics that xtabond2 changed after Roodman (2009).** The Sargan
statistic's $\hat\sigma^2$ divides by the differenced equation's observations
(not the levels'), and the model test includes the constant. dpd-stream
follows 3.7.2; the article's two numbers differ for that reason.

**Defaults.** The default estimator differs: xtabond2 and xtdpdgmm are
one-step, pydynpd two-step with Windmeijer's correction, pdynmc iterated.
dpd-stream is two-step Windmeijer. MMSC counts the constant in $k$ in
dpd-stream; pydynpd's omits it while its Hansen degrees of freedom count it.

**Identification tests.** After xtabond2, Stata's `underid` computes
Kleibergen–Paap by `ranktest`'s `nosvd` form (a J statistic at LIML
estimates) and Sanderson–Windmeijer by a two-step GMM with a $Z'HZ$ first
step. dpd-stream computes ivreg2's forms: Kleibergen–Paap by the SVD algebra
and Sanderson–Windmeijer by 2SLS. They are asymptotically equivalent under the
null and differ by a few percent in finite samples
([identification](examples/identification.md)).

## What each implementation offers

- **xtabond2** is the reference for options (`h()`, `split`, `passthru`,
  `mz`, `pca`) and the tests. It holds each panel's instrument matrix in
  memory.
- **xtdpdgmm** has more specialized options than dpd-stream. dpd-stream matches its `nl()`, but not its rescaling in
  unbalanced panels.
- **pydynpd** is a port of xtabond2's commands to Python: difference and
  system GMM, collapsed or not, two-step with Windmeijer's correction.
- **plm `pgmm`** has difference and system GMM with Windmeijer's correction and
  the AR tests.
- **pdynmc** adds Ahn–Schmidt moments and iterated GMM, and is the slowest
  at scale.
- **systemgmmkit** builds collapsed instruments only (`collapse=False` is
  ignored), clusters by panel only, takes no weights, and has no nonlinear
  moments, absorbed effects or identification tests. At a million panels it
  refuses the time variable as an "unsafe unit-spaced grid" and asks for a
  contiguous period index.
- **dpd-stream** has none of: continuously-updating GMM, several absorbed fixed
  effects at once, GPUs, or xtdpdgmm's more specialized options.

For speed and memory at scale, see the [benchmarks](benchmarks/index.md).
`benchmarks/external/` has the scripts behind the numerical comparisons.

## Comparing output

To compare dpd-stream with another implementation, match the model first: the
same equation for each instrument (above), the same lag limits (xtdpdgmm's
`lag(1 3)` is lags 1 to 3 of the variable, as xtabond2's), the same
transformation and scale, the same sample (gaps), and the same variance
(`small`, one- or two-step). A one-step fit with an identity weight depends
on the moments alone, which separates a difference in the moments from a
difference in the weighting.
