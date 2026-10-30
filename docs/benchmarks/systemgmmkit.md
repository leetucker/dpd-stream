# systemgmmkit

[systemgmmkit](https://github.com/Akanom/systemgmmkit) (1.0.5) is a new
Python package for difference and system GMM, among other panel estimators.
Its native engine builds each panel's instrument matrix in memory, as
pydynpd does, and follows pydynpd's conventions for instruments. It is in the
speed benchmarks (the collapsed models: that is all its native engine
builds), and this page compares its numbers with dpd-stream's.

The comparison is `benchmarks/external/systemgmmkit_compare.py`. It fits 300
simulated panels of 8 periods (y on its first lag, x predetermined, w
exogenous; collapsed GMM-style instruments, two-step, Windmeijer-corrected)
with both, and reports the largest differences. Two conventions have to be
matched first, and neither is an error in either package:

- **Standard errors.** systemgmmkit applies xtabond2's small-sample factor
  $G/(G-1)\cdot(N-1)/(N-K)$ to the variance. They equal dpd-stream's with
  `small=True` to ten digits.
- **IV-style instruments in system GMM.** systemgmmkit uses them in the
  levels equation only; xtabond2's, and dpd-stream's default, is both
  equations. `iv(w, eq='level')` is systemgmmkit's.

```{list-table}
:header-rows: 1

* - Model
  - Panel
  - Coefficients
  - Standard errors
  - Hansen
* - Difference GMM
  - balanced, unbalanced, with time effects
  - 2e-15
  - 2e-15 (relative)
  - 4e-14
* - System GMM
  - balanced, unbalanced
  - 1e-15
  - 2e-5 (relative)
  - 1e-13
* - System GMM
  - with time effects
  - 0.06
  - 17%
  - 11
* - Any
  - unbalanced with gaps
  - 0.01
  - 3–6%
  - 0.4–0.7
* - Orthogonal deviations
  - balanced
  - difference: 0.11; system: 2e-15
  - difference: 2.2 times; system: 3e-6
  - difference: 1.3; system: 1e-13
```

Largest differences, from `benchmarks/results/systemgmmkit_compare.csv`
(300 panels, seed 7; re-run it for the other statistics).

**Where they agree.** Difference GMM in first differences agrees to rounding
in the estimates, standard errors and the Arellano–Bond, Sargan and Hansen
statistics, balanced or unbalanced and with time effects. So does system
GMM without time effects, except for the Arellano–Bond AR(1) statistic
(−8.203 and −8.212 on one unbalanced panel).

**Where they do not.**

- **Gaps.** systemgmmkit takes lags by position in the panel, dpd-stream (like
  xtabond2) by the value of the period. A panel with gaps therefore gives
  systemgmmkit more observations (752 against 734 in the example) and
  different estimates. The two agree on unbalanced panels without gaps.
- **Time effects in system GMM.** The two place the dummies' instruments
  differently, and the estimates differ.
- **Forward orthogonal deviations.** Difference GMM with them has fewer
  instruments in systemgmmkit (12 against 14 here) and different estimates,
  and its Arellano–Bond AR(1) statistic is near zero where dpd-stream's, as
  Stata's, is large and negative. System GMM with them agrees in the
  estimates and the Hansen statistic, but not in the Sargan statistic.

**What it does not have.** Its native engine builds collapsed instruments
only (it ignores `collapse=False`), uses one clustering (by panel) and no
weights, and has none of dpd-stream's Ahn–Schmidt moments, absorbed effects
or identification tests. It is not a second reference for those.

**Speed.** On the collapsed models it takes 2 to 4 seconds at 1,000 panels,
25 to 43 seconds at 10,000 and 475 to 500 seconds at 100,000 (1.8 to 3.2 GB),
against 0.1, 0.1 and 0.5 to 0.6 seconds for dpd-stream. At a million panels
it refused the data: its check on the time variable raised "Integral
panel-time labels imply an unsafe unit-spaced grid" for 8 million rows, and
asks for a contiguous period index. We did not recode the time variable for it.

These differences were found and characterized, not traced to their code as
the Stata ones were; systemgmmkit is new, and its authors state that its
agreement with xtabond2 is certified on six specifications.
