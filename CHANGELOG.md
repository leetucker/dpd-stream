# Changelog

## Unreleased: 0.1.0

The first version.

- **Estimators.** Arellano–Bond difference GMM and Blundell–Bond system GMM;
  one-step, two-step or iterated; first differences or forward orthogonal
  deviations. Where conventions differ, the design follows Stata's xtabond2
  (3.7.2), the reference for the instrument layout, the generalized inverse
  (including `invsym()`'s treatment of negative pivots), the weights'
  normalizations and the tests.
- **Instruments.** Lag limits, `collapse`, `eq`, `split`, `passthru`, backward
  orthogonal deviations, IV-style `mz`, `h()` and `pca`, written in the
  formula (`gmm()`, `iv()`) or passed as `Gmm` / `Iv` objects.
- **Out of core.** Polars reads the data once, from a Parquet file or glob or
  a Lazy- or DataFrame, into blocks of whole panels. Three passes over the
  blocks, in numba, give two-step estimates with Windmeijer's correction and
  the Arellano–Bond tests; everything else comes from the sums. No instrument
  matrix is ever built.
- **Inference.** Windmeijer's (2005) correction; one- and multi-way
  clustering; `small`; aweights, pweights and fweights; the Arellano–Bond
  AR(m), Sargan, Hansen and difference-in-Hansen tests; Andrews and Lu's
  MMSC; delta-method long-run effects (`long_run`).
- **Identification tests.** Kleibergen–Paap rk LM and Wald and
  Sanderson–Windmeijer tests (Windmeijer 2025), robust to any clustering,
  with weights, PCA and absorbed effects; first-stage partial and Shea
  partial R², the Anderson–Rubin test (and, with one endogenous regressor, its
  confidence set) and Montiel Olea and Pflueger's effective F. They are
  ivreg2's forms, not the LIML-based ones of Stata's `underid`, so the digits
  differ (see the identification page). The algebra is hdfe-stream's
  `core.ivtests`.
- **Ahn–Schmidt nonlinear moments** (`nl="noserial"` or `"iid"`), with
  xtdpdgmm's first step, one-step fits and Windmeijer's correction in its
  general form. The nonlinear fits' standard errors are the exact ones for the
  moments and differ by a few percent from xtdpdgmm's (see the nonlinear
  page).
- **Absorbed effects.** Time effects (`| year`) and, in difference GMM,
  period-by-group effects (`| region^year`), with the estimates, variances and
  tests of their dummies, under first differences or orthogonal deviations,
  weights, any clustering and iterated GMM.
- **`DPDResult`.** A hdfe-stream `RegressionResult`: `summary()`, `tidy()`,
  `confint()`, `wald_test()`, `etable`, `summary_json()`, `with_vcov()`, the
  one-step results of a two-step fit (`onestep`), and lazy `predict()`,
  `resid()` and `sample()` over the source.
- **`dpd_stream.simulate`.** `simulate_dynamic` and `truth_dynamic`: dynamic
  panels with predetermined, endogenous or exogenous regressors, stationary or
  not, unbalanced, with gaps, clusters and weights.
- **Tested against** a dense implementation of the algebra, Roodman's (2009)
  published output, pydynpd 0.2.2 and plm 2.6.7's `pgmm` (on a simulated panel
  shipped with the tests; Arellano and Bond's data are not distributed), and
  Stata's xtabond2 and xtdpdgmm on about fifty specifications, whose bundle
  is in `benchmarks/external/stata`.
