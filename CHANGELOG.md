# Changelog

## 0.2.0

- **Fit resources (CPUs and memory) are constrained to your job's limits automatically.** On a
  cluster (PBS, Slurm, or a container with cgroup limits), importing `dpd_stream` caps the threads of
  Polars, numba, BLAS and pyarrow at the job's CPU allocation, and with Polars 2.x gives Polars a
  memory budget so its streaming operations spill to disk instead of growing. Caveats: the Polars
  budget is a target for its buffers, not a hard cap on the process, and Polars 1.x has no such
  option, so there only the threads are capped. The caps must be in place before Polars is
  imported: import `dpd_stream` first, or call `stream_limits.set_limits()` at the top of the script.
  With no limit found nothing changes. See "Resource limits and profiling" in the documentation.
- **`profile=True`: estimate a fit's memory and disk before running it.**
  `dpd_stream(..., profile=True)` returns a `hdfe_stream.Profile` -- the
  estimated memory and disk, and `recommend()` for a resource request -- instead
  of fitting. The instrument count comes from the layout (formula and number of
  periods, no data pass). It is an upper bound on the columns used, since empty
  ones are dropped after pass 1.
- **`max_memory_gb` and `memory_reserve`**: the fixed 8 GB and 2-4 GB thresholds
  (pass 1's per-thread moment matrices, the cluster sums, the absorbed cells) are
  replaced by the memory left (the limit, less a reserve, less what the process
  holds). Per-thread accumulators drop to fewer copies when they would not fit
  (fewer threads on that pass, same answer); cluster sums and absorbed cells that
  cannot shrink raise `MemoryError` up front. The limit defaults to the job's
  allocation (cgroup, `STREAM_LIMITS_MEMORY_GB`, or on PBS `qstat`).
- **`n_threads=` now limits numba.** It used to set only how many accumulators
  were allocated; the kernels still ran on every core. It holds numba and BLAS
  to that many threads for the fit and restores them.
- **The memory warning counts `Z'Z`.** With the identification tests on (the
  default), `Z'Z` takes as much per-thread memory as `S1`, and the warning
  counted only `S1`.
- **`diagnostics["peak_rss_gb"]`** in every fit.
- Depends on `stream-limits`, and needs hdfe-stream 0.8.0 or later.

## 0.1.0

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
