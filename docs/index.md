# dpd-stream

Arellano–Bond (difference) and Blundell–Bond (system) GMM for dynamic panels,
out of core. One-step, two-step or iterated estimation with flexible
instrument sets, Windmeijer-corrected and clustered inference, specification
and identification tests, Ahn–Schmidt nonlinear moments and absorbed fixed
effects. It works on data far larger than memory, and on large data it is much
faster than the existing implementations. Where conventions differ, it follows
Stata's xtabond2, and its results are tested against xtabond2, xtdpdgmm and
other implementations.

```python
from dpd_stream import dpd_stream

fit = dpd_stream("n ~ l(n, 1) + w + k | year | gmm(n, 2:4) + gmm(w + k, 1:3)",
                 "firms/*.parquet", panel=("firm", "year"), system=True)
fit.summary()
```

## Why

A regression of an outcome on its own lag, $y_{it} = \rho y_{i,t-1} + x_{it}'\beta
+ \alpha_i + \varepsilon_{it}$, cannot be estimated by OLS or within-groups:

- **OLS** is biased upward. The lag carries the fixed effect $\alpha_i$.
- **Within-groups** is biased downward when panels are short. Demeaning makes
  the lag correlated with the demeaned error (Nickell 1981).

Arellano and Bond (1991) difference the fixed effect away and instrument the
differenced equation with lagged levels. Arellano and Bover (1995) and
Blundell and Bond (1998) add the levels equation, instrumented with lagged
differences ("system GMM"). For two decades these estimators have been the
standard for short dynamic panels: firm investment, employment, growth,
trade, prices.

The implementations everyone uses build each panel's instrument matrix in
memory. With T periods, uncollapsed instruments number O(T²) per variable,
and the instrument matrix is N·T rows by O(T²) columns. These are Stata's
xtabond2 and xtdpdgmm, R's `plm::pgmm` and `pdynmc`, and Python's `pydynpd`.
Most hold all of it in memory, and several are written in interpreted loops
over panels. Panels of millions of units, which administrative and linked
employer–employee data routinely have, are out of reach.

## How it is fast

Everything linear GMM needs is a sum over panels:

- **Z′X, Z′y and Σᵢ Zᵢ′HZᵢ** are fixed by the data.
- **The moment covariance Σᵢ gᵢgᵢ′**, with gᵢ = Zᵢ′eᵢ, needs the residuals at
  the estimates.

dpd-stream reads the data once into blocks of whole panels, sorted by time.
After that, every pass is a numba kernel that builds each panel's rows and
its instruments. The instruments are built as a sparse list per row, so no
instrument matrix exists at any point. The kernel adds the panel into these
sums; batches of gᵢ go to BLAS.

Two-step GMM with Windmeijer-corrected errors and the Arellano–Bond tests
takes three passes, and everything else comes from the sums:

- the Sargan, Hansen and difference-in-Hansen tests;
- MMSC;
- principal components of the instruments.

See [how it works](how_it_works.md).

## Features

**Estimators**
- Difference or system GMM: one-step, two-step or iterated.
- Transformation by first differences or forward orthogonal deviations.

**Instruments**, with every xtabond2 instrument option:
- GMM-style with lag limits, `collapse`, `eq='diff'|'level'|'both'`, `split`,
  `passthru`, and backward orthogonal deviations (`orthogonal`);
- IV-style with `eq`, `passthru` and `mz`;
- xtabond2's `h()` matrices, and `pca`.

**Inference**, as xtabond2 computes it:
- Windmeijer's (2005) correction;
- one-way and multi-way clustering, which also define the two-step weight
  matrix;
- `small`-sample statistics;
- the Arellano–Bond AR(m), Sargan, Hansen and difference-in-Hansen tests.

**Beyond xtabond2**
- Iterated GMM.
- Andrews–Lu model and moment selection criteria.
- Kleibergen–Paap and Sanderson–Windmeijer identification tests (Windmeijer
  2025), robust to the fit's clustering.
- Delta-method long-run effects.
- Ahn and Schmidt's (1995) nonlinear moment conditions, under no serial
  correlation and under homoskedasticity, as xtdpdgmm's `nl()`.
- Absorbed time effects (`| year`), and absorbed high-dimensional period
  effects by group (`| region^year`) in difference GMM.

**Interface**
- pyfixest-style formulas, with fixest's `l()`/`f()`/`d()` lag operators.
- Results that work with hdfe-stream's reporting: `summary()`, `tidy()`,
  `etable`, `wald_test`, `confint`, `summary_json`; and lazy `predict()`,
  `resid()` and `sample()` frames.

## When to use this, and when not to

**Use it when:**
- the data are large, from tens of thousands of panels to hundreds of
  millions of rows, or don't fit in memory;
- you are fitting many specifications and want each in seconds;
- you want xtabond2's exact results in Python.

On Arellano and Bond's 140 firms any implementation is instant.

**Not available:**
- continuously-updating GMM;
- xtdpdgmm's more specialized options;
- several absorbed fixed effects at once;
- GPUs.

See the [benchmarks](benchmarks/index.md).

## Install

```bash
pip install dpd-stream
```

It needs Python 3.10 or later. It brings hdfe-stream (whose streaming core
and result classes it uses), Polars, NumPy, numba and SciPy.

```{toctree}
:hidden:
:caption: Guide

examples/index
how_it_works
```

```{toctree}
:hidden:
:caption: Benchmarks

benchmarks/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
changelog
```
