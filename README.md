# dpd-stream

Arellano–Bond (difference) and Blundell–Bond (system) GMM for dynamic panels,
out of core. One-step, two-step or iterated estimation with flexible
instrument sets, Windmeijer-corrected and clustered inference, specification
and identification tests, Ahn–Schmidt nonlinear moments and absorbed fixed
effects. It works on data far larger than memory, and is much faster than the
existing implementations on large panels. Where conventions differ, it follows
Stata's xtabond2, and its results are tested against xtabond2, xtdpdgmm and
other implementations.

```python
from dpd_stream import dpd_stream

fit = dpd_stream("n ~ l(n, 1) + w + k | year | gmm(n, 2:4) + gmm(w + k, 1:3)",
                 "firms/*.parquet", panel=("firm", "year"), system=True)
fit.summary()
```

## What it does

**Estimators**
- Difference or system GMM: one-step, two-step (Windmeijer-corrected) or
  iterated.
- First differences or forward orthogonal deviations.

**Instruments**
- Lag limits, `collapse`, `eq`, `split`, `passthru`, `orthogonal`, `mz`,
  `h()`, `pca`.

**Tests and inference**
- Arellano–Bond AR(m), Sargan, Hansen, difference-in-Hansen.
- Clustering, one- and multi-way.
- `small`-sample statistics.
- Weights.

**Also**
- Iterated GMM.
- MMSC model selection.
- Long-run effects.
- Kleibergen–Paap and Sanderson–Windmeijer identification tests (ivreg2's
  forms; they differ in the digits from Stata's `underid`).
- Ahn–Schmidt nonlinear moments (xtdpdgmm's `nl()`).
- Absorbed time effects, and absorbed period-by-group effects in difference
  GMM.

**Interface**
- pyfixest-style formulas with fixest's `l()`/`f()`/`d()` lag operators.
- Results with hdfe-stream's reporting: `summary()`, `tidy()`, `etable`,
  `wald_test()`, `summary_json()`.

## Why it is fast

Everything linear GMM needs is a sum over panels: Z′X, Z′y, Σ Zᵢ′HZᵢ, and
Σ gᵢgᵢ′ with gᵢ = Zᵢ′eᵢ.

- **One read of the source.** The data are read once into blocks of whole
  panels, sorted by time.
- **No instrument matrix.** Each pass is a numba kernel that builds every
  panel's rows and its sparse instruments and adds them into those sums. No
  instrument matrix is ever built.
- **Three passes in all** give two-step estimates with Windmeijer's
  correction and the Arellano–Bond tests.

The other implementations instead hold every panel's dense instrument
matrix: xtabond2, xtdpdgmm, plm, pdynmc and pydynpd.

## Performance

Two-step GMM with Windmeijer-corrected standard errors on simulated panels
of 8 periods, on a laptop (preliminary; an EC2 run will replace these):

| Panels (rows) | | dpd-stream | pydynpd | plm `pgmm` | pdynmc |
|---|---|---|---|---|---|
| 100,000 (800,000) | difference GMM | 0.8 s, 1.3 GB | 21 s, 1.0 GB | 89 s, 5.4 GB | > 600 s |
| 1,000,000 (8 million) | difference GMM | 7.3 s, 2.3 GB | 155 s, 7.9 GB | out of memory | — |
| 1,000,000 (8 million) | system GMM | 7.4 s, 2.2 GB | 162 s, 8.4 GB | > 600 s | — |

The estimates agree to about 1e-14. With uncollapsed instruments over 20
periods (5,000 panels), dpd-stream takes 0.3 s; pydynpd, plm and pdynmc
take 76, 110 and 520 s. See the benchmarks in the documentation.

## Install

```bash
pip install dpd-stream
```

## Documentation

The documentation is at https://leetucker.github.io/dpd-stream/. Build it
locally with:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`examples/` has runnable scripts, and `benchmarks/` the performance
comparisons (see `benchmarks/README.md`).

## License

MIT.
