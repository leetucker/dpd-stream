# Benchmarks

dpd-stream is compared with the other open implementations of difference and
system GMM. The data are simulated.

```bash
python benchmarks/scale_benchmark.py                 # default sizes
python benchmarks/scale_benchmark.py 1000 100000     # panels
python benchmarks/instruments_benchmark.py 10 20     # periods
python benchmarks/plot_benchmarks.py                 # docs/figures/
```

| Suite | Models | Data | Compared with |
|---|---|---|---|
| `scale_benchmark.py` | two-step difference GMM (uncollapsed, and collapsed), two-step system GMM (collapsed) | 1,000 to 10M panels of 8 periods | pydynpd, plm `pgmm` (R), pdynmc (R), systemgmmkit (collapsed models) |
| `instruments_benchmark.py` | two-step difference GMM (uncollapsed) | 5,000 panels of 5 to 40 periods | the same, but systemgmmkit: its native engine builds collapsed instruments only |

## The model

The model is the same for every library (`_models.py`):

- **Equation:** y on its first lag, x and w, with year effects.
- **Instruments:** y's lags 2 and deeper and x's lags 1 and deeper, GMM-style
  (one instrument per lag and period, or per lag when collapsed); w
  IV-style.
- **Estimator:** two-step, with Windmeijer-corrected standard errors.

**How the libraries agree.** `max_abs_diff` is the largest difference from
dpd-stream's estimates of the three coefficients.

- **Difference GMM:** the four libraries agree to about 1e-14.
- **System GMM:** pydynpd agrees with dpd-stream, which follows xtabond2.
  plm and pdynmc enter the year effects in the levels equation their own
  way, so their estimates differ by about 0.1. systemgmmkit uses IV-style
  instruments in the levels equation only, so its estimates differ by about
  0.1 too (`systemgmmkit_compare.py` matches the conventions and compares
  the statistics).
- **Collapsed difference GMM:** all five agree to about 1e-14.

## Measurement

Each configuration runs in its own process (`_harness.py`, adapted from
hdfe-stream's). Each process records:

- **Time:** wall time, split into reading the data and fitting. dpd-stream
  reads inside its fit, so its read time is zero.
- **Memory:** the peak resident memory (`VmHWM`).
- **Disk:** the peak size of its work directory. Only dpd-stream writes
  there.

**Warm-up.** dpd-stream's fit is timed after a warm-up fit of a tiny panel
in the same process, so numba's compilation and the loading of its cached
kernels (about a second) are not counted. hdfe-stream's benchmarks do the
same. R and pydynpd compile nothing.

**Failures.** A configuration that times out or runs out of memory at one
size is not run at larger ones.

**Output.** Results go to `results/<suite>.csv`, and the machine to
`results/machine.json`.

## The other libraries

They are not dependencies. Where one is missing, its configurations are
skipped.

**R** (`external/gmm.R`). Install it into a conda prefix:

```bash
micromamba create -p benchmarks/external/.r -f benchmarks/external/environment.yml
benchmarks/external/.r/bin/Rscript -e 'options(repos = "https://cloud.r-project.org"); install.packages(c("optimx", "pdynmc"))'
```

pdynmc (and optimx) are not on conda-forge; they come from CRAN.

**pydynpd** (`external/pydynpd_fit.py`) runs in an environment of its own,
because it needs NumPy older than 2.3:

```bash
uv venv -p 3.12 benchmarks/external/.pydynpd
VIRTUAL_ENV=benchmarks/external/.pydynpd uv pip install pydynpd "numpy<2.3" "pandas<3" pyarrow
```

**systemgmmkit** (`external/systemgmmkit_fit.py`) also has an environment of
its own (its pins on NumPy and pandas differ from the benchmarks'):

```bash
uv venv -p 3.12 benchmarks/external/.systemgmmkit
VIRTUAL_ENV=benchmarks/external/.systemgmmkit uv pip install systemgmmkit pyarrow
```

`python benchmarks/external/systemgmmkit_compare.py` compares its numbers with
dpd-stream's (`docs/benchmarks/systemgmmkit.md`).

**Paths.** `DPD_BENCH_RSCRIPT`, `DPD_BENCH_PYDYNPD_PYTHON` and
`DPD_BENCH_SYSTEMGMMKIT_PYTHON` point at other
installations. `DPD_BENCH_RESULTS` and `DPD_BENCH_FIGURES` send the output
elsewhere, for a trial run.

**Stata.** `external/stata/` is not a benchmark. It produces xtabond2's
results for the tests (see its README).

## On EC2

`ec2/` holds the launcher, adapted from logit-stream's. It is gitignored,
because it names the AWS account's bucket. Runs go to the same bucket under
`dpd-stream/runs/`.

```bash
uv run benchmarks/ec2/bench_ec2.py launch            # both suites, c8id.4xlarge spot
uv run benchmarks/ec2/bench_ec2.py local --dry-run   # the plan
```
