# Panels

Time and peak memory as the number of panels grows, at 8 periods each, for
three models:

- **Difference GMM:** two-step, uncollapsed instruments (lags 2 to 8 of y,
  1 to 8 of x), w IV-style, year effects, Windmeijer-corrected standard
  errors: 55 instruments.
- **Collapsed difference GMM:** the same with the instruments collapsed, the
  only kind systemgmmkit's native engine builds.
- **System GMM:** the collapsed model, with the levels equation.

Every library gets the same model (`benchmarks/_models.py`). dpd-stream also
computes, as by default, the identification and difference-in-Hansen tests,
which the others do not.

The run was on an EC2 spot instance (c8id.4xlarge: 16 vCPUs, 32 GiB, local
NVMe), with a 1,800-second limit per fit and the benchmarks' memory capped at
96% of it. Each fit is in its own process; dpd-stream's is timed after a
warm-up fit that loads numba's kernels. Figures show every fit that
finished; the others are listed below each one.

```{image} ../figures/scale_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of panels, one line per library, log-log, three panels
```
```{image} ../figures/scale_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of panels, one line per library, log-log, three panels
```

- **dpd-stream** fits a million panels (8 million rows) in 4.6 to 5.7
  seconds, and ten million panels (80 million rows) in 74 to 84 seconds.
- **pydynpd** takes 120 to 159 seconds at a million panels, and runs out of
  memory at ten million.
- **plm** takes 563 and 799 seconds at a million panels in the collapsed
  models, and runs out of memory there with uncollapsed instruments. It runs
  out of memory at ten million.
- **pdynmc** takes 280 to 340 seconds at 10,000 panels and times out at
  100,000.
- **systemgmmkit** takes 475 to 500 seconds at 100,000 panels. At a million
  panels its panel-time-grid check refuses the data (see
  [differences between implementations](../implementations.md)).

```{image} ../figures/scale_memory.light.svg
:class: only-light benchmark
:alt: Peak memory against the number of panels, one line per library, log-log, three panels
```
```{image} ../figures/scale_memory.dark.svg
:class: only-dark benchmark
:alt: Peak memory against the number of panels, one line per library, log-log, three panels
```

dpd-stream's memory is mostly fixed: about 0.5 GB of libraries and numba's
compiled kernels. It reaches 1.6 to 1.7 GB at a million panels, where pydynpd
needs 6.2 to 8.2 GB and plm up to 26 GB, and 9 to 10 GB at ten million. The
data themselves go to disk, in blocks of whole panels (10 GB at ten million
panels):

```{image} ../figures/scale_disk.light.svg
:class: only-light benchmark
:alt: Peak disk use of dpd-stream against the number of panels, log-log
```
```{image} ../figures/scale_disk.dark.svg
:class: only-dark benchmark
:alt: Peak disk use of dpd-stream against the number of panels, log-log
```

**Agreement.** In difference GMM, collapsed or not, all libraries that
finished agree with dpd-stream to 3e-13 or better. In system GMM pydynpd agrees
with dpd-stream (which follows xtabond2) to 3e-14. plm and pdynmc enter the
year effects in the levels equation their own way, and systemgmmkit uses
IV-style instruments in the levels equation only, so their estimates differ by
0.1 to 0.14.

The numbers are in `benchmarks/results/scale.csv`.
