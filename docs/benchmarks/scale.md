# Panels

Time and peak memory as the number of panels grows, at 8 periods each:

- **Difference GMM:** two-step, uncollapsed instruments (lags 2 to 8 of y,
  1 to 8 of x), w IV-style, year effects, Windmeijer-corrected standard
  errors: 55 instruments.
- **System GMM:** the same, with the instruments collapsed.

Every library gets the same model (`benchmarks/_models.py`). dpd-stream also
computes, as by default, the identification and difference-in-Hansen tests,
which the others do not.

```{note}
**Preliminary.** These are from a laptop (Intel Core Ultra 7 258V, 8 logical
CPUs, 26 GB, WSL2), with a 600-second limit per fit, and the run stopped
early. The EC2 run will replace them, with larger sizes and a longer limit.
```

```{image} ../figures/scale_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of panels, one line per library, log-log
```
```{image} ../figures/scale_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of panels, one line per library, log-log
```

- **dpd-stream** fits a million panels (8 million rows) in about 7 seconds.
- **pydynpd** takes 155–162 seconds there.
- **plm** runs out of memory (difference GMM) or time (system GMM) at a
  million panels.
- **pdynmc** times out at 100,000.

```{image} ../figures/scale_memory.light.svg
:class: only-light benchmark
:alt: Peak memory against the number of panels, one line per library, log-log
```
```{image} ../figures/scale_memory.dark.svg
:class: only-dark benchmark
:alt: Peak memory against the number of panels, one line per library, log-log
```

dpd-stream's memory is mostly fixed: about 0.6 GB of libraries and numba's
compiled kernels. It reaches 2.3 GB at a million panels, where pydynpd needs
7.9–8.4 GB. The data themselves go to disk, in blocks of whole panels:

```{image} ../figures/scale_disk.light.svg
:class: only-light benchmark
:alt: Peak disk use of dpd-stream against the number of panels, log-log
```
```{image} ../figures/scale_disk.dark.svg
:class: only-dark benchmark
:alt: Peak disk use of dpd-stream against the number of panels, log-log
```

**Agreement.** In difference GMM all four libraries agree to about 1e-14.
In system GMM pydynpd agrees with dpd-stream (which follows xtabond2) to
1e-14. plm and pdynmc enter the year effects in the levels equation their
own way, so their estimates differ by about 0.13.

The numbers are in `benchmarks/results/scale.csv`.
