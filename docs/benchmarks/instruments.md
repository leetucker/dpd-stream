# Periods and instruments

Uncollapsed GMM-style instruments number about T² per variable. This suite
holds 5,000 panels and adds periods, to 40 (about 1,600 instruments).
The model is two-step difference GMM with Windmeijer-corrected standard
errors.

```{note}
**Preliminary.** These are from a laptop (Intel Core Ultra 7 258V, 8 logical
CPUs, 26 GB, WSL2), with a 600-second limit per fit, and the run stopped
early. The EC2 run will replace them, with larger sizes and a longer limit.
```

```{image} ../figures/instruments_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of periods, one line per library, log-log
```
```{image} ../figures/instruments_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of periods, one line per library, log-log
```

```{image} ../figures/instruments_memory.light.svg
:class: only-light benchmark
:alt: Peak memory against the number of periods, one line per library, log-log
```
```{image} ../figures/instruments_memory.dark.svg
:class: only-dark benchmark
:alt: Peak memory against the number of periods, one line per library, log-log
```

- **dpd-stream** never builds a panel's instrument matrix. Each row adds
  its few nonzero instruments into the moment sums, so time grows with the
  instruments per row and memory with the j × j sums. 40 periods take 7.5
  seconds and 1.5 GB.
- **The others** hold each panel's dense instrument matrix. At 20 periods
  plm needs 9 GB and 110 seconds, pdynmc 18 GB and 520 seconds, and pydynpd
  76 seconds; none was run beyond 20 periods.

The numbers are in `benchmarks/results/instruments.csv`.
