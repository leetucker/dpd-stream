# Periods and instruments

Uncollapsed GMM-style instruments number about T² per variable. This suite
holds 5,000 panels and adds periods, to 40 (about 1,600 instruments).
The model is two-step difference GMM with Windmeijer-corrected standard
errors.

The run was on an EC2 spot instance (c8id.4xlarge: 16 vCPUs, 32 GiB, local
NVMe), with a 1,800-second limit per fit and the benchmarks' memory capped at
96% of it. Each fit is in its own process.

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
  instruments per row and memory with the j × j sums. 40 periods take 6
  seconds and 1.6 GB.
- **The others** hold each panel's dense instrument matrix. At 20 periods
  plm needs 9 GB and 47 seconds, pdynmc 18 GB and 441 seconds, and pydynpd
  24 seconds. At 30 periods plm and pdynmc run out of memory; pydynpd, which
  stays under 3 GB, takes 112 seconds, and 419 seconds at 40 periods.

The numbers are in `benchmarks/results/instruments.csv`.
