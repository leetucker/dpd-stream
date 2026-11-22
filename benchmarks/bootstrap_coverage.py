"""Monte Carlo study of the refit bootstrap: interval coverage and the Hansen test's size
and power, analytic against bootstrap.

    python benchmarks/bootstrap_coverage.py                       # 300 samples, 199 draws
    python benchmarks/bootstrap_coverage.py --samples 50 --reps 99 --processes 4
    python benchmarks/bootstrap_coverage.py --collapse                  # few instruments

Each sample is `simulate_dynamic(panels, periods)` with no time effects (so the moment
conditions hold), fitted by difference GMM with every available lag of y and x as
instruments, which makes the instruments many relative to the panels (with --collapse,
one instrument per lag instead, which makes them few). (Difference GMM,
because the simulator's w is correlated with the fixed effect, so that w is not a valid
instrument for system GMM's levels equation.) For every sample and
specification, the fit and its recentered refit bootstrap give:

- 95% intervals for each coefficient: analytic (Windmeijer-corrected two-step, or the
  uncorrected variance), percentile, studentized and symmetric studentized;
- the Hansen test at 5%: asymptotic and bootstrap.

The "invalid" specification adds x's current value to the instruments, which is
invalid because x is predetermined (it depends on the previous period's shock), so its Hansen
rejection rate is the test's power. Results go to benchmarks/results/bootstrap_coverage.csv
(one row per sample, specification and coefficient) and a summary is printed.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

SPECS = {
    # name: (formula, steps)
    "two-step": ("y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)", 2),
    "iterated": ("y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)", "iterated"),
    "invalid": ("y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 0:0) + gmm(x, 1:.) + iv(w)", 2),
}


def one_sample(arg):
    seed, panels, periods, reps, collapse = arg
    from dpd_stream import dpd_stream
    from dpd_stream.simulate import simulate_dynamic, truth_dynamic

    df = simulate_dynamic(panels, periods, time_effects=False, seed=seed)
    truth = truth_dynamic()
    rows = []
    for name, (fml, steps) in SPECS.items():
        fit = dpd_stream(fml, df, panel=("id", "year"), steps=steps, collapse=collapse,
                         verbose=False, n_threads=1)
        boot = fit.refit_bootstrap(reps=reps, seed=seed, n_threads=1)
        complete = not boot._failures
        z = stats.norm.ppf(0.975)
        se = np.sqrt(np.diag(fit.vcov))
        intervals = {"analytic": (fit.beta - z * se, fit.beta + z * se)}
        for method in ("percentile", "studentized", "symmetric"):
            tab = boot.confint(method=method, allow_incomplete=True)
            intervals[method] = (tab["Lower"].to_numpy(), tab["Upper"].to_numpy())
        h = boot.hansen(allow_incomplete=True)
        for i, coef in enumerate(fit.coefnames):
            if coef not in truth:
                continue
            row = dict(seed=seed, spec=name, coefficient=coef, estimate=fit.beta[i],
                       analytic_se=se[i], bootstrap_se=boot.standard_errors(
                           allow_incomplete=True)[i],
                       hansen_p=h["p"], hansen_bootstrap_p=h["bootstrap_p"],
                       failures=len(boot._failures), complete=complete)
            for method, (lo, hi) in intervals.items():
                row[f"covers_{method}"] = bool(lo[i] <= truth[coef] <= hi[i])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--reps", type=int, default=199)
    ap.add_argument("--panels", type=int, default=200)
    ap.add_argument("--periods", type=int, default=8)
    ap.add_argument("--collapse", action="store_true")
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or str(Path(__file__).parent / "results" / (
        "bootstrap_coverage" + ("_collapsed" if a.collapse else "") + ".csv"))
    args = [(1000 + s, a.panels, a.periods, a.reps, a.collapse) for s in range(a.samples)]
    rows = []
    with ProcessPoolExecutor(a.processes) as pool:
        for done, sample_rows in enumerate(pool.map(one_sample, args), 1):
            rows.extend(sample_rows)
            if done % 10 == 0:
                print(f"{done} of {a.samples} samples", flush=True)
    res = pl.DataFrame(rows)
    res.write_csv(out)
    cover = [c for c in res.columns if c.startswith("covers_")]
    print(f"\n{a.samples} samples of {a.panels} panels x {a.periods} periods, "
          f"{a.reps} bootstrap draws{', collapsed instruments' if a.collapse else ''}\n")
    print("95% interval coverage")
    with pl.Config(tbl_rows=50, tbl_cols=20, float_precision=3):
        print(res.group_by("spec", "coefficient", maintain_order=True).agg(
            [pl.col(c).mean().alias(c.removeprefix("covers_")) for c in cover]
            + [pl.col("estimate").std().alias("sd_estimate"),
               pl.col("analytic_se").mean().alias("mean_analytic_se"),
               pl.col("bootstrap_se").mean().alias("mean_bootstrap_se")]))
        print("\nHansen test, rejection rate at 5%")
        print(res.unique(["seed", "spec"]).group_by("spec", maintain_order=True).agg(
            (pl.col("hansen_p") < 0.05).mean().alias("asymptotic"),
            (pl.col("hansen_bootstrap_p") < 0.05).mean().alias("bootstrap"),
            (pl.col("failures") > 0).mean().alias("share_with_failed_draws")))


if __name__ == "__main__":
    main()
