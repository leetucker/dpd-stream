"""Time and peak memory as the number of panels grows, at eight periods:
two-step difference GMM (uncollapsed instruments, and collapsed) and two-step
system GMM (collapsed), with dpd-stream, pydynpd, R's plm::pgmm and pdynmc, and
systemgmmkit (the collapsed models only).

    python benchmarks/scale_benchmark.py                  # the default sizes
    python benchmarks/scale_benchmark.py 1000 10000       # panels
    python benchmarks/scale_benchmark.py --model difference --only dpd-stream

The model is in _models.py; the data are simulate_dynamic with n panels of
eight periods (n x 8 rows), written once per size. max_abs_diff compares
each fit's coefficients on y's lag, x and w with dpd-stream's. Results go to
benchmarks/results/scale.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _harness import (
    DATA_DIR,
    RESULTS_DIR,
    external_command,
    external_missing,
    run_child,
    run_configs,
    write_machine_info,
)
from _models import MODELS, commands, configs, run_dpd

SCRIPT = Path(__file__).resolve()
PERIODS = 8
# literal, for the EC2 launcher (it reads these with ast)
CONFIGS = {"difference": {"dpd-stream": "difference:dpd", "pydynpd": "difference:pydynpd",
                          "plm pgmm (R)": "difference:pgmm", "pdynmc (R)": "difference:pdynmc"},
           "difference_collapsed": {
               "dpd-stream": "difference_collapsed:dpd",
               "pydynpd": "difference_collapsed:pydynpd",
               "plm pgmm (R)": "difference_collapsed:pgmm",
               "pdynmc (R)": "difference_collapsed:pdynmc",
               "systemgmmkit": "difference_collapsed:systemgmmkit"},
           "system": {"dpd-stream": "system:dpd", "pydynpd": "system:pydynpd",
                      "plm pgmm (R)": "system:pgmm", "pdynmc (R)": "system:pdynmc",
                      "systemgmmkit": "system:systemgmmkit"}}
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
TIMEOUT_S = 1800


def data_path(n):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"scale_{n}.parquet"
    if not path.exists():
        from dpd_stream.simulate import simulate_dynamic
        simulate_dynamic(n, PERIODS, rho=(0.5,), seed=20261028).write_parquet(
            path, row_group_size=500_000)
    return path


def runner(config, n, path, workdir):
    model, lib = config.split(":", 1)
    assert lib == "dpd"
    return run_dpd(model, path, workdir)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--model", choices=list(MODELS), action="append")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "scale.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in args.sizes or DEFAULT_SIZES:
        path = data_path(n)
        for model in args.model or list(MODELS):
            run_configs(SCRIPT, out, model, n, path, configs(model),
                        command=commands(path, PERIODS, external_command, external_missing),
                        timeout=args.timeout, only=args.only, reference="dpd-stream",
                        extra=lambda _l, n=n: {"periods": PERIODS, "rows": n * PERIODS})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main()
