"""Time and peak memory as the number of periods grows, at 5,000 panels:
uncollapsed GMM-style instruments number O(T^2) per variable (T = 40 gives
about 1,600), which is what the implementations that build each panel's
instrument matrix struggle with. Two-step difference GMM only.

    python benchmarks/instruments_benchmark.py                  # the default periods
    python benchmarks/instruments_benchmark.py 10 20            # periods

The model is in _models.py; the data are simulate_dynamic with 5,000 panels
of T periods. Results go to benchmarks/results/instruments.csv.
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
PANELS = 5_000
CONFIGS = {"difference": {"dpd-stream": "difference:dpd", "pydynpd": "difference:pydynpd",
                          "plm pgmm (R)": "difference:pgmm", "pdynmc (R)": "difference:pdynmc"},
           "system": {"dpd-stream": "system:dpd", "pydynpd": "system:pydynpd",
                      "plm pgmm (R)": "system:pgmm", "pdynmc (R)": "system:pdynmc",
                      "systemgmmkit": "system:systemgmmkit"}}
DEFAULT_SIZES = [5, 10, 20, 30, 40]
TIMEOUT_S = 1800


def data_path(n):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"periods_{n}.parquet"
    if not path.exists():
        from dpd_stream.simulate import simulate_dynamic
        simulate_dynamic(PANELS, n, rho=(0.5,), seed=20261028).write_parquet(path)
    return path


def runner(config, n, path, workdir):
    model, lib = config.split(":", 1)
    assert lib == "dpd"
    return run_dpd(model, path, workdir)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int, help="periods")
    ap.add_argument("--model", choices=list(MODELS), action="append")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "instruments.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in args.sizes or DEFAULT_SIZES:
        path = data_path(n)
        for model in args.model or ["difference"]:
            run_configs(SCRIPT, out, model, n, path, configs(model),
                        command=commands(path, n, external_command, external_missing),
                        timeout=args.timeout, only=args.only, reference="dpd-stream",
                        extra=lambda _l, n=n: {"periods": n, "rows": PANELS * n})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main()
