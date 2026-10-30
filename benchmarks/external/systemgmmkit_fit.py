"""One fit with systemgmmkit, for benchmarks/_harness.py, in its own environment
(benchmarks/external/.systemgmmkit). Reads the Parquet file (timed as "read"),
fits, and prints

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {...}}

    python systemgmmkit_fit.py '<spec as JSON>'

spec: path, T (the periods), system, collapse. The model is the benchmarks'
(see gmm.R), through systemgmmkit's native engine: first differences, two-step,
Windmeijer-corrected, y's lags 2 to T-1 and x's 1 to T-1 as GMM-style
instruments, w IV-style, time dummies. Its native engine builds collapsed
instruments only (a spec with collapse=False gets the same ones), so the
benchmarks run it on the collapsed models alone.
"""

import json
import sys
import time
import warnings


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return 0.0


def main():
    spec = json.loads(sys.argv[1])
    warnings.simplefilter("ignore")
    import pandas as pd
    from systemgmmkit import DynamicPanelSpec, GMMStyle, IVStyle, run_native_dynamic_panel_gmm
    baseline = hwm_mb()
    t0 = time.perf_counter()
    df = pd.read_parquet(spec["path"], columns=["id", "year", "y", "x", "w"])
    t_read = time.perf_counter() - t0
    T = int(spec["T"])
    t0 = time.perf_counter()
    df = df.sort_values(["id", "year"])
    df["L1_y"] = df.groupby("id")["y"].shift(1)
    model = DynamicPanelSpec(
        dependent="y", regressors=["L1_y", "x", "w"],
        gmm=[GMMStyle("y", 2, T - 1), GMMStyle("x", 1, T - 1)], iv=[IVStyle("w")],
        time_dummies=True, system=bool(spec.get("system")), collapse=True,
        transformation="fd", steps="twostep")
    fit = run_native_dynamic_panel_gmm(model, df, entity="id", time="year", windmeijer=True)
    t_fit = time.perf_counter() - t0
    b = fit.params
    coefs = {"l(y, 1)": float(b["L1_y"]), "x": float(b["x"]), "w": float(b["w"])}
    print("RESULT " + json.dumps({
        "t_read": t_read, "t_fit": t_fit, "peak_memory_mb": hwm_mb(),
        "baseline_memory_mb": baseline, "peak_disk_mb": 0, "n_obs": int(fit.nobs),
        "instruments": int(fit.n_instruments), "iterations": 2, "converged": True,
        "coefs": coefs}), flush=True)


if __name__ == "__main__":
    main()
