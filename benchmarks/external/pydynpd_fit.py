"""One fit with pydynpd, for benchmarks/_harness.py, in pydynpd's own
environment (benchmarks/external/.pydynpd: it needs NumPy < 2.3 and
pandas < 3). Reads the Parquet file (timed as "read"), fits, and prints

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {...}}

    python pydynpd_fit.py '<spec as JSON>'

spec: path, T (the periods), system, collapse. The model is the benchmarks'
(see gmm.R): pydynpd's command

    y L1.y x w | gmm(y, 2:T) gmm(x, 1:T) iv(w) | timedumm [nolevel] [collapse]

two-step (its default), with Windmeijer-corrected standard errors.
"""

import contextlib
import io
import json
import sys
import time


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return 0.0


def main():
    spec = json.loads(sys.argv[1])
    import pandas as pd
    from pydynpd import regression
    baseline = hwm_mb()
    t0 = time.perf_counter()
    df = pd.read_parquet(spec["path"], columns=["id", "year", "y", "x", "w"])
    t_read = time.perf_counter() - t0
    T = int(spec["T"])
    opts = ["timedumm"]
    if not spec.get("system"):
        opts.append("nolevel")
    if spec.get("collapse"):
        opts.append("collapse")
    cmd = (f"y L1.y x w | gmm(y, 2:{T}) gmm(x, 1:{T}) iv(w) | " + " ".join(opts))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        m = regression.abond(cmd, df, ["id", "year"])
    model = m.models[0]
    table = model.regression_table
    t_fit = time.perf_counter() - t0
    b = dict(zip(table["variable"], table["coefficient"]))
    coefs = {"l(y, 1)": float(b["L1.y"]), "x": float(b["x"]), "w": float(b["w"])}
    print("RESULT " + json.dumps({
        "t_read": t_read, "t_fit": t_fit, "peak_memory_mb": hwm_mb(),
        "baseline_memory_mb": baseline, "peak_disk_mb": 0, "n_obs": int(model.num_obs),
        "iterations": 2, "converged": True, "coefs": coefs}), flush=True)


if __name__ == "__main__":
    main()
