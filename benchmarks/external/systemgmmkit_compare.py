"""dpd-stream's results against systemgmmkit's (its native engine), on
simulated panels: how far apart the coefficients, standard errors and
statistics are, for each model. Not a benchmark of speed (see
scale_benchmark.py); an accuracy check.

    python benchmarks/external/systemgmmkit_compare.py          # prints, writes
                                                                # results/systemgmmkit_compare.csv

systemgmmkit runs in its own environment (benchmarks/external/.systemgmmkit, or
DPD_BENCH_SYSTEMGMMKIT_PYTHON), in a child process below. The model is the
benchmarks' with collapsed instruments (the only ones its native engine
builds): y on its first lag, x and w; y's lags 2 to 7 and x's 1 to 7 as
GMM-style instruments, w IV-style; two-step, Windmeijer-corrected. dpd-stream
is run as systemgmmkit's conventions need to be matched to compare anything:

- standard errors with `small=True`: systemgmmkit applies xtabond2's
  small-sample factor G/(G-1) (N-1)/(N-K) to the variance (they equal
  dpd-stream's `small=True` ones to ten digits);
- system GMM with `iv(w, eq='level')`: systemgmmkit uses IV-style instruments
  in the levels equation only (xtabond2's and dpd-stream's default is both).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("DPD_BENCH_RESULTS", HERE.parent / "results"))
PYTHON = os.environ.get("DPD_BENCH_SYSTEMGMMKIT_PYTHON",
                        str(HERE / ".systemgmmkit" / "bin" / "python"))
T = 8
# label, simulate_dynamic options, time effects as dummies
PANELS = {
    "balanced": dict(),
    "balanced, time effects": dict(time_effects=True),
    "unbalanced": dict(unbalanced=True),
    "unbalanced, gaps": dict(unbalanced=True, p_gap=0.05),
}
MODELS = [("difference", "fd", False), ("system", "fd", True),
          ("difference, orthogonal deviations", "fod", False),
          ("system, orthogonal deviations", "fod", True)]

CHILD = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
import pandas as pd
import systemgmmkit as s
from systemgmmkit import DynamicPanelSpec, GMMStyle, IVStyle
path, tdum, tr, system = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4] == "1"
df = pd.read_parquet(path).sort_values(["id", "year"])
df["L1_y"] = df.groupby("id")["y"].shift(1)
spec = DynamicPanelSpec(dependent="y", regressors=["L1_y", "x", "w"],
    gmm=[GMMStyle("y", 2, 7), GMMStyle("x", 1, 7)], iv=[IVStyle("w")], time_dummies=tdum,
    system=system, collapse=True, transformation=tr, steps="twostep")
r = s.run_native_dynamic_panel_gmm(spec, df, entity="id", time="year", windmeijer=True)
k = ["L1_y", "x", "w"]
print("RESULT " + json.dumps(dict(b=[float(r.params[i]) for i in k],
    se=[float(r.std_errors[i]) for i in k], ar1=float(r.ar1_z), ar2=float(r.ar2_z),
    hansen=float(r.hansen_j_stat), sargan=float(r.sargan_stat), L=int(r.n_instruments),
    N=int(r.nobs))))
"""


def theirs(path, tdum, tr, system):
    out = subprocess.run([PYTHON, "-c", CHILD, str(path), str(int(tdum)), tr, str(int(system))],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": (out.stderr.strip().splitlines() or ["no result"])[-1][:120]}


def ours(df, tdum, tr, system):
    from dpd_stream import dpd_stream
    fml = ("y ~ l(y, 1) + x + w" + (" | year" if tdum else "")
           + " | gmm(y, 2:7, collapse=True) + gmm(x, 1:7, collapse=True) + iv(w"
           + (", eq='level'" if system else "") + ")")
    r = dpd_stream(fml, df, panel=("id", "year"), verbose=False, system=system, transform=tr,
                   small=True)
    c = r.coef()
    k = ["l(y, 1)", "x", "w"]
    return dict(b=[float(c[i]) for i in k], se=[float(v) for v in np.asarray(r.se)[:3]],
                ar1=r.ar[0]["z"], ar2=r.ar[1]["z"], hansen=r.hansen["statistic"],
                sargan=r.sargan["statistic"], L=int(r.n_instruments), N=int(r.n_obs))


def main():
    import warnings

    from dpd_stream.simulate import simulate_dynamic
    warnings.simplefilter("ignore")
    if not Path(PYTHON).exists():
        sys.exit(f"systemgmmkit is not installed ({PYTHON}); see benchmarks/README.md")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for panel, opts in PANELS.items():
            tdum = bool(opts.get("time_effects"))
            df = simulate_dynamic(300, T, rho=(0.5,), seed=7,
                                  **{**opts, "time_effects": tdum})
            path = Path(tmp) / "panel.parquet"
            df.write_parquet(path)
            for model, tr, system in MODELS:
                a, b = ours(df, tdum, tr, system), theirs(path, tdum, tr, system)
                row = dict(panel=panel, model=model)
                if "error" in b:
                    row["error"] = b["error"]
                else:
                    row.update(
                        max_abs_diff_b=float(np.max(np.abs(np.subtract(a["b"], b["b"])))),
                        max_rel_diff_se=float(np.max(np.abs(np.divide(b["se"], a["se"]) - 1))),
                        diff_ar2_z=abs(a["ar2"] - b["ar2"]),
                        diff_hansen=abs(a["hansen"] - b["hansen"]),
                        diff_sargan=abs(a["sargan"] - b["sargan"]),
                        instruments=f"{a['L']} / {b['L']}", obs=f"{a['N']} / {b['N']}")
                rows.append(row)
                print({k: (f"{v:.2g}" if isinstance(v, float) else v) for k, v in row.items()})
    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(RESULTS / "systemgmmkit_compare.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
