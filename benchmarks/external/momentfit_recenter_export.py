"""Inputs for momentfit_recenter_check.R: the original sample and one bootstrap
draw, as each panel's rows of y, X and Z (from the dense reference, tests/_oracle.py),
for the cases of tests/test_momentfit.py.

    python benchmarks/external/momentfit_recenter_export.py
    Rscript benchmarks/external/momentfit_recenter_check.R     # see that script

Writes benchmarks/external/.momentfit/: <case>_<orig|draw>.parquet (columns unit, y,
x0.., z0..; the instrument columns are those nonempty in the original sample),
<case>_<orig|draw>_S1.csv (sum_i Z_i'HZ_i, the first step's weight's inverse) and
cases.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from _oracle import GMM, IV, xtabond2  # noqa: E402
from test_momentfit import CASES, panel_frame, resample  # noqa: E402
from test_oracle import _oracle_data  # noqa: E402

OUT = Path(__file__).parent / ".momentfit"


def dense(df, case):
    c = CASES[case]
    data = _oracle_data(df)
    o = xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w"],
                 [GMM(["y"], lags=(2, 4)), GMM(["x"], lags=(1, 3), collapse=True)],
                 [IV(["w"])], system=c["system"], orthogonal=c.get("orthogonal", False),
                 twostep=True, robust=True, diffsargan=False, artests=0)
    return o.dense


def frame(d, cols, unit_map=None):
    Z, X, Y = d["Z"][..., cols], d["X"], d["Y"]
    n, r, j = Z.shape
    unit = np.repeat(np.arange(n), r)
    if unit_map is not None:
        unit = unit_map[unit]
    flat = {"unit": unit, "y": Y.reshape(-1)}
    flat.update({f"x{k}": X[..., k].reshape(-1) for k in range(X.shape[2])})
    flat.update({f"z{c}": Z[..., c].reshape(-1) for c in range(j)})
    out = pl.DataFrame(flat)
    # rows with no instrument add nothing to any moment
    keep = np.abs(Z.reshape(-1, j)).sum(1) > 0
    return out.filter(pl.Series(keep)), d["S1"][np.ix_(cols, cols)]


def main():
    OUT.mkdir(exist_ok=True)
    meta = {}
    for case, c in CASES.items():
        df = panel_frame()
        d0 = dense(df, case)
        cols = np.flatnonzero(np.abs(d0["Z"]).sum((0, 1)) > 0)
        draw, n_units = resample(df, c["seed"], c.get("cluster"))
        d1 = dense(draw, case)
        if np.any(np.abs(d1["Z"][..., cols]).sum((0, 1)) == 0):
            raise SystemExit(f"{case}: an instrument column is empty in the draw; pick "
                             "another seed (a generalized inverse would be ambiguous)")
        unit_map = None
        if c.get("cluster"):
            # each panel's cluster copy, in the oracle's (sorted) panel order
            unit_map = (draw.group_by("id").agg(pl.col(c["cluster"]).first()).sort("id")
                        [c["cluster"]].to_numpy())
            orig_map = (df.group_by("id").agg(pl.col(c["cluster"]).first()).sort("id")
                        [c["cluster"]].to_numpy())
        else:
            orig_map = None
        for tag, d, m in (("orig", d0, orig_map), ("draw", d1, unit_map)):
            f, S1 = frame(d, cols, m)
            f.write_parquet(OUT / f"{case}_{tag}.parquet")
            np.savetxt(OUT / f"{case}_{tag}_S1.csv", S1, delimiter=",")
        meta[case] = dict(n_units=int(n_units), k=int(d0["X"].shape[2]), j=int(len(cols)),
                          steps=c.get("steps", 2), clustered=bool(c.get("cluster")),
                          columns=cols.tolist())
    (OUT / "cases.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(meta)} cases to {OUT}")


if __name__ == "__main__":
    main()
