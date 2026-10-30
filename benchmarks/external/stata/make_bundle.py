"""Write the Stata reference bundle: the simulated data and a .do file
that runs every specification in specs.py with xtabond2 (and underid where
asked) and writes the results to ``stata_refs.json``.

    python benchmarks/external/stata/make_bundle.py [outdir]

Then copy the output directory to a machine with Stata 15 or later and run
``do dpd_refs.do`` there (see README.md); copy ``stata_refs.json`` and
``data/abdata.csv`` back to tests/data/ in this repository.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from specs import DATASETS, SPECS  # noqa: E402

from dpd_stream.simulate import simulate_dynamic  # noqa: E402


def sim_data(name="sim"):
    df = simulate_dynamic(**DATASETS[name])
    # cl2: a clustering that crosses the panels (and g); 120 clusters, more
    # than the instruments of any spec using it (else the moment covariance
    # is singular and the cells and dummies models' generalized inverses differ)
    return df.with_columns((pl.col("id") % 5).alias("g"),
                           (pl.col("wt") * 4).round(0).cast(pl.Int64).alias("fw"),
                           ((pl.col("id") * 7 + pl.col("year")) % 120).alias("cl2"))


def fetch_abdata(dest):
    """Download Stata's abdata for machines without internet access (best
    effort: the .do file uses webuse first)."""
    import urllib.request
    req = urllib.request.Request("https://www.stata-press.com/data/r18/abdata.dta",
                                 headers={"User-Agent": "dpd-stream bundle"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except OSError as err:
        print(f"could not download abdata.dta ({err}); copy it to {dest} yourself "
              "(Stata: webuse abdata, clear / save data/abdata.dta)")
        return False


def main(out=None):
    out = Path(out) if out else HERE / "bundle"
    (out / "data").mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        sim_data(name).write_csv(out / "data" / f"{name}.csv", float_precision=17)
    shutil.copy(HERE / "dpdsave.do", out / "dpdsave.do")
    fetch_abdata(out / "data" / "abdata.dta")
    lines = [
        "* dpd-stream: reference results from xtabond2 (written by make_bundle.py)",
        "* needs: ssc install xtabond2, xtdpdgmm, ivreg2, ranktest, underid",
        "version 15",
        "clear all",
        "set more off",
        "mata: mata set matafavor speed",
        "capture log close",
        "log using dpd_refs.log, replace text",
        "which xtabond2",
        "which underid",
        "do dpdsave.do",
        'file open fh using "stata_refs.json", write replace',
        # webuse needs the internet; fall back on the copy this script downloaded
        "capture webuse abdata, clear",
        'if _rc != 0 use "data/abdata.dta", clear',
        'export delimited using "data/abdata.csv", replace',
        "xtset id year",
        "preserve",
    ]
    current = "abdata"
    for s in SPECS:
        if s["data"] != current:
            lines += ["restore", f'import delimited using "data/{s["data"]}.csv", clear asdouble',
                      "xtset id year", "preserve"]
            current = s["data"]
        lines += ["restore, preserve", f"* {s['name']}"]
        lines += s["stata"].split("\n")
        if "svmat" not in s["stata"] and s.get("underid"):
            lines[-1] += " svmat"
        lines.append(f"dpdsave {s['name']} {1 if s.get('underid') else 0}")
    lines += ["file close fh", "log close"]
    (out / "dpd_refs.do").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/dpd_refs.do, dpdsave.do and data/{', '.join(DATASETS)}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
