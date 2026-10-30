"""Arellano and Bond (1991) and Blundell and Bond (1998), as in Roodman's
(2009) xtabond2 article, on the UK firm data (Stata's abdata).

    python examples/arellano_bond.py

The data are downloaded from Stata Press (they are not distributed with
dpd-stream; reading them needs pandas): n, w, k and ys are the logs of
employment, wages, capital and industry output. The results reproduce the
article's tables (Roodman 2009, pp. 111-118) to the digits printed.
"""

import io
import urllib.request

import pandas as pd
import polars as pl

from dpd_stream import dpd_stream

URL = "https://www.stata-press.com/data/r18/abdata.dta"
req = urllib.request.Request(URL, headers={"User-Agent": "dpd-stream example"})
with urllib.request.urlopen(req) as resp:
    raw = io.BytesIO(resp.read())
d = (pl.from_pandas(pd.read_stata(raw)).rename({"id": "firm"})
     .with_columns(pl.col("firm").cast(pl.Int64), pl.col("year").cast(pl.Int64)))
YR = " + ".join(f"yr{y}" for y in range(1976, 1985))

# ---- Arellano and Bond's Table 4, column (a1): one-step difference GMM
ab = dpd_stream("n ~ l(n, 1:2) + w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + " + YR
                + " | gmm(l(n, 1)) + iv(w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + "
                + YR + ")", d, panel=("firm", "year"), steps=1, verbose=False)
ab.summary()

# ---- the same, two-step: column (a2), with Windmeijer-corrected errors
ab2 = dpd_stream("n ~ l(n, 1:2) + w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + " + YR
                 + " | gmm(n, 2:.) + iv(w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + "
                 + YR + ")", d, panel=("firm", "year"), verbose=False)
print(ab2.tidy().head(4))

# ---- Blundell and Bond: system GMM, wages and capital endogenous
bb = dpd_stream("n ~ l(n, 1) + w + l(w, 1) + k + l(k, 1) + " + YR
                + " | gmm(l(n, 1) + l(w, 1) + l(k, 1)) + iv(" + YR + ", eq='level')",
                d, panel=("firm", "year"), system=True, steps=1, small=True, verbose=False)
bb.summary()
