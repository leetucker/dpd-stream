"""Polars' 64-bit runtime (hdfe-stream issue #5, fixed in 0.6.2): row indices
are UInt64, not UInt32, whatever the size of the frame.

`with_row_index` and the expressions that count (`pl.len`, `n_unique`,
non-averaged `rank`, `arg_sort`, `cum_count`) return polars' index type:
UInt32 on the default runtime and UInt64 with `polars-runtime-64`. hdfe-stream
coded levels with it, which broke its fits (and the row store here, which
builds on it). The runtime is chosen at install time, so the test makes those
return UInt64 on any install; with that runtime installed the patches change
nothing and the test passes all the same.
"""

import numpy as np
import polars as pl
import pytest

from dpd_stream import dpd_stream

FML = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"
ABSORBED = "y ~ l(y, 1) + x + w | g^year | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"


def _widen_index(monkeypatch):
    def widen(cls, name, keep=lambda *a, **k: True):
        original = getattr(cls, name)

        def wrapped(self, *args, _original=original, **kwargs):
            out = _original(self, *args, **kwargs)
            return out.cast(pl.UInt64) if keep(*args, **kwargs) else out
        monkeypatch.setattr(cls, name, wrapped)

    for cls in (pl.DataFrame, pl.LazyFrame):
        original = cls.with_row_index

        def with_row_index(self, name="index", offset=None, *, _original=original):
            out = _original(self, name, offset)
            return out.with_columns(pl.col(name).cast(pl.UInt64))

        monkeypatch.setattr(cls, "with_row_index", with_row_index)
    for name in ("n_unique", "arg_sort", "cum_count"):
        widen(pl.Expr, name)
    # the averaging ranks are Float64
    widen(pl.Expr, "rank", lambda method="average", *a, **k: method != "average")
    original_len = pl.len
    monkeypatch.setattr(pl, "len", lambda: original_len().cast(pl.UInt64))
    assert pl.DataFrame({"a": [1]}).with_row_index().schema["index"] == pl.UInt64
    assert pl.DataFrame({"a": [1]}).select(pl.len()).schema["len"] == pl.UInt64
    assert pl.DataFrame({"a": [1]}).select(pl.col("a").n_unique()).schema["a"] == pl.UInt64


def _fits(df):
    df = df.with_columns((pl.col("id") % 4).alias("g"))
    base = dict(panel=("id", "year"), verbose=False)
    return {
        "difference": dpd_stream(FML, df, **base),
        "system": dpd_stream(FML, df, system=True, **base),
        "clustered": dpd_stream(FML, df, vcov={"CRV1": "cluster"}, **base),
        "weighted": dpd_stream(FML, df, weights="wt", **base),
        "absorbed cells": dpd_stream(ABSORBED, df, **base),
        "nonlinear": dpd_stream(FML, df, nl="noserial", **base),
    }


@pytest.mark.parametrize("name", ["difference", "system", "clustered", "weighted",
                                  "absorbed cells", "nonlinear"])
def test_fits_with_uint64_row_index(panel_data, monkeypatch, name):
    native = _fits(panel_data)[name]
    with monkeypatch.context() as m:
        _widen_index(m)
        wide = _fits(panel_data)[name]
        assert wide.coefnames == native.coefnames
        np.testing.assert_allclose(wide.beta, native.beta, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(wide.se, native.se, rtol=1e-7)
        assert wide.hansen["statistic"] == pytest.approx(native.hansen["statistic"], rel=1e-9)
        assert wide.n_obs == native.n_obs
        # the lazy outputs read the source again
        assert wide.resid().collect().equals(native.resid().collect())
        assert wide.sample().collect().equals(native.sample().collect())


def test_parquet_source_with_uint64_row_index(panel_data, monkeypatch, tmp_path):
    path = tmp_path / "panel.parquet"
    panel_data.write_parquet(path)
    native = dpd_stream(FML, str(path), panel=("id", "year"), verbose=False)
    with monkeypatch.context() as m:
        _widen_index(m)
        wide = dpd_stream(FML, str(path), panel=("id", "year"), verbose=False)
        np.testing.assert_allclose(wide.beta, native.beta, rtol=1e-8, atol=1e-10)
