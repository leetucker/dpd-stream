# Out of core

{download}`Download the script: out_of_core.py <../../examples/out_of_core.py>`

The source can be a Parquet file or glob, or a Polars LazyFrame. It is never
collected:

1. Pass 0 streams it into blocks of whole panels on disk (`storage="disk"`,
   the default), sized by `block_mb`.
2. Each later pass reads the blocks in turn.

Memory holds a block or two and the moment matrices, whatever the size of
the data.

```python
r = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)",
               "big.parquet", panel=("id", "year"), storage="disk", block_mb=64)
```

**Where the run directory goes.** The blocks live in a run directory, which
is deleted when the fit ends. Its location, in order of precedence:

1. `workdir=`;
2. `$DPD_STREAM_WORKDIR`;
3. `$HDFE_STREAM_WORKDIR`;
4. the system temporary directory.

`dpd_stream.cleanup()` removes directories left by killed processes.

**Residuals and predictions** are lazy Polars frames over the source.
`resid("fd")` gives the first-differenced residuals; `resid()` the levels
residuals, which include the panel effect. They can be sunk without
collecting:

```python
r.resid("fd").sink_parquet("resid.parquet")
```

:::{dropdown} The script
```{literalinclude} ../../examples/out_of_core.py
:language: python
```
:::
