"""The bootstrap's refits against the dense reference (`_oracle.recentered`): on
resamples of whole panels (or clusters), the recentered estimates, their
Windmeijer-corrected variance and the Hansen statistic are those of Hall and
Horowitz's (1996) recentered GMM, to roundoff."""

import numpy as np
import polars as pl
import pytest
from _oracle import GMM, IV, recentered, xtabond2
from test_oracle import _oracle_data

from dpd_stream.estimator import DPD

FML = "y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)"
OGMM = [GMM(["y"], lags=(2, 4)), GMM(["x"], lags=(1, 3), collapse=True)]
TOL = 1e-8


def _fit(df, **kw):
    est = DPD(FML, df, ("id", "year"), verbose=False, **kw)
    return est, est.fit()


def _oracle(df, system, orthogonal, cluster=None, small=False):
    data = _oracle_data(df)
    return xtabond2(data["id"], data["year"], data, "y", ["l(y, 1)", "x", "w"], OGMM,
                    [IV(["w"])], system=system, orthogonal=orthogonal, twostep=True,
                    robust=True, cluster=[cluster] if cluster else (), small=small,
                    diffsargan=False, artests=0)


def _resample(df, seed, key="id"):
    """A resample of whole `key` groups with replacement; each copy's panels get
    IDs of their own (and with clusters, each copy a cluster code of its own)."""
    groups = np.sort(df[key].unique().to_numpy())
    draw = np.random.default_rng(seed).integers(0, len(groups), len(groups))
    parts = []
    for copy, g in enumerate(draw):
        part = df.filter(pl.col(key) == groups[g])
        part = part.with_columns((pl.col("id") + copy * 100_000).alias("id"))
        if key != "id":
            part = part.with_columns(pl.lit(copy).alias(key))
        parts.append(part)
    return pl.concat(parts), len(groups)


CASES = {
    # name: (fit options, oracle system, orthogonal)
    "diff 2step": (dict(system=False), False, False),
    "sys 2step": (dict(system=True), True, False),
    "sys fod 2step": (dict(system=True, transform="fod"), True, True),
    "diff 1step robust": (dict(system=False, steps=1), False, False),
    "sys iterated": (dict(system=True, steps="iterated"), True, False),
    "diff small": (dict(system=False, small=True), False, False),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("seed", [1, 2])
def test_recentered_draw_matches_oracle(panel_data, case, seed):
    kw, system, orth = CASES[case]
    df = panel_data.select("id", "year", "y", "x", "w")
    est, f = _fit(df, **kw)
    o = _oracle(df, system, orth)
    # the shift: the original fit's moment sum, which the oracle agrees with
    Z, X, Y = (o.dense[k] for k in ("Z", "X", "Y"))
    ghat = np.einsum("nri,nr->i", Z, Y) - np.einsum("nri,nrk->ik", Z, X) @ f.info["b"]
    msum, live = f.info["moment_sum"], f.info["live_mask"]
    assert len(msum) == len(ghat)
    np.testing.assert_allclose(msum[live], ghat[live], rtol=1e-7, atol=1e-8)
    assert np.all(ghat[~live] == 0) and abs(msum).max() > 1e-3

    draw, n = _resample(df, seed)
    _, fd = _fit(draw, estimates_only=True, draw_se=True, moment_shift=(msum, n),
                 fixed_columns=live, fixed_xsh=f.info["xsh"], tdelta=est.store.tdelta, **kw)
    od = _oracle(draw, system, orth)
    steps = kw.get("steps", 2)
    b1, b, V, hansen = recentered(od, ghat, n, steps=steps)
    assert fd.names == f.names
    np.testing.assert_allclose(fd.info["b1"], b1, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(fd.b, b, rtol=TOL, atol=TOL)
    vfac = fd.info["vfac"]["other"]
    np.testing.assert_allclose(fd.V, V * vfac, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(fd.info["hansen"], hansen, rtol=1e-7)


@pytest.mark.parametrize("small", [False, True])
def test_recentered_cluster_draw_matches_oracle(panel_data, small):
    """Whole clusters drawn, each copy a cluster of its own; the moments are
    recentered by cluster, ghat / G."""
    df = panel_data.select("id", "year", "y", "x", "w", "cluster")
    kw = dict(system=True, vcov={"CRV1": "cluster"}, small=small)
    est, f = _fit(df, **kw)
    o = _oracle(df, True, False, cluster="cluster", small=small)
    Z, X, Y = (o.dense[k] for k in ("Z", "X", "Y"))
    ghat = np.einsum("nri,nr->i", Z, Y) - np.einsum("nri,nrk->ik", Z, X) @ f.info["b"]
    msum, live = f.info["moment_sum"], f.info["live_mask"]
    np.testing.assert_allclose(msum[live], ghat[live], rtol=1e-7, atol=1e-8)

    draw, G = _resample(df, 5, key="cluster")
    _, fd = _fit(draw, estimates_only=True, draw_se=True, moment_shift=(msum, G),
                 fixed_columns=live, fixed_xsh=f.info["xsh"], tdelta=est.store.tdelta, **kw)
    od = _oracle(draw, True, False, cluster="cluster", small=small)
    # each panel's cluster code, in the oracle's (sorted) panel order
    pc = draw.group_by("id").agg(pl.col("cluster").first()).sort("id")["cluster"].to_numpy()
    b1, b, V, hansen = recentered(od, ghat, G, steps=2, groups=pc, small=small)
    np.testing.assert_allclose(fd.b, b, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(fd.V, V * fd.info["vfac"]["other"], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(fd.info["hansen"], hansen, rtol=1e-7)


@pytest.mark.parametrize("kw", [dict(system=False), dict(system=True, steps="iterated"),
                                dict(system=True, steps=1),
                                dict(system=True, vcov={"CRV1": "cluster"})])
def test_estimates_only_gives_the_same_estimates(panel_data, kw):
    """Without a shift, a refit's estimates are exactly a full fit's."""
    _, full = _fit(panel_data, **kw)
    _, quick = _fit(panel_data, estimates_only=True, **kw)
    _, fixed = _fit(panel_data, estimates_only=True, draw_se=True,
                    fixed_columns=full.info["live_mask"], fixed_xsh=full.info["xsh"], **kw)
    assert np.array_equal(quick.b, full.b) and quick.names == full.names
    np.testing.assert_allclose(fixed.b, full.b, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fixed.V, full.V, rtol=1e-10, atol=1e-14)
    assert quick.info["ar"] == [] and quick.info["diffsargan"] == []


def test_exactly_identified_shift_is_zero(panel_data):
    df = panel_data.select("id", "year", "y", "x", "w")
    est = DPD("y ~ l(y, 1) + x | iv(l(y, 2) + l(x, 1))", df, ("id", "year"), verbose=False)
    f = est.fit()
    np.testing.assert_allclose(f.info["moment_sum"], 0, atol=1e-8)
