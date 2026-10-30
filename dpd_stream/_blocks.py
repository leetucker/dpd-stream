"""Iteration over the stored blocks, and the split of a block's panels into the
chunks the kernels run in parallel."""

from __future__ import annotations

import numpy as np


def _chunks(pstart, nch):
    """Chunk boundaries over a block's panels, balanced by rows."""
    npan = len(pstart) - 1
    if npan <= 0:
        return np.zeros(1, np.int64)
    nch = max(1, min(nch, npan))
    targets = np.linspace(0, pstart[-1], nch + 1)
    b = np.searchsorted(pstart, targets[1:-1])
    out = np.unique(np.concatenate([[0], b, [npan]])).astype(np.int64)
    return out


def _blocks(store, fields=("X", "t", "pid", "cl", "w", "ag")):
    for blk in store.blocks.iterate(fields=list(fields)):
        pid = blk["pid"]
        starts = np.flatnonzero(np.diff(pid)) + 1
        pstart = np.concatenate([[0], starts, [len(pid)]]).astype(np.int64)
        yield blk, pstart
