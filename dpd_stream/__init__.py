"""dpd-stream: Arellano-Bond (difference) and Blundell-Bond (system) GMM
for dynamic panels, out of core. Names and conventions follow Stata's
xtabond2 where they overlap.

The data are read once (pass 0) into blocks of whole panels sorted by
time; every later pass reads the blocks, builds each panel's rows and
instruments in a numba kernel, and accumulates the few matrices the
estimator needs (j x j and j x K, j the number of instruments). Nothing
the size of the data is held in memory, and no instrument matrix is
built.

Layout
    api.py          dpd_stream()
    _frontend.py    formula and instrument parsing; Gmm, Iv
    layout.py       the instrument matrix's columns (xtabond2's order)
    store.py        pass 0: the source -> panel-sorted blocks
    kernels/        numba: a panel's rows and instruments; the passes
    estimator.py    the fit: the passes and the estimates
    _passes.py, _absorb.py, _idtests.py, _nl.py
                    parts of the fit (mixins of estimator.DPD): the residual
                    and Windmeijer passes; absorbed effects; identification
                    tests; Ahn-Schmidt moments
    inference.py    the algebra after the passes: generalized inverse,
                    Windmeijer, Arellano-Bond, Sargan/Hansen, MMSC
    results.py      DPDResult, DPDMulti
    simulate.py     simulate_dynamic(), truth_dynamic()
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from hdfe_stream.core.multcomp import bonferroni, holm
from hdfe_stream.core.reporting import etable
from hdfe_stream.core.workspace import cleanup

from ._frontend import Gmm, Iv
from .api import dpd_stream
from .results import DPDMulti, DPDResult

try:
    __version__ = _version("dpd-stream")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "DPDMulti",
    "DPDResult",
    "Gmm",
    "Iv",
    "__version__",
    "bonferroni",
    "cleanup",
    "dpd_stream",
    "etable",
    "holm",
]
