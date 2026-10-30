"""numba kernels: one panel at a time, panels in parallel.

`_build` turns one panel's rows (its base columns on the time grid) into
the rows of the stacked system -- transformed equation, then levels -- with
each row's regressors, outcome, weight, whether it is in the sample, and
its instruments as a sparse list of (column, value). The pass kernels call
it for each panel and accumulate what their pass needs; nothing the size
of the data is ever built, and no panel's instrument matrix is ever dense.

Panels are split into chunks, one per thread; each chunk has its own
scratch and accumulators, summed afterwards (`_sum0`).

Conventions (xtabond2's): the time grid is the global one, 0..T-1; the
transformed equation's row t holds the first difference at t, or the
forward orthogonal deviation of period t-1; missing instruments are 0;
regressors and outcome are multiplied by the weight.

Modules: `panel` (one panel's rows; the helpers every pass shares), `moments`
(the sums, the residual and Windmeijer passes), `cells` (absorbed effects),
`idtests` and `nl` (Ahn-Schmidt moments).
"""

from .cells import cellscore_pass, cross_pass, pass1_cells
from .idtests import idtest_pass
from .moments import pass1, resid_pass, windmeijer_pass
from .nl import nl_pass1

__all__ = [
    "cellscore_pass",
    "cross_pass",
    "idtest_pass",
    "nl_pass1",
    "pass1",
    "pass1_cells",
    "resid_pass",
    "windmeijer_pass",
]
