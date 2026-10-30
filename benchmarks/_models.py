"""The benchmarks' models, the same for every library (see external/gmm.R and
external/pydynpd_fit.py for their spelling there):

    difference   two-step difference GMM, Windmeijer-corrected: y on its first
                 lag, x and w, with year effects; y's lags 2 and deeper and
                 x's lags 1 and deeper as GMM-style instruments (one per lag
                 and period: O(T^2) of them), w as an IV-style instrument
    difference_collapsed
                 the same with collapsed instruments (one per lag): what
                 systemgmmkit's native engine supports
    system       the same as two-step system GMM, instruments collapsed

The data are dpd_stream.simulate.simulate_dynamic (rho 0.5, x predetermined).
dpd-stream's fit is timed after a warm-up fit in the same process (see
`warm_up`), so numba's compilation is not counted.
"""

from __future__ import annotations

import time

MODELS = {
    "difference": dict(system=False, collapse=False),
    "difference_collapsed": dict(system=False, collapse=True),
    "system": dict(system=True, collapse=True),
}
FML = "y ~ l(y, 1) + x + w | year | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)"


def warm_up(model):
    """Fit a tiny panel first, in the timed process, with the same options:
    numba compiles its kernels (or loads them from its on-disk cache, about
    a second) on the first call in a process, and that is not counted, as
    hdfe-stream's benchmarks do not count it. R and pydynpd have no
    compilation to warm."""
    from dpd_stream import dpd_stream
    from dpd_stream.simulate import simulate_dynamic
    dpd_stream(FML, simulate_dynamic(50, 6, seed=1), panel=("id", "year"), storage="memory",
               verbose=False, **MODELS[model])


def run_dpd(model, path, workdir):
    from dpd_stream import dpd_stream
    warm_up(model)
    t0 = time.perf_counter()
    fit = dpd_stream(FML, str(path), panel=("id", "year"), workdir=str(workdir), verbose=False,
                     **MODELS[model])
    t_fit = time.perf_counter() - t0
    return {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs), "iterations": 2,
            "converged": True, "instruments": int(fit.n_instruments),
            "coefs": {c: float(v) for c, v in fit.coef().items() if c in ("l(y, 1)", "x", "w")}}


def configs(model):
    out = {"dpd-stream": f"{model}:dpd", "pydynpd": f"{model}:pydynpd",
           "plm pgmm (R)": f"{model}:pgmm", "pdynmc (R)": f"{model}:pdynmc"}
    if MODELS[model]["collapse"]:               # its native engine has no other
        out["systemgmmkit"] = f"{model}:systemgmmkit"
    return out


def commands(path, T, external_command, external_missing):
    def command(label, kind):
        model, lib = kind.split(":", 1)
        spec = dict(path=path, T=T, **MODELS[model])
        if lib in ("pgmm", "pdynmc"):
            return external_command("R", fit=lib, **spec) or external_missing("R")
        if lib in ("pydynpd", "systemgmmkit"):
            return external_command(lib, **spec) or external_missing(lib)
        return None
    return command
