"""dpd-stream's side of the identity-weight diagnostics (specs ``diag_*`` in
specs.py): the one-step estimates of the linear and the Ahn-Schmidt moments
under an identity weight, from the fit's own sums, to compare with the
``stata_refs.json`` of xtdpdgmm's ``wmatrix(identity)`` fits.

    python benchmarks/external/stata/diag_identity.py [path/to/stata_refs.json]

With an identity weight the estimate depends on the moments alone, so a
difference from xtdpdgmm's is a difference in the moments, not the weighting.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "tests"))
warnings.simplefilter("ignore")

from make_bundle import sim_data  # noqa: E402

from dpd_stream import dpd_stream  # noqa: E402
from dpd_stream import inference as inf  # noqa: E402

FML = ("y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) "
       "+ iv(w, passthru=True)")  # xtdpdgmm: w in levels


def refs(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            line = re.sub(r"(?<![\w.])(-?)\.(\d)", r"\g<1>0.\g<2>", line)
            d = json.loads(line)
            out[d["name"]] = d
    return out


def our_moments(nl, data="sim_bal", kind="noserial"):
    """The fit's Moments (all moments) and one-step objects."""
    cap = {}
    orig = inf.Moments.solve

    def solve(self, A, b0, cols=None, **kw):
        cap.setdefault("mom", self)
        return orig(self, A, b0, cols=cols, **kw)
    inf.Moments.solve = solve
    try:
        r = dpd_stream(FML, sim_data(data), panel=("id", "year"), verbose=False,
                       nl=kind if nl else None, steps=1)
    finally:
        inf.Moments.solve = orig
    return r, cap.get("mom")


def one(data, prefix, ref):
    r_lin, _ = our_moments(False, data)
    print(f"[{data}] dpd-stream linear one-step (Z'HZ weight):", np.round(r_lin.beta, 6))
    _, mom = our_moments(True, data)
    j = mom.j
    lin = inf.Moments(mom.ZX, mom.ZY)
    b_id, _ = lin.solve(np.eye(j), r_lin.beta)
    print(f"[{data}] dpd-stream linear, identity weight      :", np.round(b_id, 6))
    # nonlinear moments (T-3 of them) with identity weight
    b_nl, ok = mom.solve(np.eye(mom.L), b_id)
    print(f"[{data}] dpd-stream nl(noserial), identity weight:", np.round(b_nl, 6), "ok", ok)
    # collapsed nonlinear moments: the sum of the moments
    mc = inf.Moments.__new__(inf.Moments)
    mc.ZX, mc.ZY, mc.j, mc.nl = mom.ZX, mom.ZY, j, True
    mc.s0, mc.s1, mc.s2 = (a.sum(0, keepdims=True) for a in (mom.s0, mom.s1, mom.s2))
    mc.L = j + 1
    b_c, ok = mc.solve(np.eye(j + 1), b_id)
    print(f"[{data}] dpd-stream nl collapsed, identity weight:", np.round(b_c, 6), "ok", ok)
    # nl(iid): the nonlinear and the added linear moments
    _, mi = our_moments(True, data, "iid")
    b_iid, ok = mi.solve(np.eye(mi.L), b_id)
    print(f"[{data}] dpd-stream nl(iid), identity weight     :", np.round(b_iid, 6),
          f"({mi.L - mi.j} added moments) ok", ok)
    for name in (f"diag_{prefix}lin_identity", f"diag_{prefix}nl_identity",
                 f"diag_{prefix}nl_collapse_identity", f"diag_{prefix}nl_iid_identity",
                 f"diag_{prefix}nl_iid_rescale_identity"):
        if name in ref:
            print(f"[{data}] Stata {name:30s}:", np.round(ref[name]["b"], 6),
                  "" if "zrank_nl" not in ref[name] else f"(nonlinear moments: "
                  f"{ref[name]['zrank_nl']})")


SYS_FML = ("y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) "
           "+ iv(w, passthru=True, eq='diff') + iv(w, eq='level')")
FOD_FML = FML


def identity_fits(fml, kw, kinds):
    """Our identity-weight one-step estimates: linear moments, then each nl kind
    (all from the sums of an nl fit, whose linear moments are the plain ones)."""
    out = {}
    for kind in kinds:
        cap = {}
        orig = inf.Moments.solve

        def solve(self, A, b0, cols=None, _cap=cap, _orig=orig, **k):
            _cap.setdefault("mom", self)
            return _orig(self, A, b0, cols=cols, **k)
        inf.Moments.solve = solve
        try:
            r = dpd_stream(fml, sim_data("sim_bal"), panel=("id", "year"), verbose=False,
                           nl=kind, steps=1, **kw)
        finally:
            inf.Moments.solve = orig
        mom = cap["mom"]
        if "linear" not in out:
            lin = inf.Moments(mom.ZX, mom.ZY)
            out["linear"], _ = lin.solve(np.eye(mom.j), r.beta)
        out[kind], _ = mom.solve(np.eye(mom.L), out["linear"])
    return out


def more(ref):
    """System GMM and forward orthogonal deviations (balanced data)."""
    for label, fml, kw, kinds, names in (
            ("system", SYS_FML, dict(system=True, constant=False), ("noserial", "iid"),
             ("diag_sys_lin_identity", "diag_sys_nl_identity", "diag_sys_nl_iid_identity")),
            ("fod", FOD_FML, dict(transform="fod"), ("noserial",),
             ("diag_fod_lin_identity", "diag_fod_nl_identity"))):
        fits = identity_fits(fml, kw, kinds)
        ours = [fits["linear"]] + [fits[k] for k in kinds]
        for n, b in zip(names, ours):
            print(f"[{label}] dpd-stream {n:28s}:", np.round(b, 6))
            if n in ref:
                nm = f"{ref[n].get('zrank')} linear + {ref[n].get('zrank_nl')} nonlinear"
                print(f"[{label}] Stata      {n:28s}:", np.round(ref[n]["b"], 6), f"({nm})")


def main(path=None):
    ref = refs(path) if path else {}
    one("sim_bal", "", ref)
    one("sim", "u_", ref)
    more(ref)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
