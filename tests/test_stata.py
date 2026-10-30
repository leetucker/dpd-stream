"""dpd-stream against Stata's xtabond2 (and underid), on the specifications
in benchmarks/external/stata/specs.py.

The reference values come from running that bundle in Stata (see its
README); the tests skip until ``tests/data/stata_refs.json`` and
``tests/data/abdata_stata.csv`` are there.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "external" / "stata"))

from make_bundle import sim_data  # noqa: E402
from specs import SPECS  # noqa: E402

from dpd_stream import dpd_stream  # noqa: E402

DATA = Path(__file__).parent / "data"
REFS = DATA / "stata_refs.json"


def _json_line(line):
    """One line of Stata's output: its %g format writes .5 and -.5 for 0.5 and
    -0.5 and . for a missing value, none of them JSON."""
    line = re.sub(r"(?<![\w.])(-?)\.(\d)", r"\g<1>0.\2", line)
    line = re.sub(r"(?<![\w.\"])\.(?![\w.])", "null", line)
    return json.loads(line)


def _refs():
    if not REFS.exists():
        return {}
    return {d["name"]: d for d in map(_json_line, REFS.read_text().splitlines()) if d}


@pytest.fixture(scope="module")
def datasets():
    out = {name: sim_data(name) for name in ("sim", "sim_bal")}
    if (DATA / "abdata_stata.csv").exists():
        out["abdata"] = pl.read_csv(DATA / "abdata_stata.csv")
    return out


@pytest.mark.parametrize("spec", [pytest.param(s, id=s["name"])
                                  for s in SPECS if not s.get("ref_only")])
def test_stata(spec, datasets):
    ref = _refs().get(spec["name"])
    if ref is None:
        pytest.skip("no Stata results yet (see benchmarks/external/stata/README.md)")
    if spec["data"] not in datasets:
        pytest.skip("Stata's abdata export (tests/data/abdata_stata.csv) is missing")
    r = dpd_stream(spec["fml"], datasets[spec["data"]], panel=("id", "year"), verbose=False,
                   **spec["kw"])
    b = np.array(ref["b"])
    V = np.array(ref["V"]).reshape(len(b), len(b))
    names = ref["names"]
    if spec.get("absorbed"):
        # the absorbed cells are Stata's dummies; compare the others
        k = len(r.beta)
        keep = [i for i, n in enumerate(names) if not n.startswith("_I")][:k]
        b, V, names = b[keep], V[np.ix_(keep, keep)], [names[i] for i in keep]
    else:
        assert len(b) == len(r.beta), (names, r.coefnames)
    # Time dummies in a full set are collinear in the transformed equation, and
    # which one is dropped (the others, and the constant, are then relative to
    # it) is arbitrary: Stata's xtabond2 3.7.2 drops 1977 and 1979, we drop
    # 1977 and 1984. Everything else is the same either way.
    free = [i for i, n in enumerate(names) if not re.fullmatch(r"yr\d+|_cons", n)]
    if len(free) < len(b) and ("yr" in spec["fml"] or "| year" in spec["fml"]):
        assert free == list(range(len(free)))
        b, V = b[free], V[np.ix_(free, free)]
        r_beta, r_se = r.beta[free], r.se[free]
    else:
        r_beta, r_se = r.beta, r.se
    if spec.get("close"):
        # xtdpdgmm: another first step, so close (within a fifth of a
        # standard error), not equal
        assert np.all(np.abs(r_beta - b) < 0.2 * np.sqrt(np.diag(V))), (r_beta, b)
        return
    # (xtdpdgmm's Gauss-Newton stops at about 1e-5)
    np.testing.assert_allclose(r_beta, b, rtol=3e-5 if "xtdpdgmm" in spec["stata"] else 1e-6,
                               atol=1e-9)
    np.testing.assert_allclose(r_se, np.sqrt(np.diag(V)), rtol=spec.get("se_rtol", 1e-5))
    if "xtdpdgmm" in spec["stata"]:
        # it counts the levels' observations and panels, and stores no test statistics
        return
    assert r.n_obs == pytest.approx(ref["N"])
    assert r.n_groups == ref["N_g"]
    if not spec.get("absorbed"):
        assert r.n_instruments == ref["j"]
    np.testing.assert_allclose(r.sargan["statistic"], ref["sargan"], rtol=1e-5)
    if "hansen" in ref:
        np.testing.assert_allclose(r.hansen["statistic"], ref["hansen"], rtol=1e-5)
    np.testing.assert_allclose(r.sigma2, ref["sig2"], rtol=1e-5)
    for m in (1, 2):
        if f"ar{m}" in ref:
            np.testing.assert_allclose(r.ar_test(m)["z"], ref[f"ar{m}"], rtol=1e-5)
    if "diffsargan" in ref and not spec.get("absorbed") and not spec.get("nodiff"):
        ours = sorted(round(d["excluded"], 6) for d in r.diff_hansen)
        theirs = sorted(round(row[0], 6) for row in ref["diffsargan"])
        np.testing.assert_allclose(ours, theirs, rtol=1e-5)


UNDERID = [s for s in SPECS if s.get("underid")]


@pytest.mark.xfail(strict=True, reason=(
    "Stata's underid, with xtabond2, uses ranktest's J-type KP statistic at the LIML "
    "estimates (nosvd) and, for SW, an un-partialled two-step GMM whose first step is "
    "weighted by Z'HZ (xtabond2's H); this implements the SVD form (ivreg2's) and 2SLS"))
@pytest.mark.parametrize("spec", UNDERID, ids=[s["name"] for s in UNDERID])
def test_stata_underid(spec, datasets):
    ref = _refs().get(spec["name"])
    if ref is None or "kp_lm" not in ref:
        pytest.skip("no underid results for this specification")
    r = dpd_stream(spec["fml"], datasets[spec["data"]], panel=("id", "year"), verbose=False,
                   **spec["kw"])
    np.testing.assert_allclose(r.idtests["kp_lm"]["statistic"], ref["kp_lm"], rtol=1e-5)
