"""The specifications checked against Stata (xtabond2 3.7.2, underid).

Each is written twice: as the xtabond2 command and as the dpd_stream call
that should reproduce it. `make_bundle.py` writes the data and a .do file
from this table; tests/test_stata.py reads the results the .do file writes
(``stata_refs.json``) and compares.

Data sets:
- ``abdata``: Stata's own (``webuse abdata``), exported by the .do file to
  ``data/abdata.csv``, so the comparison is on exactly Stata's (float) data.
- ``sim``: `dpd_stream.simulate.simulate_dynamic`, unbalanced with gaps,
  with clusters, weights and a panel-level group ``g``.
"""

YR = "yr1977 yr1978 yr1979 yr1980 yr1981 yr1982 yr1983 yr1984"
YRP = " + ".join(YR.split())

SIM = dict(n_panels=300, T=8, rho=(0.5,), unbalanced=True, p_gap=0.08, n_clusters=40,
           weights=True, heteroskedastic=True, seed=20261028)
# the same, balanced, without gaps and without common time effects. The
# Ahn-Schmidt moments need a model that has no omitted time effects (the
# specs here have no period dummies), so they are checked on these data; with
# the time effects of `sim` the moments are invalid (J rejects) and tiny
# differences in conventions are amplified
SIM_BAL = dict(SIM, unbalanced=False, p_gap=0.0, time_effects=False)
DATASETS = {"sim": SIM, "sim_bal": SIM_BAL}

SPECS = [
    # ---------------------------------------------------------------- abdata
    dict(name="ab_diff_1step_robust", data="abdata",
         stata=f"xtabond2 n L.n L2.n w L.w L(0/2).(k ys) {YR}, gmm(L.n) "
               f"iv(w L.w L(0/2).(k ys) {YR}) noleveleq robust",
         fml=f"n ~ l(n, 1:2) + w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + {YRP} "
             f"| gmm(l(n, 1)) + iv(w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + {YRP})",
         kw=dict(steps=1)),
    dict(name="ab_diff_2step_small", data="abdata",
         stata=f"xtabond2 n L.n L2.n w L.w L(0/2).(k ys) {YR}, gmm(L.(n w k)) "
               f"iv(L(0/2).ys {YR}) noleveleq twostep robust small",
         fml=f"n ~ l(n, 1:2) + w + l(w, 1) + k + l(k, 1:2) + ys + l(ys, 1:2) + {YRP} "
             f"| gmm(l(n, 1) + l(w, 1) + l(k, 1)) + iv(ys + l(ys, 1:2) + {YRP})",
         kw=dict(small=True), underid=True),
    dict(name="ab_sys_1step_small", data="abdata",
         stata=f"xtabond2 n L.n L(0/1).(w k) {YR}, gmm(L.(n w k)) iv({YR}, eq(level)) "
               "robust small",
         fml=f"n ~ l(n, 1) + w + l(w, 1) + k + l(k, 1) + {YRP} "
             f"| gmm(l(n, 1) + l(w, 1) + l(k, 1)) + iv({YRP}, eq='level')",
         kw=dict(system=True, steps=1, small=True)),
    dict(name="ab_sys_2step_collapse_fod", data="abdata",
         stata=f"xtabond2 n L.n L(0/1).(w k) {YR}, gmm(L.(n w k), collapse) iv({YR}) "
               "twostep robust orthogonal",
         fml=f"n ~ l(n, 1) + w + l(w, 1) + k + l(k, 1) + {YRP} "
             f"| gmm(l(n, 1) + l(w, 1) + l(k, 1), collapse=True) + iv({YRP})",
         kw=dict(system=True, transform="fod"), underid=True),
    dict(name="ab_sys_2step_iid", data="abdata",
         stata=f"xtabond2 n L.n w k {YR}, gmm(n, lag(2 4)) gmm(w k, lag(1 2)) iv({YR}) "
               "twostep",
         fml=f"n ~ l(n, 1) + w + k + {YRP} | gmm(n, 2:4) + gmm(w + k, 1:2) + iv({YRP})",
         kw=dict(system=True, vcov="iid")),
    dict(name="ab_diff_1step_iid_h1", data="abdata",
         stata=f"xtabond2 n L.n w k {YR}, gmm(L.n) iv(w k {YR}) noleveleq h(1) small",
         fml=f"n ~ l(n, 1) + w + k + {YRP} | gmm(l(n, 1)) + iv(w + k + {YRP})",
         kw=dict(steps=1, vcov="iid", h=1, small=True)),
    # ---------------------------------------------------------------- simulated
    dict(name="sim_diff_2step", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4)) gmm(x, lag(1 3) collapse) iv(w) "
               "noleveleq twostep robust",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(), underid=True),
    dict(name="sim_diff_fod_2step", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4)) gmm(x, lag(1 3) collapse) iv(w) "
               "noleveleq twostep robust orthogonal",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(transform="fod")),
    dict(name="sim_sys_2step", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 .)) gmm(x, lag(1 3)) iv(w) twostep robust",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 1:3) + iv(w)",
         kw=dict(system=True), underid=True),
    dict(name="sim_sys_fod_1step_iid", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4)) gmm(x, lag(1 3) collapse) iv(w) "
               "orthogonal",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(system=True, transform="fod", steps=1, vcov="iid")),
    dict(name="sim_sys_h2_1step_iid", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4)) iv(w) h(2)",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + iv(w)",
         kw=dict(system=True, steps=1, vcov="iid", h=2)),
    dict(name="sim_sys_h1_2step", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4)) gmm(x, lag(1 3)) iv(w) h(1) twostep "
               "robust",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)",
         kw=dict(system=True, h=1)),
    dict(name="sim_diff_cluster_small", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4)) gmm(x, lag(1 3) collapse) iv(w) "
               "noleveleq twostep cluster(cluster) small",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(vcov={"CRV1": "cluster"}, small=True)),
    dict(name="sim_sys_twoway", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4) collapse) gmm(x, lag(1 3) collapse) "
               "iv(w) twostep cluster(cluster year)",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4, collapse=True) "
             "+ gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(system=True, vcov={"CRV1": "cluster+year"}), nodiff=True),
    dict(name="sim_sys_options", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 3) split) gmm(x, lag(1 2) eq(diff)) "
               "iv(w, eq(level)) iv(L.w, passthru eq(diff)) iv(L2.x, mz) twostep robust",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:3, split=True) + gmm(x, 1:2, eq='diff') "
             "+ iv(w, eq='level') + iv(l(w, 1), passthru=True, eq='diff') "
             "+ iv(l(x, 2), mz=True)",
         kw=dict(system=True)),
    dict(name="sim_sys_pca", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 .)) gmm(x, lag(1 .)) iv(w) twostep robust pca",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:.) + gmm(x, 1:.) + iv(w)",
         kw=dict(system=True, pca=True)),
    dict(name="sim_sys_aweights", data="sim",
         stata="xtabond2 y L.y x w [aw=wt], gmm(y, lag(2 4)) gmm(x, lag(1 3) collapse) iv(w) "
               "twostep robust",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(system=True, weights="wt")),
    dict(name="sim_diff_pweights_1step", data="sim",
         stata="xtabond2 y L.y x w [pw=wt], gmm(y, lag(2 4)) iv(w) noleveleq",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + iv(w)",
         kw=dict(steps=1, weights="wt", weights_type="pweights")),
    dict(name="sim_sys_fweights", data="sim",
         stata="xtabond2 y L.y x w [fw=fw], gmm(y, lag(2 4)) iv(w) twostep robust",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + iv(w)",
         kw=dict(system=True, weights="fw", weights_type="fweights")),
    # weights with orthogonal deviations, in panels with gaps: the deviation
    # stored at a gap takes the panel's weight (weights are constant within
    # panels)
    dict(name="sim_diff_fod_aweights", data="sim",
         stata="xtabond2 y L.y x w [aw=wt], gmm(y, lag(2 4)) iv(w) noleveleq twostep robust "
               "orthogonal",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + iv(w)",
         kw=dict(transform="fod", weights="wt")),
    # orthogonal deviations clustered by a variable, in panels with gaps: does
    # the deviation stored at a gap (of the period before) have a cluster?
    dict(name="sim_diff_fod_cluster", data="sim",
         stata="xtabond2 y L.y x w, gmm(y, lag(2 4)) iv(w) noleveleq twostep orthogonal "
               "cluster(cluster)",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + iv(w)",
         kw=dict(transform="fod", vcov={"CRV1": "cluster"})),
    dict(name="sim_diff_two_lags_bod", data="sim",
         stata="xtabond2 y L(1/2).y x w, gmm(y, lag(1 3) orthogonal eq(diff)) iv(w) "
               "noleveleq twostep robust orthogonal",
         fml="y ~ l(y, 1:2) + x + w | gmm(y, 1:3, orthogonal=True, eq='diff') + iv(w)",
         kw=dict(transform="fod")),
    # Ahn-Schmidt moments, as xtdpdgmm. With weight(0) the first step is the
    # linear one-step estimate, and dpd-stream should match; with xtdpdgmm's
    # default first step (the nonlinear moments weighted by their inverse
    # counts) and with nl(iid), its weighting and rescaling conventions may
    # differ, so those are checked as close (within a fifth of an SE) on the
    # unbalanced data; on balanced data they are compared exactly (below)
    dict(name="sim_diff_nl_w0", data="sim",
         stata="xtdpdgmm L(0/1).y x w, model(diff) gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w) nl(noserial, weight(0)) twostep vce(robust) "
               "norescale noconstant",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(nl="noserial", nl_weight=0)),
    dict(name="sim_diff_nl", data="sim",
         stata="xtdpdgmm L(0/1).y x w, model(diff) gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w) nl(noserial) twostep vce(robust) "
               "norescale noconstant",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(nl="noserial"), close=True),
    dict(name="sim_diff_nl_1step", data="sim",
         stata="xtdpdgmm L(0/1).y x w, model(diff) gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w) nl(noserial) onestep vce(robust) "
               "norescale noconstant",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(nl="noserial", steps=1), close=True),
    dict(name="sim_diff_nl_iid", data="sim",
         stata="xtdpdgmm L(0/1).y x w, model(diff) gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w) nl(iid) twostep vce(robust) "
               "norescale noconstant",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(nl="iid"), close=True),
    # the absorbed g^year effect, as explicit dummies (dpd-stream absorbs them)
    dict(name="sim_diff_cells", data="sim",
         stata="xi i.g*i.year, noomit\nxtabond2 y L.y x w _I*, gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w _I*) noleveleq twostep robust",
         fml="y ~ l(y, 1) + x + w | g^year | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(), absorbed=True),
    dict(name="sim_diff_cells_fod", data="sim",
         stata="xi i.g*i.year, noomit\nxtabond2 y L.y x w _I*, gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w _I*) noleveleq twostep robust orthogonal",
         fml="y ~ l(y, 1) + x + w | g^year | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(transform="fod"), absorbed=True),
    dict(name="sim_diff_cells_cross", data="sim",
         stata="xi i.g*i.year, noomit\nxtabond2 y L.y x w _I*, gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w _I*) noleveleq twostep cluster(cl2)",
         fml="y ~ l(y, 1) + x + w | g^year | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(vcov={"CRV1": "cl2"}), absorbed=True),
    dict(name="sim_diff_cells_aweights", data="sim",
         stata="xi i.g*i.year, noomit\nxtabond2 y L.y x w _I* [aw=wt], gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w _I*) noleveleq twostep robust",
         fml="y ~ l(y, 1) + x + w | g^year | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(weights="wt"), absorbed=True),
    # '| year': time effects, estimated as dummies and not reported
    dict(name="sim_sys_year", data="sim",
         stata="xi i.year, noomit\nxtabond2 y L.y x w _I*, gmm(y, lag(2 4)) "
               "gmm(x, lag(1 3) collapse) iv(w _I*) twostep robust",
         fml="y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) + iv(w)",
         kw=dict(system=True), absorbed=True),
]

# the difference, system and Ahn-Schmidt fits again on balanced data
SPECS += [dict(s, name=s["name"].replace("sim_", "bal_", 1), data="sim_bal")
          for s in SPECS
          if s["name"] in ("sim_diff_2step", "sim_sys_2step", "sim_diff_nl_w0", "sim_diff_nl",
                           "sim_diff_nl_1step", "sim_diff_nl_iid")]

# Diagnostics for the Ahn-Schmidt moments against xtdpdgmm (reference only: the
# tests do not compare them; see DESIGN_NOTES.md). With wmatrix(identity) the
# one-step estimate depends on the moments alone, not on any weighting
# convention, so these separate the two. Compare with dpd-stream's moments
# under an identity weight (a script, not the public API).
_XD = ("xtdpdgmm L(0/1).y x w, model(diff) gmm(y, lag(2 4)) gmm(x, lag(1 3) collapse) iv(w) ")
SPECS += [
    dict(name="diag_lin_default_1step", data="sim_bal", ref_only=True,
         stata=_XD + "onestep vce(robust) noconstant", fml="", kw={}),
    dict(name="diag_lin_identity", data="sim_bal", ref_only=True,
         stata=_XD + "onestep wmatrix(identity) vce(robust) noconstant", fml="", kw={}),
    dict(name="diag_nl_identity", data="sim_bal", ref_only=True,
         stata=_XD + "nl(noserial) onestep wmatrix(identity) vce(robust) norescale noconstant",
         fml="", kw={}),
    dict(name="diag_nl_collapse_identity", data="sim_bal", ref_only=True,
         stata=_XD + "nl(noserial, collapse) onestep wmatrix(identity) vce(robust) "
                     "norescale noconstant", fml="", kw={}),
    # the same on the unbalanced panels (with gaps): where the nonlinear
    # moments' definition at the panels' ends is not matched
    dict(name="diag_u_lin_identity", data="sim", ref_only=True,
         stata=_XD + "onestep wmatrix(identity) vce(robust) noconstant", fml="", kw={}),
    dict(name="diag_u_nl_identity", data="sim", ref_only=True,
         stata=_XD + "nl(noserial) onestep wmatrix(identity) vce(robust) norescale noconstant",
         fml="", kw={}),
    dict(name="diag_u_nl_collapse_identity", data="sim", ref_only=True,
         stata=_XD + "nl(noserial, collapse) onestep wmatrix(identity) vce(robust) "
                     "norescale noconstant", fml="", kw={}),
    # nl(iid): the nonlinear moments (ubar_i D.u_it) and the linear
    # y_{t-2} D.u_{t-1} - y_{t-1} D.u_t it adds; with and without xtdpdgmm's
    # rescaling of them
    dict(name="diag_nl_iid_identity", data="sim_bal", ref_only=True,
         stata=_XD + "nl(iid) onestep wmatrix(identity) vce(robust) norescale noconstant",
         fml="", kw={}),
    dict(name="diag_nl_iid_rescale_identity", data="sim_bal", ref_only=True,
         stata=_XD + "nl(iid) onestep wmatrix(identity) vce(robust) noconstant",
         fml="", kw={}),
    dict(name="diag_u_nl_iid_identity", data="sim", ref_only=True,
         stata=_XD + "nl(iid) onestep wmatrix(identity) vce(robust) norescale noconstant",
         fml="", kw={}),
    dict(name="diag_u_nl_iid_rescale_identity", data="sim", ref_only=True,
         stata=_XD + "nl(iid) onestep wmatrix(identity) vce(robust) noconstant",
         fml="", kw={}),
]

# System GMM and forward orthogonal deviations, against xtdpdgmm (balanced
# data, no constant). The moment groups are written out in both programs: the
# differenced (or orthogonal) equation's GMM instruments, w in levels, and for
# system GMM the levels equation's D.y(t-1), D.x(t) and w. With wmatrix(identity)
# (reference only) the estimate depends on the moments alone; the two-step
# fits use xtdpdgmm's default weights.
_SYS = ("xtdpdgmm L(0/1).y x w, noconstant model(diff) gmm(y, lag(2 4)) "
        "gmm(x, lag(1 3) collapse) iv(w) gmm(y, lag(1 1) difference model(level)) "
        "gmm(x, lag(0 0) difference collapse model(level)) iv(w, model(level)) ")
_FOD = ("xtdpdgmm L(0/1).y x w, noconstant model(fodev) gmm(y, lag(1 3)) "
        "gmm(x, lag(0 2) collapse) iv(w) ")
SPECS += [
    dict(name="diag_sys_lin_identity", data="sim_bal", ref_only=True,
         stata=_SYS + "onestep wmatrix(identity) vce(robust) norescale", fml="", kw={}),
    dict(name="diag_sys_nl_identity", data="sim_bal", ref_only=True,
         stata=_SYS + "nl(noserial) onestep wmatrix(identity) vce(robust) norescale",
         fml="", kw={}),
    dict(name="diag_sys_nl_iid_identity", data="sim_bal", ref_only=True,
         stata=_SYS + "nl(iid) onestep wmatrix(identity) vce(robust) norescale", fml="", kw={}),
    dict(name="diag_fod_lin_identity", data="sim_bal", ref_only=True,
         stata=_FOD + "onestep wmatrix(identity) vce(robust) norescale", fml="", kw={}),
    dict(name="diag_fod_nl_identity", data="sim_bal", ref_only=True,
         stata=_FOD + "nl(noserial) onestep wmatrix(identity) vce(robust) norescale",
         fml="", kw={}),
    # default weights, two-step: compared in the tests
    dict(name="bal_sys_nl_iid", data="sim_bal",
         stata=_SYS + "nl(iid) twostep vce(robust) norescale",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) "
             "+ iv(w, passthru=True, eq='diff') + iv(w, eq='level')",
         kw=dict(system=True, constant=False, nl="iid"), se_rtol=0.25),
    dict(name="bal_sys_nl", data="sim_bal",
         stata=_SYS + "nl(noserial) twostep vce(robust) norescale",
         fml="y ~ l(y, 1) + x + w | gmm(y, 2:4) + gmm(x, 1:3, collapse=True) "
             "+ iv(w, passthru=True, eq='diff') + iv(w, eq='level')",
         kw=dict(system=True, constant=False, nl="noserial"), se_rtol=0.25),
    # reference only: xtdpdgmm's forward orthogonal deviations are scaled by
    # (T-t)/(T-t+1), not its square root (the linear identity-weight fit
    # matches to six digits only with that scale), so its estimates are not
    # xtabond2's whenever instruments are shared across periods
    dict(name="bal_fod_nl", data="sim_bal", ref_only=True,
         stata=_FOD + "nl(noserial) twostep vce(robust) norescale",
         fml="", kw={}),
]

# xtdpdgmm's iv(w) with model(diff) uses w itself (untransformed) as the
# instrument of the differenced equation; xtabond2's iv(w) uses D.w. The
# xtdpdgmm specs above ask for the former, so dpd-stream's side takes w in
# levels (passthru). (Found by the identity-weight diagnostics: with it the
# linear fit matches to six digits.) The Ahn-Schmidt (noserial) fits then match
# to six digits in the estimates, balanced or not, nl(iid) too, system GMM
# included. Standard errors: the weight matrix (e(W)) and J agree with Stata's
# to 1e-6 and 1e-7, and our Jacobian satisfies the first-order condition at
# Stata's optimum, but its reported model-based variance uses a Jacobian that
# differs by up to 3% (two-step, balanced; 7% one-step; more when unbalanced), so se_rtol is loose.
for _s in SPECS:
    if "xtdpdgmm" in _s["stata"] and not _s.get("ref_only"):
        _s["fml"] = _s["fml"].replace("iv(w)", "iv(w, passthru=True)")
        if "_nl" in _s["name"]:
            _s.pop("close", None)
            _s["se_rtol"] = 0.07 if _s["name"].startswith("bal_") else 0.25

