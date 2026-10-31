"""The estimator, `dpd_stream`."""

from __future__ import annotations

from hdfe_stream.core.profile import Profile
from hdfe_stream.core.typing import PathLike, Source
from scipy import stats

from ._frontend import Gmm, Iv
from .estimator import DPD
from .results import DPDResult


def dpd_stream(fml: str, data: Source, panel: tuple, *, system: bool = False,
               steps: int | str = 2, transform: str = "fd", vcov="robust",
               gmm: list[Gmm] | None = None, iv: list[Iv] | None = None,
               collapse: bool = False, h: int | None = None, small: bool = False,
               artests: int = 2, constant: bool = True, pca: bool = False,
               components: int = 0, weights: str | None = None,
               weights_type: str = "aweights", tdelta: int | None = None,
               nl: str | None = None, nl_weight: float = 1.0,
               workdir: PathLike | None = None, profile: bool = False,
               **options) -> DPDResult | Profile:
    """Arellano-Bond (difference) or Blundell-Bond (system) GMM for dynamic
    panels, out of core. Option names and conventions follow Stata's xtabond2
    where they overlap.

    Parameters
    ----------
    fml : str
        ``"y ~ regressors | time effects | instruments"``, the last two
        optional. Lags use fixest's operators: ``l(x, 1)``, ``l(y, 1:2)``,
        ``f(x, 1)``, ``d(x)``. The instrument part is ``gmm()`` and ``iv()``
        terms joined by ``+``: ``gmm(vars, a:b, collapse=True, eq='diff',
        passthru=True, orthogonal=True, split=True)`` (lags a to b of the
        variables, ``.`` for no upper limit; default ``1:.``, as xtabond2) and
        ``iv(vars, eq='level', passthru=True, mz=True)``. Several variables
        in one term are joined by ``+``: ``gmm(n + w, 2:.)``.
        Time effects (``| year``) are absorbed: estimated as a dummy per
        period, each also an IV-style instrument (xtabond2's ``yr*`` in the
        regressors and in ``iv()``), and not reported. In difference GMM,
        ``| g^year`` (g constant within panels: region, industry) absorbs a
        period effect for each value of g, partialled out of the moments,
        with the same estimates as its dummies would give.
    data : str, Path, polars LazyFrame or DataFrame
        A Parquet file (or glob), or a Polars frame. It is read once.
    panel : (str, str)
        The panel identifier and the time variable (integer periods).
    system : bool, default False
        System GMM (Blundell and Bond 1998): the levels equation is added,
        with a constant. False is difference GMM (xtabond2's ``noleveleq``).
    steps : {1, 2, "iterated"}, default 2
        One-step, two-step, or iterated GMM (the weight matrix updated until
        the estimates converge, as xtdpdgmm's ``igmm``).
    transform : {"fd", "fod"}, default "fd"
        First differences, or forward orthogonal deviations (xtabond2's
        ``orthogonal``), which keep more observations in unbalanced panels.
    vcov : default "robust"
        ``"robust"``: clustered by panel, and for two-step with Windmeijer's
        (2005) correction. ``"iid"``: the non-robust one-step variance (for
        two-step, the uncorrected one). ``{"CRV1": "var"}`` or
        ``{"CRV1": "var1+var2"}``: clustered by any variables, multi-way by
        inclusion and exclusion; as in xtabond2's ``cluster()``, the
        clustering also defines the two-step weight matrix.
    gmm, iv : list of Gmm, list of Iv, optional
        Instruments as objects, added after the formula's.
    collapse : bool, default False
        Collapse every GMM-style group.
    h : {1, 2, 3}, optional
        xtabond2's ``h()``: the one-step weight matrix's prior error
        covariance (default 3).
    small : bool, default False
        t and F statistics with small-sample corrections, as xtabond2's
        ``small``.
    artests : int, default 2
        Arellano-Bond tests up to this order.
    constant : bool, default True
        System GMM's constant (False: xtabond2's ``noconstant``).
    pca : bool, default False
        Replace the GMM-style instruments by their principal components
        (xtabond2's ``pca``); `components` fixes how many.
    weights : str, optional
        A column of weights; `weights_type` ``"aweights"`` (default) or
        ``"pweights"`` (robust variance).
    tdelta : int, optional
        The time variable's step (default 1).
    nl : {None, "noserial", "iid"}
        Add Ahn and Schmidt's (1995) nonlinear moment conditions, as
        xtdpdgmm's ``nl()``. ``"noserial"``, valid when the idiosyncratic
        errors are serially uncorrelated: E[u_iT (u_it - u_i,t-1)] = 0 for
        t < T, u the errors in levels (with the panel effect). ``"iid"``,
        when they are also homoskedastic over time: E[ubar_i (u_it -
        u_i,t-1)] = 0 for every period, and the linear E[y_i,t-2 du_i,t-1 -
        y_i,t-1 du_it] = 0. One per period, or one of each with `collapse`.
        Estimated by Gauss-Newton, with Windmeijer's correction in its
        general form; a one-step fit has the sandwich variance.
    nl_weight : float, default 1
        The first step's weight on the nonlinear moments relative to the
        linear ones (xtdpdgmm's ``nl(, weight())``): its weight matrix is
        the linear moments' (Z'HZ)^- and, for each nonlinear moment, 1 /
        (the number of panels) times this. 0 ignores them in the first step, which is then
        the linear one-step estimate.
    workdir : str or Path, optional
        Where the run directory goes (default ``$DPD_STREAM_WORKDIR``, then
        ``$HDFE_STREAM_WORKDIR``, then the system temporary directory).
    profile : bool, default False
        Return a `hdfe_stream.Profile` -- the estimated memory and disk, and a
        resource request (``p.recommend()``) -- instead of fitting. It reads
        the data once and holds nothing.
    **options
        ``storage`` (``"disk"`` or ``"memory"``), ``block_mb`` (rows per
        block, by size), ``tol`` and ``maxiter`` (iterated GMM),
        ``n_threads``, ``max_memory_gb`` (the memory the fit may use; the default is the job's
        allocation) and ``memory_reserve`` (the share of it held back, default 0.15), ``diffsargan`` (False skips the difference-in-Hansen
        tests), ``idtests`` (``"auto"``: the identification tests when the
        excluded instruments times the endogenous regressors are at most
        500; True, False), ``verbose``, ``logger``, ``log_level``.

    Returns
    -------
    DPDResult

    Examples
    --------
    >>> from dpd_stream import dpd_stream
    >>> from dpd_stream.simulate import simulate_dynamic
    >>> df = simulate_dynamic(2_000, 8, seed=1)
    >>> fit = dpd_stream("y ~ l(y, 1) + x + w | year | gmm(y, 2:4) + gmm(x, 1:3) + iv(w)",
    ...                  df, panel=("id", "year"), system=True, verbose=False)
    >>> fit.summary()                                           # doctest: +SKIP
    """
    est = DPD(fml, data, panel, gmm=gmm or (), iv=iv or (), system=system, steps=steps,
              transform=transform, h=h, vcov=vcov, small=small, artests=artests,
              collapse=collapse, constant=constant, pca=pca, components=components,
              weights=weights, weights_type=weights_type, tdelta=tdelta, workdir=workdir,
              nl=nl, nl_weight=nl_weight, **options)
    if profile:
        return est.profile()
    f = est.fit()
    return _result(est, f)


def _result(est, f) -> DPDResult:
    i = f.info
    small = est.small
    df_t = i["df_r"]
    if df_t is None:
        df_t = i["n_groups"] - (est.system and est.constant)
    key = est.vkey
    steps = est.steps

    def tests(stat, df):
        return dict(statistic=float(stat), df=int(df), p=float(stats.chi2.sf(stat, df)))

    all_vcovs = {key: (i["V"], df_t)}
    alt, alt_key = None, None
    if steps == 1:
        alt_key = "robust" if key == "iid" else "iid"
        alt = i["onestep_vcovs"].get(alt_key)
    elif steps == 2:
        alt_key = "iid" if key != "iid" else "robust"
        alt = i["V2robust"] if key == "iid" else i["V2"]
    if alt is not None:
        if alt_key == "robust":
            alt_key = "robust" if not est.cvars else key
        fac = i["vfac"]["iid1" if (steps == 1 and alt_key == "iid") else "other"]
        all_vcovs[alt_key] = ((alt + alt.T) / 2 * fac, df_t)
    og = dict(min=i["g_min"], max=i["g_max"],
              avg=i["n_obs"] / i["n_groups"] if i["n_groups"] else 0.0)
    hansen = (tests(i["hansen"], i.get("hansen_df", i["sargan_df"]))
              if i["hansen"] is not None else None)
    onestep = None
    if steps != 1:
        V1 = i["V1robust"] if est.robust else i["V1"]
        onestep = DPDResult(fml=est.fml, depvar=est.spec.depvar, coefnames=list(f.names),
                            beta=i["b1"], vcov=V1, vcov_type=key, df_t=df_t,
                            n_obs=i["n_obs"], estimator="system" if est.system else "difference",
                            steps=1, transform="fod" if est.orth else "fd",
                            n_groups=i["n_groups"], obs_per_group=og,
                            n_instruments=i["j0"], sargan=tests(i["sargan"], i["sargan_df"]),
                            small=small, weights=est.weights, weights_type=est.weights_type)
    lay = est.lay
    names = lay.column_names()
    inst = [names[c] for c in est.live] if est.P is None else []
    return DPDResult(
        fml=est.fml, depvar=est.spec.depvar, coefnames=list(f.names), beta=f.b, vcov=f.V,
        vcov_type=key, df_t=df_t, n_obs=int(i["n_obs"]), all_vcovs=all_vcovs,
        n_clusters=i["n_clusters"], collin_vars=list(i["dropped"]),
        diagnostics=dict(seconds=i["seconds"], peak_rss_gb=i.get("peak_rss_gb"),
                         wttot=i["wttot"], j_columns=i["j"],
                         generalized_inverse_one_step=i["pinv1"],
                         generalized_inverse_two_step=i.get("pinv2"),
                         iterations=i.get("iterations"), notes=lay.notes),
        weights=est.weights, weights_type=est.weights_type,
        estimator="system" if est.system else "difference", steps=steps,
        transform="fod" if est.orth else "fd", n_groups=int(i["n_groups"]),
        obs_per_group=og, n_instruments=int(i["j0"]), ar=i["ar"],
        sargan=tests(i["sargan"], i["sargan_df"]), hansen=hansen,
        diff_hansen=i["diffsargan"], model_test=i["wald"], sigma2=float(i["sig2"]),
        mmsc=i.get("mmsc"), onestep=onestep, pca=i["pca"], small=small,
        absorbed=i.get("absorbed"), idtests=i.get("idtests"),
        nl=(dict(moments=est.Lnl, converged=i.get("nl_converged", True), kind=est.nl,
                 weight=est.nl_weight) if est.nl else None),
        instrument_list=inst, time_effects=est.spec.time_effects,
        _model=dict(source=est.data, panel=est.panel, tdelta=est.store.tdelta,
                    bases=est.spec.bases, yterm=est.spec.y, coef_all=i["coef_all"],
                    terms={nm: lay.terms[t] for nm, t in zip(lay.xnames, lay.x)},
                    iv_cols=[(lay.terms[t], eq, passthru, mz)
                             for t, eq, passthru, mz, _ in lay.iv_cols],
                    complete_terms=[lay.terms[t] for t in lay.complete_terms],
                    system=est.system, orth=est.orth, weights=est.weights))
