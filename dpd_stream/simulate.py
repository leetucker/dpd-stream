"""Simulated dynamic panels, for the examples, the tests and the benchmarks.

`simulate_dynamic` draws the model

    y_it = rho_1 y_i,t-1 + ... + beta x_it + gamma w_it + alpha_i + lambda_t + eps_it

with a fixed effect alpha_i, optional time effects lambda_t, a regressor x
that is predetermined (correlated with past shocks), endogenous
(correlated with the current one) or strictly exogenous, and a strictly
exogenous w. Both x and y depend on alpha_i, so OLS is biased upward and
within-groups downward (Nickell 1981); difference and system GMM are
consistent.

With ``stationary=True`` the panels start from their long-run means, so
that the deviations of the initial y from them are unrelated to alpha_i:
the condition under which system GMM's levels moments hold (Blundell and
Bond 1998). With ``stationary=False`` they start at a multiple of alpha_i,
which violates it.

Everything is numpy and Polars, seeded and deterministic.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["simulate_dynamic", "truth_dynamic"]


def simulate_dynamic(n_panels: int = 1_000, T: int = 8, *, rho=(0.5,), beta: float = 1.0,
                     gamma: float = 0.5, x_type: str = "predetermined", delta: float = 0.5,
                     theta: float = 0.5, phi: float = 0.3, sd_alpha: float = 1.0,
                     sd_eps: float = 1.0, time_effects: bool = True, stationary: bool = True,
                     burn: int = 50, unbalanced: bool = False, p_gap: float = 0.0,
                     n_clusters: int | None = None, weights: bool = False,
                     heteroskedastic: bool = False, start_year: int = 2000,
                     seed: int = 0) -> pl.DataFrame:
    """A simulated dynamic panel.

    Parameters
    ----------
    n_panels, T : int
        Panels and periods (before any are dropped).
    rho : sequence of float
        The coefficients on the lags of y (stationary: their sum below 1).
    beta, gamma : float
        The coefficients on x and on w.
    x_type : {"predetermined", "endogenous", "exogenous"}
        x_it = delta x_i,t-1 + theta alpha_i + phi eps_i,t-1 (predetermined)
        or phi eps_it (endogenous) or nothing (exogenous), + noise.
    sd_alpha, sd_eps : float
        Standard deviations of the fixed effects and the shocks.
    time_effects : bool
        Add common time effects lambda_t (then fit with time effects).
    stationary : bool
        Start each panel at its long-run mean (see the module docstring).
    burn : int
        Periods simulated and discarded before the first one kept.
    unbalanced : bool
        Panels start and end at random periods (each keeps at least 3).
    p_gap : float
        Probability that an observation inside a panel is missing (a gap).
    n_clusters : int, optional
        Add a column ``cluster`` grouping the panels into this many clusters.
    weights : bool
        Add a column ``wt`` of panel-level weights (uniform on 0.5..2).
    heteroskedastic : bool
        Shocks' standard deviation varies by panel (uniform on 0.5..1.5 times
        `sd_eps`).
    start_year : int
        The first period's ``year``.
    seed : int

    Returns
    -------
    polars.DataFrame
        Columns ``id``, ``year``, ``y``, ``x``, ``w`` (and ``cluster``,
        ``wt``), sorted by id and year.
    """
    if x_type not in ("predetermined", "endogenous", "exogenous"):
        raise ValueError("x_type must be 'predetermined', 'endogenous' or 'exogenous'")
    rho = np.atleast_1d(np.asarray(rho, float))
    p = len(rho)
    rng = np.random.default_rng(seed)
    N = int(n_panels)
    total = burn + T
    alpha = rng.normal(0, sd_alpha, N)
    scale = rng.uniform(0.5, 1.5, N) * sd_eps if heteroskedastic else np.full(N, sd_eps)
    eps = rng.normal(0, 1, (N, total)) * scale[:, None]
    lam = rng.normal(0, 0.5, total) if time_effects else np.zeros(total)
    w = rng.normal(0, 1, (N, total)) + 0.3 * alpha[:, None]
    x = np.zeros((N, total))
    y = np.zeros((N, total))
    x_mean = theta * alpha / (1 - delta)
    lr = (beta * x_mean + gamma * 0.3 * alpha + alpha) / (1 - rho.sum())
    if stationary:
        y[:, :p] = lr[:, None] + rng.normal(0, sd_eps, (N, p))
        x[:, 0] = x_mean + rng.normal(0, 1, N)
    else:
        y[:, :p] = (2.0 * alpha)[:, None] + rng.normal(0, sd_eps, (N, p))
        x[:, 0] = rng.normal(0, 1, N)
    for t in range(1, total):
        fb = 0.0
        if x_type == "predetermined":
            fb = phi * eps[:, t - 1]
        elif x_type == "endogenous":
            fb = phi * eps[:, t]
        x[:, t] = delta * x[:, t - 1] + theta * alpha + fb + rng.normal(0, 1, N)
        if t >= p:
            ar = sum(rho[k] * y[:, t - 1 - k] for k in range(p))
            y[:, t] = ar + beta * x[:, t] + gamma * w[:, t] + alpha + lam[t] + eps[:, t]
    keep = np.ones((N, T), bool)
    if unbalanced:
        first = rng.integers(0, max(1, T - 3), N)
        length = rng.integers(3, T + 1, N)
        tt = np.arange(T)
        keep &= (tt[None, :] >= first[:, None]) & (tt[None, :] < (first + length)[:, None])
    if p_gap > 0:
        gaps = rng.random((N, T)) < p_gap
        gaps[:, 0] = False
        keep &= ~gaps
    ii, tt = np.nonzero(keep)
    cols = {
        "id": ii.astype(np.int64),
        "year": (start_year + tt).astype(np.int64),
        "y": y[:, burn:][ii, tt],
        "x": x[:, burn:][ii, tt],
        "w": w[:, burn:][ii, tt],
    }
    if n_clusters:
        cl = rng.integers(0, n_clusters, N)
        cols["cluster"] = cl[ii].astype(np.int64)
    if weights:
        cols["wt"] = rng.uniform(0.5, 2.0, N)[ii]
    return pl.DataFrame(cols)


def truth_dynamic(rho=(0.5,), beta: float = 1.0, gamma: float = 0.5) -> dict:
    """The true coefficients of `simulate_dynamic`, named as a formula
    ``y ~ l(y, 1:p) + x + w`` names them."""
    rho = np.atleast_1d(np.asarray(rho, float))
    out = {f"l(y, {k + 1})": float(r) for k, r in enumerate(rho)}
    out.update({"x": float(beta), "w": float(gamma)})
    return out
