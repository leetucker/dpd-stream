# Ahn–Schmidt moments

{download}`Download the script: nonlinear.py <../../examples/nonlinear.py>`

**The problem.** When y is persistent (ρ near 1), its lagged levels say
little about its changes. Difference GMM's instruments are then weak, and
its estimates imprecise. Two remedies:

- **System GMM** adds the levels equation. But it needs the panels to start
  near their long-run means, a stationarity assumption.
- **Ahn and Schmidt (1995)** showed that with serially uncorrelated errors
  the levels errors satisfy $E[u_{iT}(u_{it} - u_{i,t-1})] = 0$. These are
  moments quadratic in the coefficients, valid without stationarity.

`nl="noserial"` adds them, one per period, or one with `collapse=True`.

These panels start far from their long-run means, so system GMM's extra
moments are invalid:

```python
df = simulate_dynamic(5_000, 7, rho=(0.9,), stationary=False, seed=6)
```

```text
difference GMM                 rho = 0.8608 (0.0231)   Hansen p = 0.872
difference GMM + Ahn-Schmidt   rho = 0.8993 (0.0166)   Hansen p = 0.749
system GMM                     rho = 0.9095 (0.0041)   Hansen p = 0.433
```

The true ρ is 0.9. The Ahn–Schmidt moments cut difference GMM's standard
error by about 30% and pull the estimate to the truth. System GMM's estimate looks
more precise than it is, because its moments do not hold here.

**How it is fit,** as xtdpdgmm's `nl()`:

1. **The first step** is a one-step GMM estimate with the nonlinear moments.
   Its weight matrix is the linear moments' (Z'HZ)⁻¹. The nonlinear moments
   are treated as independent of them, each with weight 1 / (the number of
   panels) times `nl_weight` (xtdpdgmm's `weight()`, default 1). With
   `nl_weight=0` the first step ignores them, and is the linear one-step
   estimate.
2. **The second step** minimizes the full objective by Gauss–Newton. The
   objective is not convex; the result reports whether the step converged.
3. **The standard errors** have Windmeijer's correction in its general,
   nonlinear form. A one-step fit (`steps=1`) reports the first step's
   sandwich variance.

**Homoskedastic errors.** `nl="iid"` uses Ahn and Schmidt's moments for
errors that are also homoskedastic over time. These are
$E[\bar u_i (u_{it} - u_{i,t-1})] = 0$ for every period, and the linear
$E[y_{i,t-2}\Delta u_{i,t-1} - y_{i,t-1}\Delta u_{it}] = 0$. The linear ones
also need the other regressors to be predetermined or exogenous.

**Matching xtdpdgmm.** dpd-stream's `nl="noserial"` estimates equal xtdpdgmm's
to six digits (xtdpdgmm's Gauss–Newton stops at about 1e-5), on balanced and on
unbalanced panels with gaps: with the default first step, with `nl_weight=0`,
and one-step. That checks the moments, the first-step weights, the solver and
the second step. Two rules matter in unbalanced panels, and both agree with
xtdpdgmm: the term $u_{iT}\Delta u_{it}$ exists only if period $t+1$ is
observed too, and every nonlinear moment has the same initial weight, 1 over
the number of panels, however many panels contribute to it. With
`nl="iid"` the same holds (balanced and unbalanced, to about 1e-5), once the
linear moments it adds are weighted as xtdpdgmm does: as instruments of the
differenced equation, within the linear block's $Z'HZ$ weight, not with the
nonlinear moments. System GMM matches too, with `nl="noserial"` and
`nl="iid"`. Three differences remain:

- **Standard errors** of the nonlinear fits differ by up to 3% (two-step; 7% one-step) on balanced
  panels and more on unbalanced ones (those of the linear fits agree
  exactly). The cause is xtdpdgmm's: its stored second-step weight matrix
  equals ours entry by entry and its Hansen statistic agrees to seven digits,
  and our Jacobian of the moments satisfies the first-order condition at
  xtdpdgmm's estimate (it is the exact derivative of the objective xtdpdgmm
  minimizes), but the variance xtdpdgmm reports is not the one that Jacobian
  gives. We keep the exact one. Treat the nonlinear standard errors as
  experimental when a few percent matter.
- **Forward orthogonal deviations:** xtdpdgmm scales them by
  $(T-t)/(T-t+1)$, not by its square root as xtabond2 and dpd-stream do, so
  the two give different estimates whenever instruments are shared across
  periods (collapsed ones, `iv()`). With one instrument per period the
  scale cancels. The `nl` fits with `transform="fod"` are therefore checked
  against the dense implementation, not against xtdpdgmm.
- **Counts:** xtdpdgmm reports the levels' observations and panels, we the
  differenced rows and panels, and it counts the linear moments `nl(iid)`
  adds among the linear ones.

The pdynmc package, which implements the same moments, agrees with
dpd-stream's one-step estimates once its one-moment-shorter set (it starts a
period later) is matched.

When comparing with xtdpdgmm, note that its `iv(w)` with `model(diff)`
instruments with $w$ itself; xtabond2's, and dpd-stream's, use $\Delta w$.
`iv(w, passthru=True)` is xtdpdgmm's.

:::{dropdown} The script
```{literalinclude} ../../examples/nonlinear.py
:language: python
```
:::
