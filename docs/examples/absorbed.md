# Absorbed period effects

{download}`Download the script: absorbed.py <../../examples/absorbed.py>`

**The problem.** When shocks are common to a region in a year, year dummies
are not enough: the errors share a region-year component, and the
instruments are correlated with it. A dummy for every region-year cell would
fix that, but with 400 regions it means 2,400 more regressors (the cells
with observations), and as many more instruments.

**The fix.** In difference GMM, `| region^year` absorbs them. The region
must be constant within each panel. The estimates are the same as with the
dummies (see [how it works](../how_it_works.md#absorbed-effects)).

```python
inst = "gmm(y, 2:4) + gmm(x, 1:3) + iv(w)"
naive = dpd_stream(f"y ~ l(y, 1) + x + w | year | {inst}", df, panel=("id", "year"))
cells = dpd_stream(f"y ~ l(y, 1) + x + w | region^year | {inst}", df, panel=("id", "year"))
```

```text
year effects only:   [0.3347 0.9283 0.4494]
region-year cells:   [0.4938 0.9966 0.4988] {'term': 'region^year', 'effects': 2400, 'method': 'cells'}
```

The true values are 0.5, 1 and 0.5.

**Everything is the dummies'.** The estimates and their variances
(Windmeijer-corrected or not), σ², and the Sargan, Hansen,
difference-in-Hansen and Arellano–Bond tests are those of the model with
the dummies among the regressors and instruments, with any clustering.
Windmeijer's correction and the Arellano–Bond tests take one or two more
passes over the data for it (see
[how it works](../how_it_works.md#absorbed-effects)).

**Options.** The term is a single `g^year`, with g constant within panels, in
difference GMM. It works with:

- first differences or orthogonal deviations;
- any clustering, weights, one-step, two-step or iterated GMM;
- the identification tests, which partial the effects out.

It does not work with PCA, or with Ahn and Schmidt's moments, which are
nonlinear in the effects.

**`| year` is different.** Time effects alone are few, so `| year` is
estimated as a dummy per period, each its own instrument, and the dummies are
not reported. Everything is then exact, in both estimators, with every
option.

:::{dropdown} The script
```{literalinclude} ../../examples/absorbed.py
:language: python
```
:::
