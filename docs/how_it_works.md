# How it works

## The estimator

Write the model in levels, for panel $i$ and period $t$, as $y_{it} =
x_{it}'\beta + \alpha_i + \varepsilon_{it}$, where $x$ includes lags of $y$.

**The transformation.** A transformation removes $\alpha_i$: first
differences, or forward orthogonal deviations. System GMM stacks the
transformed equation and the levels equation. For each panel, $X_i$ and
$y_i$ are the stacked regressors and outcome, and $Z_i$ are the
instruments:

- **GMM-style instruments:** one column per lag and period (or per lag, when
  collapsed);
- **IV-style instruments:** one column per variable.

**One-step GMM** weights the moments by $A_1 = (\sum_i Z_i'HZ_i)^{-1}$, where
$H$ is the prior covariance of the transformed errors (xtabond2's `h()`):

$$\hat\beta_1 = (X'Z A_1 Z'X)^{-1} X'Z A_1 Z'y.$$

**Two-step GMM** replaces $A_1$ by $A_2 = (\sum_c g_c g_c')^{-1}$, where $g_c =
Z_c'\hat e_c$ is a cluster's moment at the one-step residuals. The clusters
are the panels by default.

Everything the estimator and its tests need is a sum over panels:

| Quantity | Size |
|---|---|
| Z′X | j × K |
| Z′y | j |
| Σᵢ Zᵢ′HZᵢ | j × j |
| Σ g g′ | j × j |
| the sums behind Windmeijer's correction and the AR tests | j × K, and scalars |

Here j is the number of instruments and K the number of regressors.

## Pass 0: blocks of panels

The source is read once by Polars. It can be a Parquet file or glob, or a
Lazy- or DataFrame.

**What is stored.** Every distinct expression the model uses is evaluated
row by row:

- the outcome;
- the regressors;
- the instruments' variables;
- `log(x)`, interactions and dummies.

The time variable becomes an index on the global grid of T periods, and
cluster variables become integer codes. Rows are written sorted by panel
and time, as blocks of whole panels, using hdfe-stream's `BlockStore`. The
blocks are on disk by default, or in memory with `storage="memory"`.

**What is not stored:** lags, differences and instruments.

**Missing values.** Rows with missing values are kept, because a missing
value breaks a lag. As in Stata, a lag exists only where the panel has an
observation: `l(x, 1)` at a gap is missing, not the value before it.

## The passes

Each pass reads the blocks in turn, the next one in a background thread
while the current one is worked on. A numba kernel takes the panels in
parallel. For each panel, the kernel builds the stacked rows:

- the transformed and levels equations;
- whether each row is in the sample;
- each row's instruments, as a list of (column, value) pairs.

The pairs follow xtabond2's column layout exactly. A row of an uncollapsed
GMM-style group has one entry per lag and variable, not one per instrument
column. So the cost of a pass is proportional to the nonzero instruments,
which is O(N T L_row): N panels, T periods, L_row the nonzeros per row. The
cost of a dense build is O(N T j).

**Pass 1** accumulates the moments that don't depend on the estimates:

- Z′X and Z′y;
- Σᵢ Zᵢ′HZᵢ, from H's few nonzeros per row;
- the cross-products for σ² and for the collinearity check;
- the counts;
- and, if asked, the moments for PCA, the identification tests, absorbed
  effects and Ahn–Schmidt moments.

The one-step estimate, σ² and the Sargan test follow, without the data.

**Pass 2** computes each panel's $g_i = Z_i'\hat e_i$ at the one-step
estimates, in batches. BLAS then forms $\sum g_i g_i'$. That gives:

- the robust one-step variance;
- the two-step estimate;
- the Hansen test.

Iterated GMM repeats this pass.

**Pass 3** returns to the one-step residuals for Windmeijer's sums. xtabond2's

$$\textstyle\sum_c \big[(e_{1c}'Z_c A_2 Z'e_2)\, Z_c'X_c + Z_c'e_{1c}\,(e_2'Z A_2 Z_c'X_c)\big]$$

needs both residuals and $A_2$, which is why it is a pass of its own. The
same pass computes the Arellano–Bond tests at the two-step residuals.

## Everything else, from the sums

- **Difference-in-Hansen tests.** Each instrument group's test re-estimates
  the model without the group. Following xtabond2, its weight is the
  matching submatrix of the full model's moment covariance. These are
  closed-form functions of Z′X, Z′y and S.
- **PCA.** xtabond2's `pca` replaces the GMM-style instruments by principal
  components of their correlation matrix. The projection P is applied to
  the sums after the fact: P′Z′X, P′SP.
- **MMSC.** Andrews and Lu's criteria come from Hansen's J.

## The generalized inverse

With more instruments than the data identify, the moment covariance is
singular. This is common with small samples or time dummies. xtabond2 then
uses Stata's `invsym`, which sweeps the pivots in column order and drops a
column whose pivot vanishes. The estimates then depend on the order of the
instruments. dpd-stream sweeps the same way, with the columns in xtabond2's
order:

1. IV-style columns, then GMM-style;
2. within each, the groups in reverse order of specification;
3. within a group, by lag, then period, then variable.

The results therefore agree in these cases too.

## Clustering

Clustering follows xtabond2's `cluster()`: the clusters define the two-step
weight matrix, not only the variance. With several variables, every
combination enters by inclusion and exclusion (Cameron, Gelbach and Miller
2011). The kernel sums the moments by cluster in one of three ways:

- **By panel:** the default, and a cluster variable constant within panels.
  The kernel sums each panel's moments, then adds them up by cluster.
- **By observation:** the intersection of the panel and the time variable.
- **In a table of cluster sums:** any other clustering, such as by year.

Multi-way clustering in a dimension with few clusters (years, typically) is
fragile. Its part of the weight matrix has the rank of its cluster count, so
the combination can be singular or indefinite. Some coefficients' variances
can then be negative, and two-step estimates unreliable. Clustering a
one-step fit avoids the weight-matrix problem.

## Identification tests

The Kleibergen–Paap and Sanderson–Windmeijer tests (see
[identification tests](examples/identification.md)) need the instruments'
cross-products, which pass 1 accumulates, and the robust covariances of
vec(Z′Y) and vec(Z′V̂), V̂ the first-stage residuals. Those come from one
more pass, summed by cluster as the moment covariance is. The algebra on
those sums is hdfe-stream's `core.ivtests`, shared with feols_stream's 2SLS.

## Absorbed effects

**Time effects.** `| year` is estimated as a dummy per period, each its own
IV-style instrument, as xtabond2's `yr*` in the regressors and in `iv()`.
The dummies are not reported, and are left out of the model's Wald test.
There are only T of them, so this costs little, and everything is exact.

**Period effects by group.** `| g^year` in difference GMM absorbs a
separate year effect for each value of g, where g is constant within
panels: a region, an industry. There can be thousands of such cells, too
many for dummies, so the estimator partials them out instead. Write D for
the effects' dummies in the transformed equation and W for the weights:

- **The instruments** become Z − D Mz, where Mz = (D′WD)⁻D′WZ. With first
  differences, D is the cell's indicator and Mz the cell's means of Z. With
  orthogonal deviations, D holds the deviations of the dummies, so each
  group has a T × T block of D′WD.
- **Same estimates as the dummies.** The effects are exactly identified by
  their own IV-style moments. Concentrating them out of the GMM objective
  gives the same estimates of the other coefficients as the dummies would,
  at any weight matrix.
- **Exact weight matrices.** The two-step and iterated weight matrices use
  the dummies model's residuals. Those take the effects' GMM estimates, not
  their means. The estimates need the covariance of the effects' moments
  with the instruments', one more pass when clusters cross panels.

**Windmeijer's correction and the AR tests.** Both depend on the effects'
estimates, not just on the coefficients': the two-step estimates move with
the one-step effects through the weight matrix, and the AR statistic with
the effects through the residuals. Treating the effects as concentrated
out would ignore their own sampling noise. Done exactly, as the dummies
model does, they need the covariance of the effects' moments D′e, a cells ×
cells matrix: 3.2 GB at 20,000 cells. It is never formed:

- **Everything is linear in the moments.** The effects' estimates are a
  fixed linear function of the moments, so each statistic needs that
  covariance only in a few directions: one per coefficient for Windmeijer's
  correction, two per AR test.
- **A pass per set of directions.** Each direction's score, summed by
  cluster, is a number per cluster. The variance in those directions comes
  from one more pass, with every clustering. The derivatives in the effects
  are cells × K sums the Windmeijer pass already forms, so the memory added
  is that of the cell sums, a few per coefficient.

That is one more pass for a two-step fit with Windmeijer's correction or
with the AR tests, and one more for the AR tests with clusterings by
variables. One-step fits need one more for robust AR tests. One-step
non-robust tests need none: their covariance is the one-step weight
matrix's, which pass 1 has by cell.

So everything comes out the same as the dummies model: the estimates and
their variances, with or without Windmeijer's correction; σ²; and the
Sargan, Hansen, difference-in-Hansen, AR and identification tests.

**Not with Ahn–Schmidt moments.** Those are nonlinear in the effects, which
then cannot be partialled out. `| year` (dummies) works with them.

## Ahn–Schmidt moments

With `nl="noserial"`, each panel adds the moments $u_{iT}(u_{it} -
u_{i,t-1})$ for $t < T$, where $u = y - x'\beta$ in levels and $T$ is the
panel's last period. With `nl="iid"`, the moments are $\bar u_i(u_{it} -
u_{i,t-1})$ for every period, and the linear $y_{i,t-2}\Delta u_{i,t-1} -
y_{i,t-1}\Delta u_{it}$. All are at most quadratic in β, so pass 1
accumulates their coefficients. The moments, their sum and its derivative
at any β then come from memory.

The fit proceeds as xtdpdgmm's:

1. **First step:** one-step GMM with all the moments, by Gauss–Newton from
   the linear one-step estimate. Its weight matrix is (Z′HZ)⁻ for the
   linear moments and, treated as independent of them, the nonlinear ones'
   inverse counts times `nl_weight` (xtdpdgmm's `weight()`).
2. **Weight:** the robust covariance of all the moments at that estimate,
   from one pass.
3. **Second step:** Gauss–Newton on the full objective, in memory.

Windmeijer's correction takes its general nonlinear-GMM form, with the
first step's sandwich variance.

**How close to xtdpdgmm.** xtdpdgmm starts Gauss–Newton from zero, and
rescales moments in unbalanced panels. With `nl_weight=0` (its
`weight(0)`) the first step is linear, and the two should agree; the Stata
bundle checks both.

## Instruments moved to deeper lags

In system GMM, a collapsed GMM-style group's levels instrument is the
difference at lag a − 1. If that column is zero throughout the sample,
xtabond2 moves it to deeper lags until it has values, up to lag b − 1. The
budget of moves is shared by the group's variables, in order. Pass 1 records
which deeper lags have values, and in that rare case pass 1 runs again with
the moved columns. xtabond2 moves the column a row at a time across the
whole data set, so a panel's last value can land in the next panel's first
row. dpd-stream moves it within each panel.

## Numerical agreement

dpd-stream agrees to about 1e-8 with each of these:

- **A dense implementation of xtabond2's algebra,** kept in the tests. It
  covers every option, on unbalanced panels with gaps.
- **xtabond2's published results:** Roodman 2009, to the printed digits, on
  Stata's abdata (which dpd-stream does not distribute; the tests run when
  it is there).
- **pydynpd,** on a simulated panel shipped with the tests.
- **plm's `pgmm`,** for Windmeijer's correction and the AR tests, on the
  same panel.

Two thresholds are looser than Stata's, because Stata forms cross-products
in quad precision:

- **Collinear regressors** are detected at a pivot of 1e-9, not
  `invsym`'s 1e-13. An exact collinearity summed in double precision over
  millions of rows leaves a pivot near 1e-12, not near 1e-16.
- **Singular moment covariances** leave estimates about 1e-6 apart, because
  each generalized inverse amplifies its own roundoff.
