# Examples

Each page mirrors a script in the repository's `examples/` directory and
shows its output. Run the scripts from the repository's root:

```bash
python examples/quickstart.py
```

| Page | What it shows |
|---|---|
| [Quickstart](quickstart.md) | Difference and system GMM; `summary()`, `etable`, the one-step results, long-run effects |
| [Arellano and Bond's employment equations](arellano_bond.md) | Arellano and Bond (1991) and Blundell and Bond (1998), matching Roodman's (2009) xtabond2 output |
| [Choosing instruments](instruments.md) | Lag limits, `collapse`, forward orthogonal deviations, PCA, Hansen and difference-in-Hansen tests, MMSC |
| [Standard errors](inference.md) | Windmeijer's correction, one- and two-way clustering, `small` |
| [Absorbed period effects](absorbed.md) | Region-by-year effects in difference GMM |
| [Ahn–Schmidt moments](nonlinear.md) | Nonlinear moment conditions for persistent series |
| [Identification tests](identification.md) | Kleibergen–Paap and Sanderson–Windmeijer tests |
| [Out of core](out_of_core.md) | A Parquet panel larger than memory |

## The simulated data

`dpd_stream.simulate.simulate_dynamic` draws

$$y_{it} = \rho_1 y_{i,t-1} + \dots + \beta x_{it} + \gamma w_{it} + \alpha_i + \lambda_t + \varepsilon_{it}$$

**The variables:**

- **x** is predetermined. It responds to past shocks, x_it = 0.5 x_i,t−1 +
  0.5 α_i + 0.3 ε_i,t−1 + noise. It can also be made endogenous or strictly
  exogenous.
- **w** is strictly exogenous, but correlated with α_i.

**Options:**

- **Starting point.** With `stationary=True`, panels start from their
  long-run means, the condition system GMM's levels moments need. With
  `stationary=False` they don't.
- **Unbalanced panels and gaps:** `unbalanced=True` and `p_gap=`.
- **Extra columns:** `n_clusters=` adds a column of clusters, and
  `weights=True` a column of panel weights.

`truth_dynamic()` returns the true coefficients, named as the formulas name
them.

```{toctree}
:hidden:

quickstart
arellano_bond
instruments
inference
absorbed
nonlinear
identification
out_of_core
```
