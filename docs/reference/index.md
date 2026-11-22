# Function reference

Everything here is importable from `dpd_stream`, except the simulators,
which are in `dpd_stream.simulate`.

| Name | What it is |
|---|---|
| **[Estimation](estimation.md)** | |
| `dpd_stream` | difference and system GMM |
| `Gmm`, `Iv` | GMM-style and IV-style instrument groups, as objects |
| **[Results](results.md)** | |
| `DPDResult` | a fitted model: coefficients, variances, specification tests, residuals |
| `DPDMulti` | several results, for the reporting methods |
| `DPDBootstrapResult` | a fit's refit bootstrap (`DPDResult.refit_bootstrap`): intervals, the bootstrap Hansen test |
| `BootstrapFailureWarning` | some of a bootstrap's refits failed |
| **[Utilities](utilities.md)** | |
| `etable` | regression tables, through pyfixest (hdfe-stream's) |
| `bonferroni`, `holm` | multiple-testing corrections (hdfe-stream's) |
| `cleanup` | remove run directories left by killed processes |
| `simulate_dynamic`, `truth_dynamic` | simulated dynamic panels with known coefficients |

```{toctree}
:hidden:

estimation
results
utilities
```
