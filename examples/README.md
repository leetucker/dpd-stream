# Examples

Each script runs on its own, from the repository's root:

```bash
python examples/quickstart.py
```

| File | What it shows |
|---|---|
| `quickstart.py` | Difference and system GMM on a simulated panel; `summary()`, `etable`, the one-step results, long-run effects |
| `arellano_bond.py` | Arellano and Bond (1991) and Blundell and Bond (1998) on their UK firm data, matching Roodman's (2009) xtabond2 results |
| `instruments.py` | Lag limits, `collapse`, forward orthogonal deviations, PCA, the Hansen and difference-in-Hansen tests, MMSC |
| `inference.py` | One- and two-step standard errors, Windmeijer's correction, one- and two-way clustering, `small` |
| `absorbed.py` | Absorbing region-by-year effects in difference GMM |
| `nonlinear.py` | Ahn and Schmidt's nonlinear moments with a persistent series |
| `identification.py` | Kleibergen-Paap and Sanderson-Windmeijer identification tests |
| `out_of_core.py` | A Parquet panel larger than memory, and residuals sunk to Parquet |

Output goes to `examples/output/`, which is not committed. The
documentation shows each script with its output.
