# Stata reference results

dpd-stream reproduces xtabond2. This bundle produces the numbers that check
it, on a machine with Stata, for the specifications in `specs.py`. Each
specification is written in both syntaxes. The tests skip until the
results are in `tests/data/`.

## Make the bundle (here)

```bash
python benchmarks/external/stata/make_bundle.py            # -> benchmarks/external/stata/bundle/
```

The bundle holds three files:

- `dpd_refs.do`, which runs every specification.
- `dpdsave.do`, which writes each result as one line of JSON.
- `data/sim.csv`, the simulated panel. It is unbalanced, has gaps, and carries
  clusters, weights and a group `g` that is constant within each panel.

Stata's own `abdata` is not included: the `.do` file loads it with
`webuse` and exports it, so the tests run on exactly Stata's data.

## Run it (on the Stata machine)

You need Stata 15 or later, and these packages from SSC:

```stata
ssc install xtabond2
ssc install xtdpdgmm
ssc install ivreg2
ssc install ranktest
ssc install underid
```

Then, in the bundle's directory:

```stata
do dpd_refs.do
```

The run takes a minute or two. It sets `mata set matafavor speed`, the mode
whose instrument construction dpd-stream follows. It writes three files:

- `stata_refs.json`: one line of results per specification.
- `data/abdata.csv`: Stata's abdata, exported.
- `dpd_refs.log`: the full output, including each package's version.

## Bring back

Copy two files into this repository's `tests/data/`:

| Bundle file | Copy to |
|---|---|
| `stata_refs.json` | `tests/data/stata_refs.json` |
| `data/abdata.csv` | `tests/data/abdata_stata.csv` |

`tests/test_stata.py` then compares every specification.
