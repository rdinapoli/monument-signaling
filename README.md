# monument-signaling

Code and figures for *Monument construction as multilevel signaling:
individual rewards and fading signals explain why groups build and keep
building* (DiNapoli and Lipo, Evolutionary Human Sciences, submitted).

The model is analytical, so there are no data files. Every number and
figure in the paper and its supplement is produced by the code here.
`output/figures/` already contains every figure, so nothing needs to run
in order to inspect the results.

## How to use

Requires Python 3.11 or later. To reproduce everything (tests, then all
figures; about 6 minutes):

```bash
bash run_all.sh
```

Or step by step:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
PYTHONPATH=src .venv/bin/python -m pytest tests/              # 606 tests
PYTHONPATH=src .venv/bin/python scripts/generate_figures.py   # or any other script
```

## What is where

```
src/signaling/     the model
  layer1.py          individual signaling equilibrium and channel selection
  layer2.py          intergroup assessment and war avoidance
  layer3.py          cooperation networks and crisis buffering
  price_equation.py  multilevel assembly and the critical threshold
  spatial.py, emergence.py, placement.py, competition.py, size_inequality.py
  calibration.py     parameter values
  plotting.py        shared plotting helpers
tests/             test suite, one file per module
scripts/           figure scripts (table below) and compute_rapanui_energetics.py
output/figures/    all figures (fig<n> main text, figS<n> supplement)
docs/notation.pdf  every symbol in the paper and the function that implements each equation
run_all.sh         reproduces everything
```

| Script | Figures |
|---|---|
| `generate_figures.py` | 2, 3, 6; S1, S9, S10, S13, S21, S22, S23, S25 |
| `generate_supplementary_figures.py` | S2, S3, S4, S5, S7, S8, S14, S15, S18, S20, S24, S27 |
| `generate_emergence_figure.py` | 4 |
| `generate_spatial_figure.py` | 5; S26 |
| `generate_figure_audience_weights.py` | S6 |
| `generate_figure_assessment_alternatives.py` | S11 |
| `generate_figure_self_assessment.py` | S12 |
| `generate_figure_k0_sensitivity.py` | S16 |
| `generate_figure_omega_k0_joint.py` | S17 |
| `generate_figure_Cn_joint.py` | S19 |
| `generate_figure_placement.py` | S28 |
| `generate_figure_competition.py` | S29 |

Fig. 1 is a drawing (`fig1_overview_source.svg`), not a script output.

## Contact

Beau DiNapoli (dinapoli@binghamton.edu) and Carl P. Lipo, Department of Anthropology, Binghamton University.

## Citation

DiNapoli, B. and Lipo, C. P. (submitted). Monument construction as multilevel signaling: individual rewards and fading signals explain why groups build and keep building. *Evolutionary Human Sciences*. A DOI-bearing `CITATION.cff` will be added on acceptance.

## License

MIT (see `LICENSE`).
