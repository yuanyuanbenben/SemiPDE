# In-vitro cell culture assay (PC-3 scratch test)

Code, data and results for the first real-data application: the scratch assay of
PC-3 prostate cancer cells from Jin et al., *J. Theor. Biol.* (2016),
doi:10.1016/j.jtbi.2015.10.040.

## Contents

```
semipde/              the shared implementation
scripts/              realdata_cell.py, realdata_ci.py, realdata_mechanism.py
realdata/realdata1/   preprocessed .npy curves and the raw .xlsx supplements
runs/realdata_cell/   fitted results (JSON) and the proposal's state_dict files
figures/              fitted mechanism figure
```

Run the commands below from this directory.

## Data

The assay has six initial cell densities (10,000–20,000 cells), three replicate
wells each, 18 curves in total, observed at 0, 12, 24, 36 and 48 h on 38 spatial
positions (50 µm spacing, 1.9 mm span). The split is fixed and not shuffled:

- 12 training curves;
- 3 validation curves, one replicate per density (`[2, 11, 17]`), used only to
  select the epoch;
- 3 test curves, the legacy selection `[4, 9, 14]`.

The initial condition is the observed `t = 0` slice and is not fitted.

## Models

Two variants of the same framework are reported:

- **linear**: `d_t u = D Δu + f(u)`;
- **porous**: `d_t u = D d_x((u/K) d_x u) + f(u)` with `K = 1.7e-3` fixed.

The diffusivity `D` and the unknown mechanism `f(u)` are estimated jointly by
profiling; the mechanism is a neural network. The reported 95% interval for `D`
is the Wald interval from the parametric inference in the paper.

## Results

Mean squared errors are in the raw cell-density scale; `D` is in µm²/h.

Linear diffusion:

| method | D | λ (1/h) | train MSE | test MSE ×1e8 |
|---|---|---|---|---|
| **SemiPDE** | **1427** [1367, 1487] | — | 1.97e-8 | **1.624** |
| logistic benchmark | 913.9 | 0.0487 | 2.57e-8 | 1.942 |
| baseline, `f = 0` | 403.5 | — | 2.15e-7 | 20.387 |
| nonparametric | — | — | — | 2.385 |
| PINN | 108.7 | — | — | 0.438 |
| cascading | 385.0 | — | — | 0.926 |

Porous diffusion:

| method | D | λ (1/h) | train MSE | test MSE ×1e8 |
|---|---|---|---|---|
| **SemiPDE** | **3538** [3381, 3696] | — | 1.76e-8 | **1.567** |
| logistic benchmark | 3036 | 0.0490 | 2.24e-8 | 1.800 |
| baseline, `f = 0` | 1579 | — | 2.12e-7 | 20.131 |
| nonparametric | — | — | — | 2.385 |
| PINN | 189.1 | — | — | 0.463 |
| cascading | 592.8 | — | — | 0.938 |

For the nonparametric, PINN and cascading rows the test column is in-sample and
is not directly comparable with the first three rows. In the porous convention
of the legacy code the reported `D` is a factor four larger; dividing by four
gives 885 (ours) and 759 (benchmark), consistent with the published 897/833.

The estimated mechanism is close to a logistic: a parabola explains 89% (linear)
and 94% (porous) of its variation, with a peak shifted slightly to the right of
`K/2`. The figure is `figures/realdata1_nonpart_ours.png`.

## Reproducing

```bash
python -I scripts/realdata_cell.py --variant linear
python -I scripts/realdata_cell.py --variant porous
python -I scripts/realdata_ci.py --variant linear
python -I scripts/realdata_ci.py --variant porous
python -I scripts/realdata_mechanism.py --out figures/realdata1_nonpart_ours.png
```

Each driver merges its entry into `runs/realdata_cell/<variant>.json`, so a run
can be restarted with only the missing methods. The proposal's fitted
`state_dict` is written to `runs/realdata_cell/<variant>_ours_model.pt`; the
interval script reads it back and does not refit `D`.
