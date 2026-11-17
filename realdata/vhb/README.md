# Multi-rate viscoelasticity of VHB 4905

Calibration of the constitutive response of VHB 4905 on the authors'
three-dimensional finite-element model. The unknown nonparametric part (the
interpolated equilibrium and Maxwell functions) is represented by a neural
network; the parametric Zener chain is the baseline.

## Data and protocol

The experimental data are the six-grip-speed uniaxial tension curves deposited at
Zenodo record [17019463](https://doi.org/10.5281/zenodo.17019463); the
finite-element model and its calibration data are from DAVIS 1.0.0
(Wiesheier, Moreno-Mateos, Steinmann; Zenodo
[10.5281/zenodo.17542379](https://doi.org/10.5281/zenodo.17542379)). Each run has
53 loading stages up to nominal strain 5.3; repetition 1 is used.

| rate (mm/s) | role |
|---|---|
| 0.015 | equilibrium (time-independent) |
| 0.15, 1.5, 15 | training |
| 3.75 | validation (model selection) |
| 7.5 | test (reported once, never used for selection) |

Observations are the sum of the nodal reactions on the constrained top face and
the vertical displacement at eight hole-edge points on the 96-cell mesh.

## Model

Each Maxwell branch carries an equilibrium energy `W_1(I_1) + W_2(I_2b,poly)`
and a creep potential `Phi(I_1(tau^2) - 3 I_2(tau))`, all evaluated on the
solver's interpolation nodes. The neural network replaces only these node
values:

- one tanh MLP (width 8) per branch and function, 225 weights for three branches;
- output `node = base + scale * (softplus(MLP) - softplus(0))`, clamped at zero;
- the last layer is zero-initialised, so the initial network is bit-identical to
  the converged Zener solution. This warm start is necessary: the solver loses
  the timestep when the branch-0 `W_1` function is perturbed by about 0.2%.

Calibration is Levenberg–Marquardt on the authors' weighted residuals
(`10/max|F|` on force, `1/max|u|` on displacement) using the solver's analytic
sensitivities chained through the network by autograd. The reported three-branch
network is the checkpoint with the lowest validation cost; the test rate is not
used.

The equilibrium response is the released `vhb_eq` values scaled by 0.25 and is
shared by both models; only the Maxwell branches are calibrated. The Zener
baseline uses neo-Hookean `W(I_1)` and linear creep potentials. All moduli are
quoted under the authors' force-symmetry factor 4.

## Results

Five-rate weighted objective (sum over the five non-equilibrium rates; lower is
better):

| model | parameters | total | 0.15 | 1.5 | 3.75 | 7.5 | 15 |
|---|---|---|---|---|---|---|---|
| **SemiPDE (NN, 3 branches)** | 225 | **86.1** | 2.1 | 7.8 | 63.0 | 6.5 | 6.6 |
| SemiPDE (NN, 2 branches) | 150 | 132.0 | 8.3 | 16.1 | 82.7 | 17.0 | 7.9 |
| Zener, 3 branches | 6 | 241.9 | 12.3 | 47.4 | 105.8 | 43.7 | 32.7 |
| Zener, 2 branches | 4 | 288.2 | 14.0 | 52.2 | 110.7 | 56.9 | 54.4 |
| released interpolation (uncalibrated) | — | 2582.8 | — | — | — | — | — |

The three-branch network is lower than both Zener baselines at all five rates and
by a factor 2.8 / 3.3 on the total objective. On the held-out rates the signed
mean force error and the hole-edge displacement are:

| model | 3.75 force mean / max | 3.75 u (mm) | 7.5 force mean / max | 7.5 u (mm) |
|---|---|---|---|---|
| **SemiPDE (NN, 3 branches)** | −17.1% / 43.5% | **1.72** | −6.6% / 46.8% | **0.86** |
| Zener, 3 branches | −14.8% / 29.9% | 2.96 | −7.2% / 22.2% | 2.09 |
| Zener, 2 branches | −16.1% / 33.9% | 2.79 | −9.8% / 36.6% | 1.91 |

The weighted objective and the displacement clearly favour the network. The
largest per-stage force error at 3.75 mm/s is larger for the network than for the
three-branch Zener baseline; this is visible in `results/figures/force_error.png`.

Function shapes and the force–strain comparison are in
`results/figures/constitutive_functions.png`, `force_strain.png` and
`force_error.png`.

## Contents

```
scripts/                 data audit, run-tree setup, calibration, evaluation, plots
results/nn3/             adopted model: metrics, theta.npy, nn_config.json,
                         interp_neq/, per-rate curves
results/nn2/             two-branch network
results/zener3/, zener2/ Zener baselines (same metric and curve format)
results/released/        uncalibrated released values, evaluated
results/comparison.csv   per-rate force/displacement errors for all models
results/figures/         paper figures
```

`theta.npy` holds the network weights; `nn_config.json` holds the architecture,
the Zener base and the node scales needed to rebuild the interpolated values.

## Reproducing

The finite-element solver and the experimental data are not redistributed here.
Download DAVIS 1.0.0 from Zenodo, build the `platewithhole` variant with
deal.II 9.6, and then, from this directory:

```bash
python -I scripts/prepare_author_run.py \
    --source <path/to/davis-viscoelasticity-1.0.0> --dest runs/author_tree

FIT3=runs/author_tree/fe_solver/input/fit3
python -I scripts/setup_fit_tree.py --tree runs/author_tree --fit-dir "$FIT3"
python -I scripts/setup_hybrid_interp.py --tree runs/author_tree \
    --fit-dir "$FIT3" --eq-scale 0.25
python -I scripts/setup_extra_branch.py --fit-dir "$FIT3" \
    --nbranches 3 --mu 0.03 --c 0.292

# Zener baseline (warm start for the network)
python -I scripts/fit_vhb.py --mode neq --model zener --tree runs/author_tree \
    --fit-dir "$FIT3" --iters 40 --jobs 3 --nbranches 3 \
    --init-zener 0.0374449422,0.071747,0.03,0.0249168939,1.1152242261,0.292 \
    --out results/fit_zener3

# three-branch network with validation checkpoint selection
python -I scripts/fit_vhb.py --mode neq --model nn --tree runs/author_tree \
    --fit-dir "$FIT3" --iters 60 --jobs 3 --nbranches 3 --max-rel-step 5 \
    --nn-width 8 --nn-val-rate ldot3p75 --nn-val-every 1 --nn-patience 8 \
    --init-zener 0.005001,0.285059,0.051517,0.017075,5.16036,0.072203 \
    --out results/fit_nn3
```

The complete five-rate evaluation, curve collection and plotting commands are
listed in the top-level `README.md` under *Real-data experiment 2*: they use
`fit_vhb.py --iters 0`, `collect_eval.py`, `per_rate_cost.py`,
`plot_comparison.py` and `plot_functions.py`. The fits write their identified
node values into the live tree, so snapshot `$FIT3/interp_neq` after each fit if
the function plots are needed.

> Two fits or evaluations must not run concurrently on the same
> `runs/author_tree`: the solver input files are keyed by experiment name and
> are overwritten across runs.
