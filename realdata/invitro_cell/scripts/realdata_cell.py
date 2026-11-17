"""The cell-culture application: ours and the four simulation baselines.

Fits the scratch-assay data of :mod:`semipde.realdata` with the proposal and
with the same four baselines the simulation tables use, and forms the 95% Wald
interval for the proposal's diffusivity.

    python scripts/realdata_cell.py --variant linear
    python scripts/realdata_cell.py --variant porous --methods ours para

Everything is written to ``runs/realdata_cell/<variant>.json``, one entry per
method, merged on rewrite: a run that is interrupted and restarted with the
methods it still needs picks up where it stopped.  The console table is printed
from the merged file, so it is always the whole experiment that is shown.

Units
-----
The fits work in the module's dimensionless units (length unit 50 um, time in
hours, state divided by 1e-3).  The table converts ``D`` to the paper's
``um^2/h`` and the mean squared errors to the raw cell-density scale, so the
columns are comparable with the published application table.
"""

from __future__ import annotations

import os
import sys

# Single-threaded BLAS: the host is shared, and a 12-curve rollout on a 38-point
# grid does not benefit from threading beyond the overhead.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semipde.baselines import fit_baseline  # noqa: E402
from semipde.inference import estimate_covariance  # noqa: E402
from semipde.model import ProfilingModel  # noqa: E402
from semipde.realdata import (  # noqa: E402
    STATE_SCALE,
    LogisticCell,
    cell_case,
    cell_split,
    theta_to_um2h,
)
from semipde.train import train  # noqa: E402

METHODS = ("ours", "logistic", "para", "nonpara", "pinn", "cascading")


def predict(model, curves):
    with torch.no_grad():
        return model(curves.initial, curves.t, curves.z)


def errors(model, curves) -> float:
    """Mean squared error in the scaled units the data are stored in.

    ``nan`` when the model cannot answer for these curves: the baselines hand
    back the prediction they were built on and ignore the arguments (Baseline 2
    is one regression per held-out curve; Baseline 3 and 4 refit the held-out
    solutions when they are constructed), so only the held-out column is defined
    for them.
    """
    pred = predict(model, curves)
    if pred.shape != curves.states.shape:
        return float("nan")
    return float((pred - curves.states).pow(2).mean())


def summary(model, split) -> dict:
    mse_train = errors(model, split.train)
    mse_test = errors(model, split.test)
    return {
        "train_mse": mse_train,
        "test_mse": mse_test,
        # The paper's scale: densities, not the 1e-3-rescaled states.
        "train_mse_raw": mse_train * STATE_SCALE ** 2,
        "test_mse_raw": mse_test * STATE_SCALE ** 2,
        "test_rmse_raw": math.sqrt(mse_test) * STATE_SCALE,
        "theta": {},
    }


def theta_of(model) -> dict:
    try:
        return {k: float(v) for k, v in model.theta_detached().items()}
    except AttributeError:  # pragma: no cover - BaselineModel always has one
        return {}


def add_interval(out, model, spec, config, split, args) -> None:
    """Algorithm 2 on the training curves, one Wald interval per coordinate.

    The default inner class is the **matched** one -- ``h_j`` reads exactly what
    the fitted ``f_hat`` reads (the state), which is the paper's ``S_n``.  The
    widened class (``location,state,derivatives``) that the size study
    calibrates on the simulation is *not* usable here: ``derivatives`` hands
    ``h_j`` the Laplacian, which is precisely what ``D`` multiplies, so the
    inner problem can undo the ``D`` perturbation to ``1e-8`` of the trajectory
    and ``M_hat`` -- the part of ``d_theta u`` no change of ``f`` can reproduce
    -- collapses.  Measured on this case: ``se`` goes from a few hundred
    ``um^2/h`` to ``6623`` and the interval to ``[-11554, 14408]``.  Interest is
    in ``theta``, not in a nuisance direction ``h_j`` is allowed to invent, so
    the paper's construction it is.  ``--inner-inputs`` still overrides.
    """
    spec.set_inner_inputs(args.inner_inputs)
    inf = estimate_covariance(
        model, split.train,
        inner_solve="exact", basis_degree=args.basis_degree,
        lambda_grid=[args.ridge], n_est=config.n_total,
        seed=args.seed, verbose=args.verbose,
    )
    spec.set_inner_inputs(None)
    iv = inf.interval(alpha=0.05)
    out["interval"] = {
        "names": list(inf.names),
        "estimate": [float(x) for x in iv["estimate"]],
        "se": [float(x) for x in iv["se"]],
        "lower": [float(x) for x in iv["lower"]],
        "upper": [float(x) for x in iv["upper"]],
        "alpha": 0.05,
        "n_est": int(inf.n_est),
        "sigma_hat": float(inf.sigma),
    }


def run_ours(spec, config, split, args) -> dict:
    model = ProfilingModel(spec, config, seed=args.seed)
    t0 = time.time()
    result = train(model, split.train, split.val, config=config,
                   epochs=args.epochs, lr=args.lr, seed=args.seed,
                   lr_final=args.lr_final, grad_clip=args.grad_clip,
                   verbose=args.verbose)
    out = summary(model, split)
    out["theta"] = theta_of(model)
    out["seconds"] = time.time() - t0
    out["best_epoch"] = int(getattr(result, "best_epoch", -1))

    # The fitted network is worth keeping: the interval is a separate, cheap
    # post-processing step (``scripts/realdata_ci.py``), and re-deriving it must
    # not mean re-fitting for an hour.
    path = args.save_model or f"runs/realdata_cell/{args.variant}_ours_model.pt"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    out["model_path"] = path
    if not args.skip_interval:
        add_interval(out, model, spec, config, split, args)
    # The fitted mechanism at the observed states.  Evaluating it there rather
    # than on a probe grid keeps the drift/feature plumbing of the model in
    # charge of the inputs, which is what ``ProfilingModel.mechanism`` is for.
    t = split.train.t.shape[0]
    u = split.train.states.reshape(-1, 1, *config.grid.shape)
    z = split.train.z.repeat_interleave(t)
    with torch.no_grad():
        f = model.mechanism(u, z).reshape(-1)
    v = u.reshape(-1)
    lo, hi = float(v.min()), float(v.max())
    # Fisher--KPP reads f = a v - b v^2; the two coefficients are what the
    # report quotes, together with the implied carrying capacity.
    A = torch.stack([v, -v * v], dim=1)
    coef = torch.linalg.lstsq(A, f).solution
    out["mechanism"] = {
        "growth_a": float(coef[0]), "crowding_b": float(coef[1]),
        "K_hat": float(coef[0] / coef[1]) if float(coef[1]) > 0 else float("nan"),
        "f_rms": float(f.pow(2).mean().sqrt()),
        "v_lo": lo, "v_hi": hi,
    }
    return out


def run_logistic(spec, config, split, args) -> dict:
    """The paper's Benchmark 1: the diffusion operator with a *known* logistic.

    ``D`` and ``lambda`` are estimated by the same profiling loop as the
    proposal -- the mechanism is off, and the logistic enters the drift with
    its own parameter -- so the row isolates "unknown f" against "the growth law
    the benchmark assumes".  This is the row the published application table
    reports; the ``para`` row (``f = 0``) is the simulation's Baseline 1 and is
    a different, harder specification.
    """
    lspec = LogisticCell(spec.grid, variant=spec.variant, scale=spec.scale)
    model = ProfilingModel(lspec, config, use_f=False, seed=args.seed)
    t0 = time.time()
    train(model, split.train, split.val, config=config,
          epochs=args.epochs, lr=args.lr, seed=args.seed,
          lr_final=args.lr_final, grad_clip=args.grad_clip,
          verbose=args.verbose)
    out = summary(model, split)
    out["theta"] = theta_of(model)
    out["seconds"] = time.time() - t0
    # No Algorithm-2 interval: ``h_j`` perturbs the *fitted mechanism*, and
    # this model has none (its growth law is parametric, so ``M_hat`` would be
    # the sensitivity of a function that is not being estimated).
    return out


def run_baseline(method, spec, config, split, args) -> dict:
    t0 = time.time()
    model, _ = fit_baseline(
        method, spec, config, split,
        epochs=args.baseline_epochs, lr=args.lr, seed=args.seed,
        theta_init=config.theta_init,
        cascading_iters=args.baseline_epochs,
        verbose=args.verbose,
    )
    out = summary(model, split)
    out["theta"] = theta_of(model)
    out["seconds"] = time.time() - t0
    return out


def print_table(store: dict) -> None:
    head = (f"{'method':<10} {'D (um^2/h)':>12} {'lam (1/h)':>8} {'se':>9} "
            f"{'95% CI':>21} {'train MSE x1e8':>15} {'test MSE x1e8':>14}")
    print("\n" + head)
    print("-" * len(head))
    for m in METHODS:
        if m not in store:
            continue
        rec = store[m]
        D = rec["theta"].get("D")
        dtxt = "--" if D is None or D != D else f"{theta_to_um2h(D):.1f}"
        lam = rec["theta"].get("lamda")
        ltxt = "  --  " if lam is None or lam != lam else f"{lam:.4f}"
        if "interval" in rec:
            iv = rec["interval"]
            j = iv["names"].index("D")
            se = f"{theta_to_um2h(iv['se'][j]):.1f}"
            ci = (f"[{theta_to_um2h(iv['lower'][j]):.0f}, "
                  f"{theta_to_um2h(iv['upper'][j]):.0f}]")
        else:
            se, ci = " ", " "
        tr = rec["train_mse_raw"]
        trtxt = "  --  " if tr != tr else f"{tr * 1e8:.3f}"
        print(f"{m:<10} {dtxt:>12} {ltxt:>8} {se:>9} {ci:>21} "
              f"{trtxt:>15} "
              f"{rec['test_mse_raw'] * 1e8:>14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("linear", "porous"), default="linear")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--epochs", type=int, default=400,
                    help="budget of the proposal's fit")
    ap.add_argument("--baseline-epochs", type=int, default=400,
                    help="budget of the parametric and PINN baselines (and of "
                         "cascading's inner alternation)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=None,
                    help="cosine-anneal the learning rate to this value "
                         "(without it the late fit is free to blow up, which "
                         "is what the 400-epoch calibration did at ~epoch 149)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="clip the gradient norm to this value")
    ap.add_argument("--ode-step", type=float, default=0.25,
                    help="rollout step in hours.  The diffusion eigenvalue is "
                         "4*D/dx^2 = 4*D here, so RK4 (real-axis limit 2.78) "
                         "is stable for D < 0.70/ode_step")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="tilde lambda_n of Algorithm 2's inner fit")
    ap.add_argument("--basis-degree", type=int, default=2)
    ap.add_argument("--inner-inputs", default="matched",
                    help="inner class of Algorithm 2: 'matched' (the paper's "
                         "S_n) or a combination of location,state,derivatives "
                         "(widened -- see add_interval)")
    ap.add_argument("--init", default="gaussian",
                    help="network initialisation; 'gaussian_zero_out' starts "
                         "the mechanism (and its slope in u) at exactly zero")
    ap.add_argument("--skip-interval", action="store_true",
                    help="train only; the interval can be added later from the "
                         "saved model with scripts/realdata_ci.py")
    ap.add_argument("--save-model", default=None,
                    help="where to write the fitted state_dict (default "
                         "runs/realdata_cell/<variant>_<method>_model.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    spec, config = cell_case(variant=args.variant, ode_step=args.ode_step,
                             epochs=args.epochs, init=args.init)
    split = cell_split()
    print(config.size_table())

    out = args.out or f"runs/realdata_cell/{args.variant}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    store = {}
    if os.path.exists(out):
        with open(out) as fh:
            store = json.load(fh)

    for method in args.methods:
        t0 = time.time()
        print(f"\n== {method} ==", flush=True)
        if method == "ours":
            rec = run_ours(spec, config, split, args)
        elif method == "logistic":
            rec = run_logistic(spec, config, split, args)
        else:
            rec = run_baseline(method, spec, config, split, args)
        store[method] = {k: v for k, v in rec.items() if k != "mechanism"}
        if "mechanism" in rec:
            store[f"{method}_mechanism"] = rec["mechanism"]
        with open(out, "w") as fh:
            json.dump(store, fh, indent=1)
        print(f"   {method}: theta={rec['theta']} "
              f"test MSE x1e8 = {rec['test_mse_raw'] * 1e8:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    print_table(store)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
