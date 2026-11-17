"""Algorithm 2 on an already-fitted cell-culture model: the 95% Wald interval.

    python scripts/realdata_ci.py --variant linear
    python scripts/realdata_ci.py --variant linear --inner-inputs location,state

The fit itself costs most of an hour, while the interval is a handful of
rollouts plus one closed-form solve, so the driver writes the fitted
``state_dict`` and this script reads it back.  It can be run repeatedly with
different inner classes without touching ``theta_hat`` -- ``h_j`` never enters
the estimator.

The default inner class is the paper's ``S_n`` (``matched``: ``h_j`` reads what
the fitted ``f_hat`` reads).  Widening it to include ``derivatives`` hands the
inner problem the Laplacian, which is exactly what ``D`` multiplies, and the
interval degenerates -- see ``add_interval`` in ``scripts/realdata_cell.py``.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semipde.inference import estimate_covariance  # noqa: E402
from semipde.model import ProfilingModel  # noqa: E402
from semipde.realdata import cell_case, cell_split, theta_to_um2h  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("linear", "porous"), default="linear")
    ap.add_argument("--model", default=None,
                    help="fitted state_dict (default "
                         "runs/realdata_cell/<variant>_ours_model.pt)")
    ap.add_argument("--init", default="gaussian_zero_out")
    ap.add_argument("--inner-inputs", nargs="+", default=["matched"])
    ap.add_argument("--basis-degree", type=int, default=2)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--ode-step", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="merge the interval into this results JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    path = args.model or f"runs/realdata_cell/{args.variant}_ours_model.pt"
    spec, config = cell_case(variant=args.variant, ode_step=args.ode_step,
                             init=args.init)
    split = cell_split()
    model = ProfilingModel(spec, config, seed=args.seed)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    theta = {k: float(v) for k, v in model.theta_detached().items()}
    print(f"loaded {path}: theta={theta}  "
          f"D={theta_to_um2h(theta['D']):.1f} um^2/h")

    head = f"{'inner class':<28} {'se':>10} {'95% CI (um^2/h)':>28} " \
           f"{'n_est':>7} {'sigma_hat':>10}"
    print("\n" + head)
    print("-" * len(head))
    last = None
    for inner in args.inner_inputs:
        t0 = time.time()
        spec.set_inner_inputs(inner)
        inf = estimate_covariance(
            model, split.train,
            inner_solve="exact", basis_degree=args.basis_degree,
            lambda_grid=[args.ridge], n_est=config.n_total,
            seed=args.seed, verbose=args.verbose,
        )
        spec.set_inner_inputs(None)
        iv = inf.interval(alpha=0.05)
        j = list(inf.names).index("D")
        se = float(iv["se"][j])
        lo, hi = float(iv["lower"][j]), float(iv["upper"][j])
        print(f"{inner:<28} {theta_to_um2h(se):>10.1f} "
              f"{'[' + format(theta_to_um2h(lo), '.0f') + ', ' + format(theta_to_um2h(hi), '.0f') + ']':>28} "
              f"{int(inf.n_est):>7} {float(inf.sigma):>10.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        last = {
            "inner_inputs": inner,
            "names": list(inf.names),
            "estimate": [float(x) for x in iv["estimate"]],
            "se": [float(x) for x in iv["se"]],
            "lower": [float(x) for x in iv["lower"]],
            "upper": [float(x) for x in iv["upper"]],
            "alpha": 0.05,
            "n_est": int(inf.n_est),
            "sigma_hat": float(inf.sigma),
            "basis_degree": int(args.basis_degree),
            "ridge": float(args.ridge),
        }

    if args.out and last is not None:
        store = {}
        if os.path.exists(args.out):
            with open(args.out) as fh:
                store = json.load(fh)
        store.setdefault("ours", {})["interval"] = last
        with open(args.out, "w") as fh:
            json.dump(store, fh, indent=1)
        print(f"\ninterval written to {args.out} (inner class {last['inner_inputs']})")


if __name__ == "__main__":
    main()
