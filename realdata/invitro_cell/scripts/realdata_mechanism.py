"""The fitted reaction mechanism of the cell-culture application.

Evaluates ``f_hat`` on the observed density support and compares it with the
logistic drift the benchmarks assume (``lambda u (1 - u/K)``) and with the best
parabola through ``f_hat``.  Prints the coefficients and the shape diagnostics,
and writes the figure that the application section refers to.

    python scripts/realdata_mechanism.py --variant linear
    python scripts/realdata_mechanism.py --variant porous --out pic/x.png

Nothing here touches the fit: both models are read back from the ``state_dict``
that ``scripts/realdata_cell.py`` writes.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semipde.model import ProfilingModel  # noqa: E402
from semipde.realdata import cell_case, cell_split, theta_to_um2h  # noqa: E402

#: The paper's figure colours the linear model red and the porous one blue.
RED, BLUE = "#e34948", "#2a78d6"

#: The window the figure shows, in units of 1e-3 cells.  Both ends of the
#: observed range are left out: near zero the state is the scratch centre and
#: the dynamics are diffusion-dominated, and past 1.5e-3 the estimate is set by
#: a few percent of the points.  The middle is where the mechanism is read off.
X_LO, X_HI = 0.2, 1.5


def evaluate(model, config, values) -> np.ndarray:
    """``f_hat`` at a batch of states, through the model's own feature path.

    The mechanism is pointwise in the state for this spec, so filling the grid
    with the value under test and reading one node gives ``f(v)`` exactly as the
    drift evaluates it.
    """
    nx = config.grid.shape[0]
    u = torch.as_tensor(values, dtype=torch.float64).view(-1, 1, 1)
    u = u.expand(-1, 1, nx).clone()
    z = torch.zeros(u.shape[0], dtype=torch.float64)
    with torch.no_grad():
        return model.mechanism(u, z)[:, 0, 0].numpy()


def logistic_lambda(variant: str, root: str) -> float:
    """The proliferation rate of the fitted Benchmark 1/2, if it was run."""
    path = os.path.join(root, f"{variant}.json")
    if os.path.exists(path):
        with open(path) as fh:
            store = json.load(fh)
        lam = store.get("logistic", {}).get("theta", {}).get("lamda")
        if lam is not None:
            return float(lam)
    return 0.05  # the paper's nominal proliferation rate


def diagnostics(name, model, spec, config, split, lam, K=1.7):
    """Print the coefficients of the parabola, the peak and the tail behaviour.

    ``K`` is the nominal monolayer density in *scaled* units (``1.7e-3 / 1e-3``),
    i.e. the carrying capacity the benchmarks fix.
    """
    # Fit on the states the data actually visit, which is the weighting the
    # application reports; a uniform grid would let the empty tail (v > 2)
    # set the quadratic and hide the shape that matters.
    t = split.train.t.shape[0]
    u = split.train.states.reshape(-1, 1, config.grid.shape[0])
    z = split.train.z.repeat_interleave(t)
    with torch.no_grad():
        f = model.mechanism(u, z).reshape(-1).numpy()
    v = u.reshape(-1).numpy()
    A = np.stack([v, -v * v], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, f, rcond=None)
    quad = A @ np.stack([a, b])
    r2 = 1.0 - np.sum((f - quad) ** 2) / np.sum((f - f.mean()) ** 2)

    # Only the range the data cover is ever plotted: past the largest observed
    # state the network is extrapolating and its shape there is not evidence.
    vmax = float(v.max())
    grid = np.linspace(0.0, vmax, 401)
    fg = evaluate(model, config, grid)
    peak = int(np.argmax(fg))
    q999 = float(np.quantile(split.train.states.numpy(), 0.999))
    above = float((v > K).mean())
    top = float((v > 1.5).mean())

    print(f"\n== {name}  D = {theta_to_um2h(float(model.theta['D'])):.1f} um^2/h")
    print(f"   f(u) = a u - b u^2 at observed states: "
          f"a = {a:.5f}, b = {b:.5f}, K_hat = {a / b:.3f} "
          f"({1000 * a / b:.0f}e-3), R2 = {r2:.4f}")
    print(f"   logistic (Benchmark): lambda = {lam:.5f}, K = {K:.2f} "
          f"({K * 1e3:.0f}e-3)")
    print(f"   peak f = {fg[peak] * 1e3:.2f}e-6 /h at u = {grid[peak] * 1e3:.0f}e-3 "
          f"(logistic peak at u = {K / 2 * 1e3:.0f}e-3)")
    print(f"   f at u/K = 1: {float(evaluate(model, config, [K])[0]) * 1e3:+.2f}e-6"
          f"   (logistic: 0)")
    print(f"   observed states: {vmax * 1e3:.0f}e-3 max, "
          f"{100 * above:.1f}% above K, {100 * top:.1f}% above 1.5e-3, "
          f"99.9% <= {q999 * 1e3:.0f}e-3")
    return {"a": a, "b": b, "r2": r2, "grid": grid, "fg": fg, "vmax": vmax}


def mass_balance(model, config, split, lam, K) -> None:
    """A check of the mechanism that never uses ``D``.

    Under no-flux boundaries the total mass changes at the rate ``int f(u) dx``
    -- the diffusion term integrates to the boundary flux, which is zero.  So
    the observed ``d/dt`` of the spatial mean is a direct measurement of the
    mean of ``f`` over the field, and it tests the *level* of the fitted
    mechanism independently of the diffusivity it is paired with.
    """
    v = split.train.states                       # (n, T, 1, n_x)
    dt = float(split.train.t[1] - split.train.t[0])
    dM = ((v[:, 1:] - v[:, :-1]).mean(-1) / dt).reshape(-1).numpy()
    mid = 0.5 * (v[:, :-1] + v[:, 1:])
    flat = mid.reshape(-1, 1, config.grid.shape[0])
    z = torch.zeros(flat.shape[0], dtype=torch.float64)
    with torch.no_grad():
        f = model.mechanism(flat, z)[:, 0].mean(-1).numpy()
    u = mid.reshape(-1, config.grid.shape[0]).mean(-1).numpy()
    lg = lam * u * (1.0 - u / K)
    print("   mass balance (d/dt of the spatial mean, no use of D):")
    print(f"      observed mean {dM.mean() * 1e3:6.2f}e-6/h; "
          f"f_hat {f.mean() * 1e3:6.2f}, bias {np.mean(f - dM) * 1e3:+.2f}, "
          f"rms {np.sqrt(np.mean((f - dM) ** 2)) * 1e3:.2f}, "
          f"corr {np.corrcoef(f, dM)[0, 1]:.2f}")
    print(f"      logistic     {lg.mean() * 1e3:6.2f}, "
          f"bias {np.mean(lg - dM) * 1e3:+.2f}, "
          f"rms {np.sqrt(np.mean((lg - dM) ** 2)) * 1e3:.2f}, "
          f"corr {np.corrcoef(lg, dM)[0, 1]:.2f}")


def plot(panels, out):
    """One axes, four curves, laid out the way the paper's figure is.

    The paper compares "the fitted mechanism" with "the mechanism the
    benchmarks assume" in a single panel -- solid for SemiPDE 1/2, dashed for
    Benchmark 1/2, red for the linear operator and blue for the porous one --
    and that is what this reproduces.  Everything is in the data's own units:
    density on the x axis, density per hour on the y axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    for p in panels:
        color, idx = (RED, "1") if p["variant"] == "linear" else (BLUE, "2")
        grid, fg, lam, K = p["grid"], p["fg"], p["lam"], p["K"]
        u = grid * 1e-3            # state -> density
        ax.plot(u, fg * 1e-3, color=color, lw=2, label=f"SemiPDE {idx}")
        # ``K`` is in units of 1e-3 cells, so the logistic curve, too, is
        # evaluated in the data's units.
        ax.plot(u, lam * u * (1.0 - u / (K * 1e-3)), color=color, lw=1.6,
                ls="--", label=f"Benchmark {idx}")
    ax.set_xlim(X_LO * 1e-3, X_HI * 1e-3)
    ax.set_ylim(0.0, 3.0e-5)
    ax.set_xlabel("density u")
    ax.set_ylabel("reaction value f(u)")
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.savefig(out, dpi=200)
    print(f"\nfigure written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["linear", "porous"],
                    choices=("linear", "porous"))
    ap.add_argument("--root", default="runs/realdata_cell")
    ap.add_argument("--out", default="pic/realdata1_nonpart_ours.png")
    args = ap.parse_args()

    panels = []
    for variant in args.variants:
        spec, config = cell_case(variant=variant, ode_step=0.25,
                                 init="gaussian_zero_out")
        split = cell_split()
        model = ProfilingModel(spec, config, seed=0)
        model.load_state_dict(torch.load(
            os.path.join(args.root, f"{variant}_ours_model.pt"),
            map_location="cpu", weights_only=True))
        model.eval()
        lam = logistic_lambda(variant, args.root)
        label = "SemiPDE 1 (ours)" if variant == "linear" else "SemiPDE 2 (ours)"
        title = ("Linear operator (SemiPDE 1)" if variant == "linear"
                 else "Porous operator (SemiPDE 2)")
        p = diagnostics(label, model, spec, config, split, lam)
        mass_balance(model, config, split, lam, 1.7)
        p.update({"lam": lam, "K": 1.7, "variant": variant,
                  "ours_label": label, "title": title})
        panels.append(p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(panels, args.out)


if __name__ == "__main__":
    main()
