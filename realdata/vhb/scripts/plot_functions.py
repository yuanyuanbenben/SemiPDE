"""Plot the calibrated constitutive functions of the two model families.

Compares the identified equilibrium values and the 18 identified Maxwell
values per branch of the spline model with the linear (neo-Hookean / Newtonian)
Zener model written onto the same interpolation grid.  The branch count is
read from the interpolation folder, so both the 2- and 3-branch fits work.

Run:  python -I plot_functions.py \
        --spline results/fit_neq_spline/interp_neq \
        --zener  results/fit_neq_zener/interp_neq \
        --mu-eq 0.0187 --out results/figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPLINE = "#0072B2"
ZENER = "#E69F00"
EQ = "#009E73"
EXTRA = ["#D55E00", "#CC79A7", "#56B4E9"]


def read_domain(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline().split(",")[0])
        return np.array([float(f.readline()) for _ in range(n)])


def read_values(path):
    return np.loadtxt(path)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, color="0.9", linewidth=0.6)


def branch_files(d, b):
    return {
        "I1": f"interpolationValues_ISOCHORIC_NONEQSEF_{b}_I1.txt",
        "I2": f"interpolationValues_ISOCHORIC_NONEQSEF_I2_{b}_POLYCONVEX.txt",
        "creep": f"interpolationValues_CREEP_POTENTIAL_I1tau2_MINUS_{b}_3I2tau.txt",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spline", required=True)
    ap.add_argument("--zener", required=True)
    ap.add_argument("--mu-eq", type=float, default=0.0187)
    ap.add_argument("--primary-label", default="spline",
                    help="legend label of the --spline (primary) values")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", action="append", default=[],
                    help="name:interp-dir overlay for the branch panels "
                         "(e.g. nn:results/fit_neq_nn/interp_neq)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    extras = []
    for spec in args.extra:
        name, d = spec.split(":", 1)
        extras.append((name, d))

    nb = 0
    while os.path.exists(os.path.join(
            args.spline, f"interpolationValues_ISOCHORIC_NONEQSEF_{nb}_I1.txt")):
        nb += 1
    n_panels = 2 + 2 * nb
    ncol = 3
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.85 * ncol, 3.1 * nrow),
                             squeeze=False)
    panels = list(axes.flat)

    x1 = read_domain(os.path.join(args.spline,
                                  "interpolationDomain_ISOCHORIC_SEF_I1.txt")) - 3.0
    x2 = read_domain(os.path.join(args.spline,
                                  "interpolationDomain_ISOCHORIC_SEF_I2_POLYCONVEX.txt"))
    w1 = read_values(os.path.join(args.spline, "interpolationValues_ISOCHORIC_SEF_I1.txt"))
    w2 = read_values(os.path.join(args.spline,
                                  "interpolationValues_ISOCHORIC_SEF_I2_POLYCONVEX.txt"))
    ax = panels[0]
    ax.plot(x1, w1, "o-", color=SPLINE, lw=1.6, ms=4,
            label=f"{args.primary_label} values")
    ax.plot(x1, 0.5 * args.mu_eq * x1, color=ZENER, lw=1.8, ls="--",
            label="neo-Hookean reference")
    ax.set_title(r"equilibrium  $W_1(I_1-3)$", fontsize=10)
    ax.set_xlabel(r"$I_1-3$", fontsize=9)
    ax.set_ylabel("W (N/mm$^2$)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    ax = panels[1]
    ax.plot(x2, w2, "o-", color=SPLINE, lw=1.6, ms=4,
            label=f"{args.primary_label} values")
    ax.axhline(0, color=ZENER, lw=1.8, ls="--", label="W$_2$ = 0 reference")
    ax.set_title(r"equilibrium  $W_2(I_{2b,\mathrm{poly}})$", fontsize=10)
    ax.set_xlabel(r"$I_{2b,\mathrm{poly}}$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    for b in range(nb):
        ax = panels[2 + 2 * b]
        xb = read_domain(os.path.join(
            args.spline, f"interpolationDomain_ISOCHORIC_NONEQSEF_{b}_I1.txt")) - 3.0
        names = branch_files(args.spline, b)
        yb = read_values(os.path.join(args.spline, names["I1"]))
        z_names = branch_files(args.zener, b)
        yz = read_values(os.path.join(args.zener, z_names["I1"]))
        ax.plot(xb, yb, "o-", color=SPLINE, lw=1.6, ms=4,
                label=args.primary_label)
        ax.plot(xb, yz, color=ZENER, lw=1.8, ls="--", label="Zener (NH)")
        for k, (name, d) in enumerate(extras):
            xe = read_domain(os.path.join(
                d, f"interpolationDomain_ISOCHORIC_NONEQSEF_{b}_I1.txt")) - 3.0
            ye = read_values(os.path.join(d, branch_files(d, b)["I1"]))
            ax.plot(xe, ye, "s", color=EXTRA[k % len(EXTRA)], ms=4,
                    label=name)
        ax.set_title(f"Maxwell branch {b + 1}  $W(I_1)$", fontsize=10)
        ax.set_xlabel(r"$I_1-3$", fontsize=9)
        ax.set_ylabel("W (N/mm$^2$)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        style(ax)

        ax = panels[2 + 2 * b + 1]
        dom = read_domain(os.path.join(
            args.spline,
            f"interpolationDomain_CREEP_POTENTIAL_I1tau2_MINUS_{b}_3I2tau.txt"))
        yb = read_values(os.path.join(args.spline, names["creep"]))
        yz = read_values(os.path.join(args.zener, z_names["creep"]))
        ax.plot(dom, yb, "o-", color=SPLINE, lw=1.6, ms=4,
                label=args.primary_label)
        ax.plot(dom, yz, color=ZENER, lw=1.8, ls="--", label="Zener (linear)")
        for k, (name, d) in enumerate(extras):
            de = read_domain(os.path.join(
                d,
                f"interpolationDomain_CREEP_POTENTIAL_I1tau2_MINUS_{b}_3I2tau.txt"))
            ye = read_values(os.path.join(d, branch_files(d, b)["creep"]))
            ax.plot(de, ye, "s", color=EXTRA[k % len(EXTRA)], ms=4,
                    label=name)
        ax.set_title(f"creep potential {b + 1}", fontsize=10)
        ax.set_xlabel(r"$I_{1\tau^2}-3I_{2\tau}$", fontsize=9)
        ax.set_ylabel(r"$\Phi$ (N/mm$^2$ s)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        style(ax)

    for ax in panels[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "constitutive_functions.png"), dpi=170)
    plt.close(fig)
    print("wrote", os.path.join(args.out, "constitutive_functions.png"))


if __name__ == "__main__":
    main()
