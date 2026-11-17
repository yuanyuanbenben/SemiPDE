"""Figures for the spline vs Zener comparison on the VHB multi-rate data.

Reads the per-rate curves saved by collect_eval.py and writes

  * force_strain.png   -- experiment vs both models, one panel per rate
  * force_error.png    -- force relative error vs strain, one panel per rate
  * functions.png      -- fitted equilibrium and Maxwell functions

Run:  python -I plot_comparison.py --spline results/eval_spline \
        --zener results/eval_zener --rates ldot0p15 ldot1p5 ldot3p75 \
        ldot7p5 ldot15p0 --out results/figures

Any number of models can be compared with ``--series name:dir`` (repeat the
flag); ``--spline``/``--zener`` are then unnecessary.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colour-blind safe palette
EXP = "#000000"
SPLINE = "#0072B2"
ZENER = "#E69F00"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
STYLES = ["-", "--", "-.", ":"]

LABEL = {"ldot0p15": "0.15 mm/s", "ldot1p5": "1.5 mm/s",
         "ldot3p75": "3.75 mm/s", "ldot7p5": "7.5 mm/s",
         "ldot15p0": "15 mm/s"}
SPLIT = {"ldot0p15": "train", "ldot1p5": "train", "ldot15p0": "train",
         "ldot3p75": "holdout", "ldot7p5": "holdout"}


def load(directory, rate):
    return np.load(os.path.join(directory, f"curves_{rate}.npz"))


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True, color="0.9", linewidth=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spline", default=None)
    ap.add_argument("--zener", default=None)
    ap.add_argument("--series", action="append", default=None,
                    help="model comparison as name:directory (repeat the flag)")
    ap.add_argument("--rates", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.series:
        pairs = [s.split(":", 1) for s in args.series]
    else:
        if not (args.spline and args.zener):
            raise SystemExit("give --series or both --spline and --zener")
        pairs = [("spline", args.spline), ("Zener", args.zener)]
    series = [(name, d, PALETTE[i % len(PALETTE)], STYLES[i % len(STYLES)])
              for i, (name, d) in enumerate(pairs)]

    ncol = 3
    nrow = int(np.ceil(len(args.rates) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.1 * nrow),
                             sharex=True, squeeze=False)
    for ax, rate in zip(axes.flat, args.rates):
        d = load(series[0][1], rate)
        ax.plot(d["strain"], d["f_exp"], color=EXP, lw=2.4, label="experiment")
        for name, directory, color, ls in series:
            m = load(directory, rate)
            ax.plot(m["strain"], 4 * m["f_sim"], color=color, lw=2.0, ls=ls,
                    label=name)
        ax.set_title(f"{LABEL[rate]}  ({SPLIT[rate]})", fontsize=10)
        style(ax)
    axes.flat[0].legend(frameon=False, fontsize=8.5)
    for ax in axes.flat[len(args.rates):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("nominal strain", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("force (N)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "force_strain.png"), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.0 * nrow),
                             sharex=True, squeeze=False)
    for ax, rate in zip(axes.flat, args.rates):
        d = load(series[0][1], rate)
        ax.axhline(0, color="0.75", lw=0.8)
        for name, directory, color, ls in series:
            m = load(directory, rate)
            ax.plot(m["strain"], 100 * (4 * m["f_sim"] - m["f_exp"]) / m["f_exp"],
                    color=color, lw=2.0, ls=ls, label=name)
        ax.set_title(f"{LABEL[rate]}  ({SPLIT[rate]})", fontsize=10)
        ax.set_ylim(-45, 45)
        style(ax)
    axes.flat[0].legend(frameon=False, fontsize=8.5)
    for ax in axes.flat[len(args.rates):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("nominal strain", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("force error (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "force_error.png"), dpi=170)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
