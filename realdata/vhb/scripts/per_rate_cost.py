"""Per-rate weighted cost from the curves written by collect_eval.py.

The objective is the authors' weighted residual
``[wf (4 F_sim - F_exp); wu (u_sim - u_exp)]`` with ``wf = 10 / max|F_exp|``
and ``wu = 1 / max|u_exp|``; the cost of a rate is the squared norm of its
block, and the five-rate cost is the sum over rates.
"""
import argparse
import glob
import json
import os

import numpy as np


def rate_cost(path):
    z = np.load(path)
    fe, fs = z["f_exp"], z["f_sim"]
    ue, us = z["u_exp"], z["u_sim"]
    wf = 10.0 / np.max(np.abs(fe))
    wu = 1.0 / np.max(np.abs(ue))
    force = float(np.sum((wf * (4.0 * fs - fe)) ** 2))
    disp = float(np.sum((wu * (us - ue)) ** 2))
    return force + disp, force, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True,
                    help="evaluation directories (each holding curves_<rate>.npz)")
    ap.add_argument("--out", default=None, help="optional JSON summary")
    args = ap.parse_args()

    doc = {}
    for d in args.eval:
        per = {}
        total = 0.0
        for path in sorted(glob.glob(os.path.join(d, "curves_*.npz"))):
            rate = os.path.basename(path)[len("curves_"):-len(".npz")]
            c, cf, cu = rate_cost(path)
            per[rate] = {"cost": c, "force": cf, "u": cu}
            total += c
        doc[d] = {"cost": total, "per_rate": per}
        parts = " ".join(f"{k} {v['cost']:.2f}" for k, v in per.items())
        print(f"{d}: total {total:.2f} | {parts}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1)


if __name__ == "__main__":
    main()
