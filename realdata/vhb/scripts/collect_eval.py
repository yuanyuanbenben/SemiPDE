"""Snapshot the forward-simulation outputs of one rate into a compact .npz.

The solver writes per-stage files under CALIBRATION_DIR/input/<kind>/<exp>/ and
the next model evaluation overwrites them, so the curves are copied right after
an evaluation.  Forces are the summed nodal reactions (x4 symmetry convention
applied where noted).

Run:  python -I collect_eval.py --input runs/author_tree/calibration/input \
        --fit-dir runs/author_tree/fe_solver/input/fit --rate ldot7p5 \
        --out results/eval_spline/curves_ldot7p5.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np


def sanitized(name):
    return name.rstrip("/").replace("/", "_")


def read_vec(path):
    with open(path, encoding="utf-8") as f:
        return np.array([float(x) for x in f.read().split()])


def stage_files(dirpath, axis=False):
    out = {}
    for path in glob.glob(os.path.join(dirpath, "time_*_rank_0*")):
        name = os.path.basename(path)
        m = re.match(r"time_([0-9.]+)_rank_0" + ("_axis_0" if axis else "") + r"\.txt$", name)
        if m:
            out[round(float(m.group(1)), 6)] = path
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--rate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.fit_dir, "experiments.json")) as f:
        table = json.load(f)
    sub = sanitized(table[args.rate]["name"])

    fe = stage_files(os.path.join(args.input, "forces_experiment", sub), axis=True)
    fs = stage_files(os.path.join(args.input, "forces_simulation", sub), axis=True)
    ue = stage_files(os.path.join(args.input, "displacement_experiment", sub))
    us = stage_files(os.path.join(args.input, "displacement_simulation", sub))

    strain = np.array(sorted(set(fe) & set(fs)))
    u_strain = np.array(sorted(set(ue) & set(us)))
    f_exp = np.array([read_vec(fe[s]).sum() for s in strain])
    f_sim = np.array([read_vec(fs[s]).sum() for s in strain])
    u_exp = np.vstack([read_vec(ue[s]) for s in u_strain])
    u_sim = np.vstack([read_vec(us[s]) for s in u_strain])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, strain=strain, f_exp=f_exp, f_sim=f_sim,
             u_strain=u_strain, u_exp=u_exp, u_sim=u_sim)
    print(f"{args.rate}: {len(strain)} force stages, {len(u_strain)} disp stages, "
          f"{u_exp.shape[1]} points -> {args.out}")


if __name__ == "__main__":
    main()
