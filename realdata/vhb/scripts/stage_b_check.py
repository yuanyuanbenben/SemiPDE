"""Stage B checks of the author solver run (guide 14, stage B).

Compares, for one experiment:

* the experimental forces the solver wrote during PREPARE with an independent
  interpolation of the processed force-displacement file at grip strain*15 mm;
* the experimental hole-edge values with the eight points read by our audit
  (as a sorted set, because the solver writes DoFs in its own order);
* the simulated force (times the symmetry factor 4) with the experimental
  force at every stage;
* the simulated displacements with the experimental ones at every stage.

Run:  python -I stage_b_check.py --input <tree>/calibration/input \
        --exp-dir rate_dep/vhb/ldot15p0/plate_d5mm_rep1/msh_6x7.5x0.25_96elm/extrap \
        --data-root <tree>/calibration/data/rate_dep/vhb/ldot15p0/plate_d5mm_rep1 \
        --rate 15 --rep 1 --out results/stage_b

--data-root is the replicate directory that holds ForceDisplacement.txt (it
sits one level above ``extrap/``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np


def sanitized(exp_dir):
    e = exp_dir.rstrip("/")
    return e.replace("/", "_")


def strain_files(dirpath):
    out = {}
    for path in glob.glob(os.path.join(dirpath, "time_*_rank_*.txt")):
        name = os.path.basename(path)
        m = re.match(r"time_([0-9.]+)_rank_(\d+)", name)
        if m:
            out[round(float(m.group(1)), 6)] = path
    return dict(sorted(out.items()))


def read_vector(path):
    with open(path, encoding="utf-8") as f:
        return np.array([float(x) for x in f.read().split()])


def read_scalar(path):
    """Reaction-force file: one nodal value per constrained DoF.

    The author's MATLAB layer sums the column
    (``Concatenate_allDirs(..., sumColumns=true)``) before applying the
    symmetry factor, so the scalar force is the sum, not the first entry.
    """
    return float(np.sum(read_vector(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--symmetry", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sub = sanitized(args.exp_dir)
    f_exp = strain_files(os.path.join(args.input, "forces_experiment", sub))
    f_sim = strain_files(os.path.join(args.input, "forces_simulation", sub))
    d_exp = strain_files(os.path.join(args.input, "displacement_experiment", sub))
    d_sim = strain_files(os.path.join(args.input, "displacement_simulation", sub))

    # independent experimental force at the same grip displacement
    fd = np.genfromtxt(os.path.join(args.data_root, "ForceDisplacement.txt"),
                       delimiter="\t", encoding="utf-8-sig")
    grip = fd[:, 1]
    force = fd[:, 0]

    rows = []
    for strain in sorted(f_exp):
        grip_disp = strain * 15.0
        f_ref = float(np.interp(grip_disp, grip, force))
        f_e = read_scalar(f_exp[strain])
        row = {"strain": strain, "grip_mm": grip_disp,
               "F_experiment_solver": f_e, "F_experiment_reinterp": f_ref,
               "abs_diff_experiment": abs(f_e - f_ref)}
        if strain in f_sim:
            f_s = read_scalar(f_sim[strain]) * args.symmetry
            row["F_simulation_x4"] = f_s
            row["rel_diff"] = (f_s - f_ref) / max(abs(f_ref), 1e-12)
        if strain in d_exp and strain in d_sim:
            ue = np.sort(read_vector(d_exp[strain]))
            us = np.sort(read_vector(d_sim[strain]))
            row["n_points"] = int(ue.size)
            row["max_abs_u_diff_mm"] = float(np.max(np.abs(us - ue)))
            row["max_abs_u_exp_mm"] = float(np.max(np.abs(ue)))
        rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    tag = f"ldot{str(args.rate).replace('.', 'p')}_rep{args.rep}"
    with open(os.path.join(args.out, f"stage_b_{tag}.json"), "w") as f:
        json.dump({"experiment": sub, "stages": rows,
                   "n_stages_force": len(f_exp),
                   "n_stages_disp": len(d_exp)}, f, indent=1)

    fd_ = [r["abs_diff_experiment"] for r in rows]
    fsim = [r.get("rel_diff") for r in rows if "rel_diff" in r]
    ud = [r.get("max_abs_u_diff_mm") for r in rows if "max_abs_u_diff_mm" in r]
    ue_ = [r.get("max_abs_u_exp_mm") for r in rows if "max_abs_u_exp_mm" in r]
    print(f"{tag}: {len(f_exp)} force stages, {len(d_exp)} displacement stages")
    print(f"  experiment file vs own interpolation: max {max(fd_):.2e} N")
    if fsim:
        print(f"  sim*4 vs experiment: mean rel {np.mean(fsim):+.3e}, "
              f"max |rel| {np.max(np.abs(fsim)):.3e}")
    if ud:
        print(f"  u displacement: max |sim-exp| {max(ud):.2e} mm "
              f"(max |exp| {max(ue_):.3f} mm)")


if __name__ == "__main__":
    main()
