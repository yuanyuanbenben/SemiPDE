"""Assemble the viscoelastic interpolation folder used by the calibration runs.

The author material file reads the equilibrium and the Maxwell interpolants
from one folder, but the released ``vhb_eq`` and ``vhb_neq_creep`` folders use
different equilibrium parameterisations (different ``I2b_POLYCONVEX`` domains,
different values).  This script builds ``<fit-dir>/interp_neq`` as

  * equilibrium values  = released ``vhb_eq`` values scaled by --eq-scale
    (0.25 = the x4 symmetry-factor resolution of stage B);
  * equilibrium domains = the ``vhb_eq`` domains;
  * Maxwell values/domains and both properties JSON files = released
    ``vhb_neq_creep`` (equilibrium identification stays off, so the
    viscoelastic sensitivity layout remains 42 columns).

Every non-equilibrium file is overwritten from the released folder, so the
script also resets branch values mutated by earlier trial runs.

Run:  python -I setup_hybrid_interp.py --tree runs/author_tree \
        --fit-dir runs/author_tree/fe_solver/input/fit --eq-scale 0.25
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

EQ_FILES = {
    "I1": ("interpolationValues_ISOCHORIC_SEF_I1.txt",
           "interpolationDomain_ISOCHORIC_SEF_I1.txt"),
    "I2": ("interpolationValues_ISOCHORIC_SEF_I2_POLYCONVEX.txt",
           "interpolationDomain_ISOCHORIC_SEF_I2_POLYCONVEX.txt"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--eq-scale", type=float, default=0.25)
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    src = os.path.join(tree, "fe_solver", "input", "interpolation")
    str_eq = os.path.join(src, "vhb_eq")
    str_neq = os.path.join(src, "vhb_neq_creep")
    dst = os.path.join(os.path.abspath(args.fit_dir), "interp_neq")
    if not os.path.isdir(dst):
        raise SystemExit(f"missing {dst}; run setup_fit_tree.py first")

    for kind, (vname, dname) in EQ_FILES.items():
        values = np.loadtxt(os.path.join(str_eq, vname)) * args.eq_scale
        np.savetxt(os.path.join(dst, vname), values, fmt="%.10e")
        shutil.copyfile(os.path.join(str_eq, dname), os.path.join(dst, dname))
        print(f"{kind}: {len(values)} values scaled by {args.eq_scale}")

    # Maxwell / creep files and both properties JSONs come from vhb_neq_creep
    for name in sorted(os.listdir(str_neq)):
        if name in EQ_FILES["I1"] + EQ_FILES["I2"]:
            continue
        shutil.copyfile(os.path.join(str_neq, name), os.path.join(dst, name))
        print("copied", name)
    print("folder:", dst)


if __name__ == "__main__":
    main()
