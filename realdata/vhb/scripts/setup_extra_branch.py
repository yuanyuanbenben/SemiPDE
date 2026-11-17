"""Add extra Maxwell branches to a fit tree built by setup_fit_tree.py.

The author solver reads the branch count from ``Number Maxwell elements`` in
``matmod_1.prm``, one ``"Maxwell element <b>"`` block per branch in
``propsGeneralizedMaxwell.json``, and one interpolation domain/value pair per
branch index (``..._<b>_...``).  This script appends the missing branches with

  * domains copied from one existing branch (default branch 0),
  * a neo-Hookean non-equilibrium energy  psi = 0.5 * mu * (I1 - 3),
  * a linear creep potential with slope c (creep value at the domain end),

so the resulting model is a generalised Zener chain with ``nbranches`` Maxwell
elements.  ``fit_vhb.py --model zener --nbranches <n>`` can then refit it.

For a linear creep potential the source comment in
``CreepPotentialI1tau2_MINUS_3I2tau.cc`` gives slope = 1/(6 eta), so a branch
with modulus mu and relaxation time tau needs c = 1/(6 mu tau).

Run:
  python -I setup_extra_branch.py --fit-dir <fit-dir> --nbranches 3 \
      --mu 0.03 --c 0.292
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np

DOMAIN_FILES = {
    "I1": "interpolationDomain_ISOCHORIC_NONEQSEF_{b}_I1.txt",
    "I2": "interpolationDomain_ISOCHORIC_NONEQSEF_I2_{b}_POLYCONVEX.txt",
    "creep": "interpolationDomain_CREEP_POTENTIAL_I1tau2_MINUS_{b}_3I2tau.txt",
}
VALUE_FILES = {
    "I1": "interpolationValues_ISOCHORIC_NONEQSEF_{b}_I1.txt",
    "I2": "interpolationValues_ISOCHORIC_NONEQSEF_I2_{b}_POLYCONVEX.txt",
    "creep": "interpolationValues_CREEP_POTENTIAL_I1tau2_MINUS_{b}_3I2tau.txt",
}

PROPS = "propsGeneralizedMaxwell.json"


def read_domain(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline().split(",")[0])
        return np.array([float(f.readline()) for _ in range(n)])


def props_block(order_creep=2, order_energy=4):
    """One Maxwell-element block identical to the released identify flags."""
    return {
        "Non Eq strain energy function": {
            "I1b": {
                "Interpolation type": "BSPLINE",
                "BSpline order": order_energy,
                "Allow extrapolation": True,
                "Identify parameters": True,
                "Identify last point": True,
            },
            "I2bPOLYCONVEX": {
                "Interpolation type": "BSPLINE",
                "BSpline order": order_energy,
                "Allow extrapolation": True,
                "Identify parameters": True,
                "Identify last point": True,
            },
        },
        "Non Eq viscosity function": {
            "I1tau2MINUS3I2tau": {
                "Interpolation type": "BSPLINE",
                "BSpline order": order_creep,
                "Allow extrapolation": True,
                "Identify parameters": True,
                "Identify last point": True,
            }
        },
    }


def set_key(path, key, value):
    pat = re.compile(r"(?m)^([ \t]*set[ \t]+" + re.escape(key) + r"[ \t]*=[ \t]*).*$")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new, n = pat.subn(lambda m: m.group(1) + value, text)
    if n != 1:
        raise RuntimeError(f"{key}: {n} matches in {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--nbranches", type=int, default=3)
    ap.add_argument("--source-branch", type=int, default=0,
                    help="existing branch whose domains are copied to new branches")
    ap.add_argument("--mu", type=float, default=0.03,
                    help="initial neo-Hookean modulus of every new branch")
    ap.add_argument("--c", type=float, default=0.292,
                    help="initial linear creep slope c = 1/(6 mu tau)")
    args = ap.parse_args()

    fit = os.path.abspath(args.fit_dir)
    interp = os.path.join(fit, "interp_neq")
    # count only branch-indexed files: the equilibrium file has no index
    existing = sum(1 for p in glob.glob(os.path.join(
        interp, "interpolationValues_ISOCHORIC_NONEQSEF_*_I1.txt"))
        if re.search(r"_NONEQSEF_\d+_I1\.txt$", os.path.basename(p)))
    if existing < args.nbranches:
        print(f"interp_neq has {existing} branches; adding "
              f"{args.nbranches - existing}")

    for b in range(existing, args.nbranches):
        for kind, name in DOMAIN_FILES.items():
            src = os.path.join(interp, name.format(b=args.source_branch))
            dst = os.path.join(interp, name.format(b=b))
            shutil.copyfile(src, dst)
            print("copied domain", os.path.basename(dst))
        x = read_domain(os.path.join(interp, DOMAIN_FILES["I1"].format(b=b)))
        np.savetxt(os.path.join(interp, VALUE_FILES["I1"].format(b=b)),
                   0.5 * args.mu * (x - 3.0), fmt="%.10e")
        i2 = read_domain(os.path.join(interp, DOMAIN_FILES["I2"].format(b=b)))
        np.savetxt(os.path.join(interp, VALUE_FILES["I2"].format(b=b)),
                   np.zeros_like(i2), fmt="%.10e")
        np.savetxt(os.path.join(interp, VALUE_FILES["creep"].format(b=b)),
                   np.array([0.0, args.c]), fmt="%.10e")
        print(f"branch {b}: mu {args.mu}, c {args.c}")

    props_path = os.path.join(interp, PROPS)
    with open(props_path, encoding="utf-8") as f:
        props = json.load(f)
    for b in range(args.nbranches):
        key = f"Maxwell element {b}"
        if key not in props:
            props[key] = props_block()
            print("props:", key, "added")
    with open(props_path, "w", encoding="utf-8") as f:
        json.dump(props, f, indent=2)

    for prm in sorted(glob.glob(os.path.join(fit, "prm_neq", "*", "matmod_1.prm"))):
        set_key(prm, "Number Maxwell elements", str(args.nbranches))
        set_key(prm, "Non-equilibrium energy invariants",
                "|".join(["I1b,I2b_POLYCONVEX"] * args.nbranches))
        set_key(prm, "Creep potential invariants",
                "|".join(["I1tau2_MINUS_3I2tau"] * args.nbranches))
        print("updated", prm)


if __name__ == "__main__":
    main()
