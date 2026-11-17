"""Write the non-equilibrium branch values of a fit tree from a source tree.

Used to build an alternative initial point for a multi-branch fit: the
branches present in the source (``--from-dir``) are copied verbatim, and any
additional branch is filled with a neo-Hookean energy

    psi = 0.5 * mu * (I1 - 3)

and a linear creep potential with slope c.  Domains are always taken from the
target tree, so the two trees must already have the same branch count.

Run:
  python -I set_branch_start.py --fit-dir <target-fit> \
      --from-dir <source-fit>/interp_neq --nbranches 3 --mu 1e-4 --c 1.667
"""

from __future__ import annotations

import argparse
import os

import numpy as np

VALUE_FILES = {
    "I1": "interpolationValues_ISOCHORIC_NONEQSEF_{b}_I1.txt",
    "I2": "interpolationValues_ISOCHORIC_NONEQSEF_I2_{b}_POLYCONVEX.txt",
    "creep": "interpolationValues_CREEP_POTENTIAL_I1tau2_MINUS_{b}_3I2tau.txt",
}
DOMAIN_FILES = {
    "I1": "interpolationDomain_ISOCHORIC_NONEQSEF_{b}_I1.txt",
    "I2": "interpolationDomain_ISOCHORIC_NONEQSEF_I2_{b}_POLYCONVEX.txt",
}


def read_domain(path):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline().split(",")[0])
        return np.array([float(f.readline()) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--from-dir", required=True,
                    help="interp_neq directory of the source fit tree")
    ap.add_argument("--nbranches", type=int, default=3)
    ap.add_argument("--mu", type=float, default=1e-4,
                    help="modulus of every branch not present in the source")
    ap.add_argument("--c", type=float, default=1.667,
                    help="creep slope of every branch not present in the source")
    args = ap.parse_args()

    interp = os.path.join(os.path.abspath(args.fit_dir), "interp_neq")
    source = os.path.abspath(args.from_dir)
    n_src = sum(1 for b in range(args.nbranches)
                if os.path.exists(os.path.join(source, VALUE_FILES["I1"].format(b=b))))
    if n_src >= args.nbranches:
        raise SystemExit(f"source already has {n_src} >= {args.nbranches} branches")
    print(f"source has {n_src} branch(es); filling {args.nbranches - n_src}")

    for b in range(n_src):
        for name in VALUE_FILES.values():
            src = os.path.join(source, name.format(b=b))
            dst = os.path.join(interp, name.format(b=b))
            with open(src, encoding="utf-8") as f:
                text = f.read()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(text)
        print(f"branch {b}: copied from source")

    for b in range(n_src, args.nbranches):
        x = read_domain(os.path.join(interp, DOMAIN_FILES["I1"].format(b=b)))
        np.savetxt(os.path.join(interp, VALUE_FILES["I1"].format(b=b)),
                   0.5 * args.mu * (x - 3.0), fmt="%.10e")
        i2 = read_domain(os.path.join(interp, DOMAIN_FILES["I2"].format(b=b)))
        np.savetxt(os.path.join(interp, VALUE_FILES["I2"].format(b=b)),
                   np.zeros_like(i2), fmt="%.10e")
        np.savetxt(os.path.join(interp, VALUE_FILES["creep"].format(b=b)),
                   np.array([0.0, args.c]), fmt="%.10e")
        print(f"branch {b}: mu {args.mu}, c {args.c}")


if __name__ == "__main__":
    main()
