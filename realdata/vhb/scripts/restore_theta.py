"""Re-apply the accepted parameter vector of a finished fit to its interp files.

fit_vhb.py evaluates each trial by writing it into the interpolation folder; if
the final trial was rejected (or the process was interrupted) the files hold
trial values while theta.npy/final_metrics.json hold the accepted ones.  This
helper writes the accepted vector back.

Run: python -I restore_theta.py --mode eq --interp <interp dir> \
        --state results/fit_eq_spline
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

EQ_I1 = "interpolationValues_ISOCHORIC_SEF_I1.txt"
EQ_I2 = "interpolationValues_ISOCHORIC_SEF_I2_POLYCONVEX.txt"


def neq_file(branch, kind):
    if kind == "I1":
        return f"interpolationValues_ISOCHORIC_NONEQSEF_{branch}_I1.txt"
    if kind == "I2":
        return f"interpolationValues_ISOCHORIC_NONEQSEF_I2_{branch}_POLYCONVEX.txt"
    return f"interpolationValues_CREEP_POTENTIAL_I1tau2_MINUS_{branch}_3I2tau.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eq", "neq"], required=True)
    ap.add_argument("--interp", required=True)
    ap.add_argument("--state", required=True, help="fit output dir with theta.npy")
    args = ap.parse_args()

    theta = np.load(os.path.join(args.state, "theta.npy"))
    with open(os.path.join(args.state, "final_metrics.json")) as f:
        doc = json.load(f)
    stored = np.array(doc["theta"])
    if not np.allclose(theta, stored):
        raise SystemExit("theta.npy and final_metrics.json disagree")

    if args.mode == "eq":
        if theta.size != 17:
            raise SystemExit(f"eq theta has {theta.size} entries, expected 17")
        np.savetxt(os.path.join(args.interp, EQ_I1), theta[:12], fmt="%.10e")
        np.savetxt(os.path.join(args.interp, EQ_I2), theta[12:], fmt="%.10e")
    else:
        if theta.size % 18:
            raise SystemExit(f"neq theta has {theta.size} entries, expected a multiple of 18")
        nb = theta.size // 18
        for b in range(nb):
            a = 18 * b
            np.savetxt(os.path.join(args.interp, neq_file(b, "I1")),
                       theta[a:a + 8], fmt="%.10e")
            np.savetxt(os.path.join(args.interp, neq_file(b, "I2")),
                       theta[a + 8:a + 16], fmt="%.10e")
            np.savetxt(os.path.join(args.interp, neq_file(b, "creep")),
                       theta[a + 16:a + 18], fmt="%.10e")
    print(f"restored {theta.size} values into {args.interp}")


if __name__ == "__main__":
    main()
