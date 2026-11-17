"""Compare the raw force records with the processed release (guide 2.2/3.3).

The raw files are fixed-width: force [N], grip displacement [mm], time [s].
The processed files are tab separated with a -273 placeholder column:
force, grip displacement, placeholder, time.

Run:  python -I check_force_sources.py --raw <raw>/extract/force_displacement_curves \
                                      --processed <extract>/calibration/data/rate_dep/vhb \
                                      --out results
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

RATES = ["ldot0p015", "ldot0p15", "ldot1p5", "ldot3p75", "ldot7p5",
         "ldot15p0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for rate in RATES:
        v = float(rate[4:].replace("p", "."))
        widths = (7, 9, 9) if v == 0.015 else (7, 8, 8)
        for rep in (1, 2, 3):
            raw_path = os.path.join(
                args.raw, f"vhb_{rate}_plate_d5mm_rep{rep}.txt")
            proc_path = os.path.join(
                args.processed, rate, f"plate_d5mm_rep{rep}",
                "ForceDisplacement.txt")
            raw = np.genfromtxt(raw_path, delimiter=widths,
                                encoding="utf-8-sig")
            proc = np.genfromtxt(proc_path, delimiter="\t",
                                 encoding="utf-8-sig")
            n = min(len(raw), len(proc))
            d = np.abs(raw[:n, :3] - proc[:n, [0, 1, 3]])
            rows.append({
                "rate_mm_s": v, "replicate": rep,
                "n_raw": len(raw), "n_processed": len(proc),
                "max_abs_force_N": float(d[:, 0].max()),
                "max_abs_disp_mm": float(d[:, 1].max()),
                "max_abs_time_s": float(d[:, 2].max()),
                "n_nan_raw": int(np.isnan(raw).sum()),
                "n_nan_processed": int(np.isnan(proc[:, [0, 1, 3]]).sum()),
            })
            print(f"{rate} rep{rep}: rows {len(raw)}/{len(proc)}  "
                  f"dF {d[:, 0].max():.2e} N  ddisp {d[:, 1].max():.2e} mm  "
                  f"dt {d[:, 2].max():.2e} s")

    with open(os.path.join(args.out, "force_source_check.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(max(r["max_abs_force_N"], r["max_abs_disp_mm"],
                    r["max_abs_time_s"]) for r in rows)
    with open(os.path.join(args.out, "force_source_check.json"), "w") as f:
        json.dump({"rows": rows, "worst_absolute_difference": worst,
                   "all_equal_to_float64_rounding": worst < 1e-9}, f,
                  indent=1)
    print(f"worst difference {worst:.2e}")


if __name__ == "__main__":
    main()
