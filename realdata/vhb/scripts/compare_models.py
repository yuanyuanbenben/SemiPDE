"""Collect calibration metrics of several model families into one table.

Reads the ``final_metrics.json`` files written by ``fit_vhb.py`` (fits and
``--iters 0`` evaluations), joins them on the experiment key, and writes a CSV
plus a markdown table.  Metrics are the authors' weighted-residual quantities
computed on the unweighted predictions: force relative error (4*F_sim vs F_exp)
and hole-edge displacement error in mm.

Run:  python -I compare_models.py --model spline:file1,file2 \
        --model zener:file3,file4 --out results/model_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json

ORDER = ["eq", "ldot0p15", "ldot1p5", "ldot3p75", "ldot7p5", "ldot15p0"]
LABEL = {"eq": "0.015 (eq)", "ldot0p15": "0.15", "ldot1p5": "1.5",
         "ldot3p75": "3.75", "ldot7p5": "7.5", "ldot15p0": "15"}
TRAIN = {"ldot0p15", "ldot1p5", "ldot15p0"}
HOLDOUT = {"ldot3p75", "ldot7p5"}


def load(paths):
    out = {}
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        metrics = doc.get("metrics", doc)
        for key, m in metrics.items():
            if isinstance(m, dict) and "force_rel_mean" in m:
                out[key] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name:path[,path...]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = {}
    for spec in args.model:
        name, paths = spec.split(":", 1)
        models[name] = load(paths.split(","))

    rows = []
    for key in ORDER:
        for name, data in models.items():
            if key not in data:
                continue
            m = data[key]
            split = "train" if key in TRAIN else ("holdout" if key in HOLDOUT else "eq")
            rows.append({
                "rate_mm_s": LABEL[key], "split": split, "model": name,
                "force_rel_mean_pct": 100 * m["force_rel_mean"],
                "force_rel_absmean_pct": 100 * m["force_rel_absmean"],
                "force_rel_absmax_pct": 100 * m["force_rel_max"],
                "u_abs_mean_mm": m["u_abs_mean"],
                "u_abs_max_mm": m["u_abs_max"],
                "u_rel_mean_pct": 100 * m["u_rel_mean"],
            })

    fields = list(rows[0])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    hdr = "| rate (mm/s) | split | model | force mean | force absmax | u mean (mm) | u max (mm) | u rel |"
    print(hdr)
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['rate_mm_s']} | {r['split']} | {r['model']} | "
              f"{r['force_rel_mean_pct']:+.1f}% | {r['force_rel_absmax_pct']:.1f}% | "
              f"{r['u_abs_mean_mm']:.2f} | {r['u_abs_max_mm']:.2f} | "
              f"{r['u_rel_mean_pct']:.1f}% |")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
