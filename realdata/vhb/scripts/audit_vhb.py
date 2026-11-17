"""Stage A data audit for the VHB 4905 multi-rate viscoelasticity data.

Reads the processed release that ships with the author code
(``calibration/data/rate_dep/vhb/``), verifies the column order of the force
files, the image-token mapping, the DIC masks, the eight common hole-edge
observation points and the descriptive force statistics.

Only numbers are read from the downloaded tree; no code from it is executed.
Every path is passed as an argument.

Run:  python -I audit_vhb.py --root <extract>/davis-viscoelasticity-1.0.0 \
                            --out results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np

RATES = ["ldot0p015", "ldot0p15", "ldot1p5", "ldot3p75", "ldot7p5",
         "ldot15p0"]
# guide 3.1: three repetitions per rate; the lowest rate has force-only rep4/5
REPS = [1, 2, 3]
EXCLUDED_REPS = {"ldot0p015": [4, 5]}
MESH = "msh_6x7.5x0.25_96elm"
MESH_EQ = "msh_6x7.5x0.25_768elm"
# the slowest rate is calibrated in the equilibrium stage: 768-element mesh
# and the vhb_eq image map (guide 3.6)
SLOW_RATE = 0.015

# guide 3.6: X_l = (-2.5 cos(l pi/16), 2.5 sin(l pi/16), 0), l = 1..8
N_HOLE = 8
R_HOLE = 2.5
ANCHOR = np.array([-R_HOLE, 0.0, 0.0])


def hole_points(n=N_HOLE, r=R_HOLE):
    el = np.arange(1, n + 1)
    return np.stack([-r * np.cos(el * np.pi / 16),
                     r * np.sin(el * np.pi / 16),
                     np.zeros_like(el)], axis=1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rate_from_dir(name):
    token = name[len("ldot"):].replace("p", ".")
    return float(token)


def read_images(root, stage="vhb_neq_creep"):
    """Return {relative folder: (strains, [image tokens])}."""
    path = os.path.join(root, "fe_solver", "input", "strain2Img",
                        stage, "images.dat")
    rows = {}
    strains = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = [c.strip() for c in line.split("\t") if c.strip() != ""]
            if cells[0] == "Experiment":
                strains = [float(c) for c in cells[1:]]
                continue
            rows[cells[0].rstrip("/")] = cells[1:]
    return strains, rows


def read_force(path):
    a = np.genfromtxt(path, delimiter="\t", encoding="utf-8-sig")
    return {"force_N": a[:, 0], "grip_mm": a[:, 1],
            "placeholder": a[:, 2], "time_s": a[:, 3]}


def read_dic(path):
    """Return xyz (N,3), u (N,), valid mask (N,)."""
    a = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    xyz = np.stack([a["Points0"], a["Points1"], a["Points2"]], axis=1)
    return xyz, a["u"], a["vtkValidPointMask"]


def match_points(xyz, pts, tol=1e-6):
    """Indices of ``pts`` in ``xyz`` (exact reference matching, guide 3.6)."""
    k = np.zeros(len(pts), dtype=int)
    for i, p in enumerate(pts):
        d = np.linalg.norm(xyz - p[None, :], axis=1)
        j = int(np.argmin(d))
        if d[j] > tol:
            return None, float(d[j])
        k[i] = j
    return k, float(np.max(np.linalg.norm(xyz[k] - pts, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_root = os.path.join(args.root, "calibration", "data", "rate_dep",
                             "vhb")
    strains, images_neq = read_images(args.root, "vhb_neq_creep")
    strains_eq, images_eq = read_images(args.root, "vhb_eq")
    pts = hole_points()
    anchor = ANCHOR[None, :]

    rows = []
    problems = []
    mesh_compare = []
    for rate_dir in RATES:
        v = rate_from_dir(rate_dir)
        for rep in REPS:
            run = f"plate_d5mm_rep{rep}"
            base = os.path.join(data_root, rate_dir, run)
            slow = abs(v - SLOW_RATE) < 1e-12
            mesh = MESH_EQ if slow else MESH
            images = images_eq if slow else images_neq
            rel = f"rate_dep/vhb/{rate_dir}/{run}/{mesh}/extrap"
            tokens = images.get(rel)
            if tokens is None:
                problems.append(f"no image mapping for {rel}")
                continue
            # tokens are image numbers, not strains; only their count is
            # constrained by the mapping table
            if len(tokens) != len(strains):
                problems.append(f"{rel}: {len(tokens)} tokens, "
                                f"{len(strains)} strains")

            ffd = read_force(os.path.join(base, "ForceDisplacement.txt"))
            dt = float(np.median(np.diff(ffd["time_s"])))
            rec = {
                "experiment_id": f"vhb_{rate_dir}_{run}",
                "material": "VHB_4905",
                "rate_mm_s": v,
                "nominal_strain_rate_s_inv": v / 15.0,
                "replicate": rep,
                "specimen_id_assumed": f"{rate_dir}_{run}",
                "force_source": os.path.relpath(
                    os.path.join(base, "ForceDisplacement.txt"), args.root),
                "dic_source": os.path.relpath(
                    os.path.join(base, mesh, "extrap"), args.root),
                "image_map_source": "fe_solver/input/strain2Img/"
                                    "vhb_neq_creep/images.dat",
                "units": "N_mm_s",
                "coordinate_transform_id": "author_reference_surface",
                "time_origin_rule": "author_image_mapping",
                "n_force_points": len(ffd["time_s"]),
                "force_median_dt_s": dt,
                "time_monotone": bool(np.all(np.diff(ffd["time_s"]) > 0)),
                "placeholder_all_minus273": bool(
                    np.all(ffd["placeholder"] == -273)),
                "n_stages": len(tokens),
            }

            uy = np.full((len(tokens), N_HOLE), np.nan)
            valid = np.zeros((len(tokens), N_HOLE), dtype=bool)
            uy_anchor = np.full(len(tokens), np.nan)
            anchor_valid = np.zeros(len(tokens), dtype=bool)
            coord_err = 0.0
            file_missing = 0
            for j, tok in enumerate(tokens):
                path = os.path.join(base, mesh, "extrap", f"u2_{tok}.txt")
                if not os.path.exists(path):
                    file_missing += 1
                    continue
                xyz, u, mask = read_dic(path)
                k, err = match_points(xyz, pts)
                if k is None:
                    problems.append(f"{rel} stage {j}: point match {err:.2e}")
                    continue
                coord_err = max(coord_err, err)
                uy[j] = u[k]
                valid[j] = mask[k] == 1
                ka, erra = match_points(xyz, anchor)
                if ka is not None:
                    uy_anchor[j] = u[ka[0]]
                    anchor_valid[j] = mask[ka[0]] == 1

            rec["mesh"] = mesh
            rec["n_valid_displacement_values"] = int(valid.sum())
            rec["valid_fraction"] = float(valid.mean())
            rec["anchor_valid_fraction"] = float(anchor_valid.mean())
            rec["stage_strain_min"] = float(min(strains))
            rec["stage_strain_max"] = float(max(strains))
            rec["coord_match_max_err_mm"] = coord_err
            rec["missing_dic_files"] = file_missing

            # descriptive force at grip = 10 mm (guide 3.7)
            i10 = np.interp(10.0, ffd["grip_mm"], ffd["force_N"])
            rec["force_at_10mm_N"] = float(i10)

            # the DIC release is resampled on both finite element surfaces;
            # compare the eight common points for every run and stage.  Both
            # meshes carry the same image tokens on disk even though the
            # author image map only drives one of them per stage.
            both_ok = all(
                os.path.exists(os.path.join(base, mesh, "extrap",
                                            f"u2_{tok}.txt"))
                for mesh in (MESH, MESH_EQ) for tok in tokens)
            if both_ok:
                diffs = []
                mask_bad = 0
                n96 = n768 = 0
                for j, tok in enumerate(tokens):
                    x96, u96, m96 = read_dic(os.path.join(
                        base, MESH, "extrap", f"u2_{tok}.txt"))
                    x768, u768, m768 = read_dic(os.path.join(
                        base, MESH_EQ, "extrap", f"u2_{tok}.txt"))
                    n96, n768 = len(u96), len(u768)
                    k96, _ = match_points(x96, pts)
                    k768, _ = match_points(x768, pts)
                    if k96 is None or k768 is None:
                        continue
                    diffs.append(float(np.max(np.abs(u96[k96] - u768[k768]))))
                    mask_bad += int(np.any((m96[k96] == 1) !=
                                           (m768[k768] == 1)))
                rec["surface_points_96elm"] = n96
                rec["surface_points_768elm"] = n768
                mesh_compare.append({
                    "run": rec["experiment_id"],
                    "stages_compared": len(diffs),
                    "max_abs_uy_diff_mm": float(max(diffs)),
                    "mask_mismatch_stages": mask_bad})
            else:
                problems.append(f"{rec['experiment_id']}: incomplete "
                                f"96/768 DIC pair")

            rows.append(rec)

    # manifest
    os.makedirs(args.out, exist_ok=True)
    fields = ["experiment_id", "material", "rate_mm_s",
              "nominal_strain_rate_s_inv", "replicate", "split",
              "specimen_id_assumed", "force_source", "dic_source",
              "image_map_source", "units", "coordinate_transform_id",
              "time_origin_rule", "n_stages", "n_valid_displacement_values",
              "valid_fraction", "anchor_valid_fraction", "stage_strain_min",
              "stage_strain_max", "n_force_points", "force_median_dt_s",
              "time_monotone", "placeholder_all_minus273",
              "coord_match_max_err_mm", "missing_dic_files",
              "force_at_10mm_N", "exclusion_reason"]
    train_rates = {0.015, 0.15, 1.5, 15.0}
    val_rates = {3.75}
    test_rates = {7.5}
    for r in rows:
        r["split"] = ("train" if r["rate_mm_s"] in train_rates else
                      "validation" if r["rate_mm_s"] in val_rates else
                      "test")
        r["exclusion_reason"] = ""
    with open(os.path.join(args.out, "data_manifest.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # descriptives per rate
    stats = {}
    for v in sorted({r["rate_mm_s"] for r in rows}):
        vals = [r["force_at_10mm_N"] for r in rows if r["rate_mm_s"] == v]
        stats[str(v)] = {"mean_N": float(np.mean(vals)),
                         "std_N": float(np.std(vals, ddof=1)),
                         "n": len(vals)}
    out = {"strains": strains, "n_stages": len(strains),
           "runs": len(rows), "rates": sorted({r["rate_mm_s"] for r in rows}),
           "force_at_10mm": stats,
           "mesh_96_vs_768_max_abs_u_mm": mesh_compare,
           "problems": problems,
           "excluded": EXCLUDED_REPS}
    with open(os.path.join(args.out, "data_audit.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"runs {len(rows)}  stages {len(strains)}  problems {len(problems)}")
    for r in rows:
        print(f"  {r['experiment_id']:28s} nF {r['n_force_points']:5d} "
              f"dt {r['force_median_dt_s']:.3f} valid {r['valid_fraction']:.2f} "
              f"F10 {r['force_at_10mm_N']:.4f} split {r['split']}")
    for v, s in stats.items():
        print(f"  rate {v:>7s}: F(10mm) {s['mean_N']:.6f} +/- {s['std_N']:.6f} "
              f"(n={s['n']})")
    if mesh_compare:
        print("  96 vs 768 element u_y: "
              f"{max(m['max_abs_uy_diff_mm'] for m in mesh_compare):.2e} mm "
              f"max diff, {sum(m['mask_mismatch_stages'] for m in mesh_compare)} "
              f"mask mismatching stages over {sum(m['stages_compared'] for m in mesh_compare)}")
    for p in problems[:10]:
        print("  PROBLEM:", p)


if __name__ == "__main__":
    main()
