"""Create a writable fit tree for calibrating the VHB spline model.

Copies the equilibrium and viscoelastic interpolation folders and builds one
private .prm per experiment (the runner mutates the .prm, so concurrent runs
must not share one), all pointing at the copied interpolation folders.

Run:  python -I setup_fit_tree.py --tree runs/author_tree --fit-dir \
        runs/author_tree/fe_solver/input/fit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil


def set_key(path, key, value):
    pat = re.compile(r"(?m)^([ \t]*set[ \t]+" + re.escape(key) + r"[ \t]*=[ \t]*).*$")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new, n = pat.subn(lambda m: m.group(1) + value, text)
    if n != 1:
        raise RuntimeError(f"{key}: {n} matches in {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)


def copy_prm_set(src_dir, dst_dir, interp_dir, experiment, end=None, dt=None,
                 strain_table=None):
    os.makedirs(dst_dir, exist_ok=True)
    prm = os.path.join(dst_dir, "parameters_plate_with_hole.prm")
    mat = os.path.join(dst_dir, "matmod_1.prm")
    shutil.copyfile(os.path.join(src_dir, "parameters_plate_with_hole.prm"), prm)
    shutil.copyfile(os.path.join(src_dir, "matmod_1.prm"), mat)
    set_key(prm, "Material file names", mat)
    set_key(prm, "Experiment name", experiment)
    set_key(mat, "Folder interpolation input", interp_dir)
    if strain_table is not None:
        set_key(prm, "Strain to image data table", strain_table)
    if end is not None:
        set_key(prm, "End time", str(end))
    if dt is not None:
        set_key(prm, "Initial time increment", str(dt))
    return prm


EXPERIMENTS = {
    "eq": {"rate": 0.015, "end": 5.3, "dt": 0.1,
           "experiment": "rate_dep/vhb/ldot0p015/plate_d5mm_rep1/"
                         "msh_6x7.5x0.25_768elm/extrap/"},
    "ldot0p15": {"rate": 0.15, "end": 530.0, "dt": 10.0,
                 "experiment": "rate_dep/vhb/ldot0p15/plate_d5mm_rep1/"
                               "msh_6x7.5x0.25_96elm/extrap/"},
    "ldot1p5": {"rate": 1.5, "end": 53.0, "dt": 1.0,
                "experiment": "rate_dep/vhb/ldot1p5/plate_d5mm_rep1/"
                              "msh_6x7.5x0.25_96elm/extrap/"},
    "ldot3p75": {"rate": 3.75, "end": 21.2, "dt": 0.4,
                 "experiment": "rate_dep/vhb/ldot3p75/plate_d5mm_rep1/"
                               "msh_6x7.5x0.25_96elm/extrap/"},
    "ldot7p5": {"rate": 7.5, "end": 10.6, "dt": 0.2,
                "experiment": "rate_dep/vhb/ldot7p5/plate_d5mm_rep1/"
                              "msh_6x7.5x0.25_96elm/extrap/"},
    "ldot15p0": {"rate": 15.0, "end": 5.3, "dt": 0.1,
                 "experiment": "rate_dep/vhb/ldot15p0/plate_d5mm_rep1/"
                               "msh_6x7.5x0.25_96elm/extrap/"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--fit-dir", required=True)
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    fe = os.path.join(tree, "fe_solver")
    inp = os.path.join(fe, "input")
    fit = os.path.abspath(args.fit_dir)
    os.makedirs(fit, exist_ok=True)

    interp_eq = os.path.join(fit, "interp_eq")
    interp_neq = os.path.join(fit, "interp_neq")
    for src, dst in [(os.path.join(inp, "interpolation", "vhb_eq"), interp_eq),
                     (os.path.join(inp, "interpolation", "vhb_neq_creep"), interp_neq)]:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    exp_table = {}
    eq = EXPERIMENTS["eq"]
    prm = copy_prm_set(os.path.join(inp, "prm", "vhb_eq"),
                       os.path.join(fit, "prm_eq"), interp_eq,
                       eq["experiment"], eq["end"], eq["dt"],
                       os.path.join(inp, "strain2Img", "vhb_eq", "images.dat"))
    exp_table["eq"] = {"name": eq["experiment"], "prm": prm,
                       "end": eq["end"], "dt": eq["dt"], "interp": interp_eq}

    for tag in ("ldot0p15", "ldot1p5", "ldot3p75", "ldot7p5", "ldot15p0"):
        info = EXPERIMENTS[tag]
        prm = copy_prm_set(os.path.join(inp, "prm", "vhb_neq_creep"),
                           os.path.join(fit, "prm_neq", tag), interp_neq,
                           info["experiment"], info["end"], info["dt"],
                           os.path.join(inp, "strain2Img", "vhb_neq_creep", "images.dat"))
        exp_table[tag] = {"name": info["experiment"], "prm": prm,
                          "end": info["end"], "dt": info["dt"],
                          "interp": interp_neq}

    with open(os.path.join(fit, "experiments.json"), "w") as f:
        json.dump(exp_table, f, indent=1)
    print("fit tree:", fit)
    print("interpolation folders:", interp_eq, interp_neq)
    print("experiments:", ", ".join(exp_table))


if __name__ == "__main__":
    main()
