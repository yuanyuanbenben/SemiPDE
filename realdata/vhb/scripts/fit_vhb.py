"""Levenberg-Marquardt calibration of the author's spline viscoelastic model.

The residual vector and the Jacobian follow the authors' MATLAB objective
(``source/core/Objective_func.m``): the summed nodal reaction is multiplied by
the symmetry factor 4, forces and displacements are scaled per experiment
(``Compute_scaleVector.m``), and the solver's analytic force/displacement
sensitivities are used as the Jacobian.

Three model families are supported:

* ``spline`` -- the free interpolation values of the author's model
  (17 equilibrium values or 42 Maxwell values);
* ``zener``  -- a neo-Hookean equilibrium plus two neo-Hookean Maxwell
  branches with linear (Newtonian) creep potentials, mapped onto the same
  interpolation values;
* ``nn``     -- the nonparametric part is a neural network: one one-hidden-layer
  tanh MLP per (branch, function) maps the invariant to the interpolation
  values at the solver nodes.  The solver, its analytic sensitivities, the
  residual and the protocol are unchanged; the Jacobian is
  ``J_nodes @ d(nodes)/d(weights)``.

Run (equilibrium):
  python -I fit_vhb.py --mode eq --model spline --tree runs/author_tree \
    --fit-dir runs/author_tree/fe_solver/input/fit --iters 30 \
    --init-scale 0.25 --out results/fit_eq_spline

Run (Maxwell, after the equilibrium values were written into interp_neq):
  python -I fit_vhb.py --mode neq --model spline --rates ldot0p15,ldot1p5,ldot15p0 \
    ... --iters 60 --out results/fit_neq_spline
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# --------------------------------------------------------------------------
# interpolation files
# --------------------------------------------------------------------------

def neq_file(branch, kind):
    if kind == "I1":
        return f"interpolationValues_ISOCHORIC_NONEQSEF_{branch}_I1.txt"
    if kind == "I2":
        return f"interpolationValues_ISOCHORIC_NONEQSEF_I2_{branch}_POLYCONVEX.txt"
    if kind == "creep":
        return f"interpolationValues_CREEP_POTENTIAL_I1tau2_MINUS_{branch}_3I2tau.txt"
    raise KeyError(kind)


EQ_I1 = "interpolationValues_ISOCHORIC_SEF_I1.txt"
EQ_I2 = "interpolationValues_ISOCHORIC_SEF_I2_POLYCONVEX.txt"


def neq_columns(nb=2):
    cols = []
    for b in range(nb):
        o = 21 * b
        cols += list(range(o, o + 8))       # I1 values
        cols += list(range(o + 9, o + 17))   # I2 values
        cols += [o + 18, o + 19]             # creep values
    return cols


def eq_columns():
    # 19 columns: I1 values, I1 last point, I2 values, I2 last point
    return list(range(12)) + list(range(13, 18))


def write_eq_values(interp_dir, i1, i2):
    np.savetxt(os.path.join(interp_dir, EQ_I1), i1, fmt="%.10e")
    np.savetxt(os.path.join(interp_dir, EQ_I2), i2, fmt="%.10e")


def read_eq_values(interp_dir):
    return (np.loadtxt(os.path.join(interp_dir, EQ_I1)),
            np.loadtxt(os.path.join(interp_dir, EQ_I2)))


def write_neq_values(interp_dir, branches):
    """branches: dict b -> {"I1": array8, "I2": array8, "creep": array2}."""
    for b, parts in branches.items():
        for kind, arr in parts.items():
            np.savetxt(os.path.join(interp_dir, neq_file(b, kind)), arr,
                       fmt="%.10e")


def read_neq_values(interp_dir, nb=2):
    out = {}
    for b in range(nb):
        out[b] = {k: np.loadtxt(os.path.join(interp_dir, neq_file(b, k)))
                  for k in ("I1", "I2", "creep")}
    return out


def neq_vector(branches, nb=2):
    v = []
    for b in range(nb):
        v += list(branches[b]["I1"]) + list(branches[b]["I2"]) + list(branches[b]["creep"])
    return np.array(v)


def neq_branches_from_vector(y, nb=2):
    out = {}
    for b in range(nb):
        a = 18 * b
        out[b] = {"I1": np.array(y[a:a + 8]),
                  "I2": np.array(y[a + 8:a + 16]),
                  "creep": np.array(y[a + 16:a + 18])}
    return out


# --------------------------------------------------------------------------
# reading solver output
# --------------------------------------------------------------------------

def sanitized(name):
    return name.rstrip("/").replace("/", "_")


def read_vec(path):
    with open(path, encoding="utf-8") as f:
        return np.array([float(x) for x in f.read().split()])


def read_matrix(path):
    """Binary FullMatrix written by Write_fullMatrix (row-major float64)."""
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(), dtype="<f8", count=rows * cols)
    return data.reshape(rows, cols)


def stage_files(dirpath, axis=False):
    out = {}
    for path in glob.glob(os.path.join(dirpath, "time_*_rank_0*")):
        name = os.path.basename(path)
        m = re.match(r"time_([0-9.]+)_rank_0" + ("_axis_0" if axis else "") + r"\.txt$", name)
        if m:
            out[round(float(m.group(1)), 6)] = path
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# experiment evaluation
# --------------------------------------------------------------------------

def set_key(path, key, value):
    pat = re.compile(r"(?m)^([ \t]*set[ \t]+" + re.escape(key) + r"[ \t]*=[ \t]*).*$")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new, n = pat.subn(lambda m: m.group(1) + value, text)
    if n != 1:
        raise RuntimeError(f"{key}: {n} matches in {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)


def ensure_prepared(info, solver, calib_input, log_dir):
    sub = sanitized(info["name"])
    fdir = os.path.join(calib_input, "forces_experiment", sub)
    if glob.glob(os.path.join(fdir, "time_*_rank_0_axis_0.txt")):
        return
    set_key(info["prm"], "Identification stage type", "PREPARE")
    with open(os.path.join(log_dir, f"prepare_{sub}.log"), "w") as f:
        proc = subprocess.run([solver, info["prm"]], stdout=f,
                              stderr=subprocess.STDOUT)
    set_key(info["prm"], "Identification stage type", "EXECUTE")
    if proc.returncode != 0:
        raise RuntimeError(f"PREPARE failed for {info['name']}")


class Experiment:
    def __init__(self, key, info, solver, calib_input):
        self.key = key
        self.name = info["name"]
        self.prm = info["prm"]
        self.interp = info["interp"]
        self.end = info["end"]
        self.dt = info["dt"]
        self.solver = solver
        self.base = calib_input
        self.sub = sanitized(self.name)
        self.f_exp = stage_files(os.path.join(self.base, "forces_experiment", self.sub), axis=True)
        self.u_exp = stage_files(os.path.join(self.base, "displacement_experiment", self.sub))
        if not self.f_exp or not self.u_exp:
            raise RuntimeError(f"experiment data missing for {self.name}; run PREPARE first")

    def run(self, log_dir):
        t0 = time.time()
        log = os.path.join(log_dir, f"run_{self.key}.log")
        with open(log, "w") as f:
            proc = subprocess.run([self.solver, self.prm], stdout=f,
                                  stderr=subprocess.STDOUT)
        return proc.returncode, time.time() - t0

    def results(self, n_params, columns):
        """Weighted residual and Jacobian for this experiment."""
        f_sim = stage_files(os.path.join(self.base, "forces_simulation", self.sub), axis=True)
        u_sim = stage_files(os.path.join(self.base, "displacement_simulation", self.sub))
        jf = stage_files(os.path.join(self.base, "sensitivity_forces", self.sub), axis=True)
        ju = stage_files(os.path.join(self.base, "sensitivity_displacements", self.sub))
        stages = [s for s in self.f_exp if s in f_sim and s in jf]
        stages_u = [s for s in self.u_exp if s in u_sim and s in ju]
        if len(stages) != len(self.f_exp):
            raise RuntimeError(f"{self.key}: {len(stages)}/{len(self.f_exp)} force stages")
        if len(stages_u) != len(self.u_exp):
            raise RuntimeError(f"{self.key}: {len(stages_u)}/{len(self.u_exp)} disp stages")

        fe = np.array([read_vec(self.f_exp[s]).sum() for s in stages])
        fs = np.array([read_vec(f_sim[s]).sum() for s in stages])
        wf = 10.0 / np.max(np.abs(fe))
        r = list(wf * (4.0 * fs - fe))
        J = []
        for s in stages:
            j = read_matrix(jf[s]).sum(axis=0)[columns]  # d(sum F)/d param
            J.append(wf * 4.0 * j)

        ue = np.concatenate([read_vec(self.u_exp[s]) for s in stages_u])
        us = np.concatenate([read_vec(u_sim[s]) for s in stages_u])
        wu = 1.0 / np.max(np.abs(ue))
        r.extend(list(wu * (us - ue)))
        for s in stages_u:
            j = read_matrix(ju[s])[:, columns]
            J.append(wu * j)

        r = np.array(r)
        J = np.vstack(J)
        return r, J, self.metrics(fe, fs, ue, us)

    def metrics(self, fe, fs, ue, us):
        rel = (4.0 * fs - fe) / fe
        du = np.abs(us - ue)
        return {
            "force_rel_mean": float(rel.mean()),
            "force_rel_absmean": float(np.abs(rel).mean()),
            "force_rel_max": float(np.abs(rel).max()),
            "u_abs_mean": float(du.mean()),
            "u_abs_max": float(du.max()),
            "u_rel_mean": float((du / np.maximum(np.abs(ue), 1e-12)).mean()),
        }


def evaluate(experiments, log_dir, n_params, columns, jobs):
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        codes = list(pool.map(lambda e: e.run(log_dir), experiments))
    failed = [(e.key, c) for e, (c, _) in zip(experiments, codes) if c != 0]
    if failed:
        return None, None, None, failed
    rs, Js, ms = [], [], {}
    for e in experiments:
        r, J, m = e.results(n_params, columns)
        rs.append(r)
        Js.append(J)
        ms[e.key] = m
    return np.concatenate(rs), np.vstack(Js), ms, []


# --------------------------------------------------------------------------
# parameter mapping
# --------------------------------------------------------------------------

def i1_nodes(interp_dir, kind="eq", branch=0):
    if kind == "eq":
        path = os.path.join(interp_dir, "interpolationDomain_ISOCHORIC_SEF_I1.txt")
    else:
        path = os.path.join(
            interp_dir, f"interpolationDomain_ISOCHORIC_NONEQSEF_{branch}_I1.txt")
    with open(path, encoding="utf-8") as f:
        n = int(f.readline().split(",")[0])
        return np.array([float(f.readline()) for _ in range(n)])


def write_zener(interp_dir, theta, nb=2):
    """theta = (mu1..munb, c1..cnb); equilibrium files are left untouched.

    Every branch is written on its own I1 domain: the solver reads
    interpolationDomain_ISOCHORIC_NONEQSEF_<b>_I1.txt per branch, and the
    branch domains differ (branch 0 ends at 7.24, branch 1 at 25.95), so using
    branch 0's nodes for every branch would rescale the physical modulus by the
    ratio of the domain spans.
    """
    mu = [theta[b] for b in range(nb)]
    c = [theta[nb + b] for b in range(nb)]
    branches = {}
    for b in range(nb):
        x = i1_nodes(interp_dir, kind="neq", branch=b)
        branches[b] = {"I1": 0.5 * mu[b] * (x - 3.0),
                       "I2": np.zeros(8),
                       "creep": np.array([0.0, c[b]])}
    write_neq_values(interp_dir, branches)


def zener_chain(interp_dir, nb=2):
    """(21*nb, 2*nb) matrix dy/dtheta over the identified columns."""
    n_values = 18 * nb
    M = np.zeros((n_values, 2 * nb))
    for b in range(nb):
        a = 18 * b
        x = i1_nodes(interp_dir, kind="neq", branch=b) - 3.0
        M[a:a + 8, b] = 0.5 * x                  # d I1 values / d mu_b
        M[a + 16 + 1, nb + b] = 1.0              # d creep[1] / d c_b
    return M


# --------------------------------------------------------------------------
# neural-network parameterisation of the nonparametric part
# --------------------------------------------------------------------------

NN_KINDS = ("I1", "I2", "creep")


def neq_domain(interp_dir, kind, branch):
    if kind == "I1":
        return i1_nodes(interp_dir, kind="neq", branch=branch)
    if kind == "I2":
        path = os.path.join(
            interp_dir,
            f"interpolationDomain_ISOCHORIC_NONEQSEF_I2_{branch}_POLYCONVEX.txt")
    elif kind == "creep":
        path = os.path.join(
            interp_dir,
            f"interpolationDomain_CREEP_POTENTIAL_I1tau2_MINUS_{branch}_3I2tau.txt")
    else:
        raise KeyError(kind)
    with open(path, encoding="utf-8") as f:
        n = int(f.readline().split(",")[0])
        return np.array([float(f.readline()) for _ in range(n)])


class NeuralConstitutive:
    """One tanh MLP per (branch, function), evaluated at the solver nodes.

    The network predicts a *correction* to a base function::

        node(x) = base(x) + scale * (softplus(mlp(x)) - log 2)

    so that a zero last layer reproduces ``base`` exactly.  Starting from a
    converged parametric solution therefore gives an initial evaluation that
    is bit-identical to it (this solver rejects interpolations that are only
    a fraction of a percent away from a working one).  The flat parameter
    vector holds, for every network, ``W1 (w)``, ``b1 (w)``, ``W2 (w)``,
    ``b2 (1)``; the MLP input is the invariant normalised to [-1, 1] over its
    interpolation domain.  ``values`` returns the 18*nb identified
    interpolation values, matching ``neq_vector``.
    """

    def __init__(self, torch, interp_dir, nb, width=8, seed=0, scales=None,
                 base=None):
        self.torch = torch
        self.width = width
        self.nb = nb
        # softplus(0); subtracting this same tensor value makes a zero last
        # layer contribute exactly no correction
        self.log2 = torch.nn.functional.softplus(
            torch.zeros(1, dtype=torch.float64))[0]
        torch.manual_seed(seed)
        current = read_neq_values(interp_dir, nb)
        self.specs = []
        params = []
        offset = 0
        node_off = 0
        for b in range(nb):
            for kind in NN_KINDS:
                x = neq_domain(interp_dir, kind, b)
                lo, hi = float(np.min(x)), float(np.max(x))
                span = hi - lo if hi > lo else 1.0
                xn = 2.0 * (x - lo) / span - 1.0
                idx = len(self.specs)
                v = np.asarray(current[b][kind], dtype=np.float64)
                if scales is not None:
                    scale = float(scales[idx])
                else:
                    scale = float(np.max(np.abs(v))) if v.size else 1.0
                    if scale < 1e-8:
                        scale = 1.0
                if base is not None:
                    base_spec = np.asarray(
                        base[node_off:node_off + len(x)], dtype=np.float64)
                    if base_spec.size != len(x):
                        raise ValueError(
                            f"nn_config base has {len(base)} values, need "
                            f"{node_off + len(x)}")
                else:
                    base_spec = v
                node_off += len(x)
                self.specs.append({
                    "branch": b, "kind": kind,
                    "xn": torch.tensor(xn, dtype=torch.float64),
                    "base": torch.tensor(base_spec, dtype=torch.float64),
                    "offset": offset, "scale": scale,
                })
                W1 = torch.randn(width, dtype=torch.float64) / np.sqrt(width)
                b1 = 0.1 * torch.randn(width, dtype=torch.float64)
                # a zero last layer makes the initial correction exactly zero,
                # so the base function is reproduced bit for bit
                W2 = torch.zeros(width, dtype=torch.float64)
                b2 = torch.zeros(1, dtype=torch.float64)
                params += [W1, b1, W2, b2]
                offset += 3 * width + 1
        self.nparams = offset
        self.w0 = torch.cat(params)
        self.scales = [s["scale"] for s in self.specs]
        self.bases = [s["base"].numpy().copy() for s in self.specs]
        self.base = np.concatenate(self.bases) if self.bases else np.zeros(0)
        # per-node normaliser and node offsets, used by the regularisers
        self.node_scale = np.concatenate(
            [np.full(len(s["xn"]), s["scale"]) for s in self.specs])
        self.node_offsets = np.cumsum(
            [0] + [len(s["xn"]) for s in self.specs[:-1]]).astype(int)

    def smoothness_matrix(self):
        """(rows, n_nodes) second-difference operator, scaled per function.

        Each row is [1, -2, 1] / scale, so a linear (Zener) function has zero
        penalty regardless of its magnitude.  Functions with fewer than three
        nodes (the two-node creep potentials) contribute no rows.
        """
        rows = []
        for s, off in zip(self.specs, self.node_offsets):
            n = len(s["xn"])
            for i in range(1, n - 1):
                row = np.zeros(self.base.size)
                row[off + i - 1:off + i + 2] = (1.0, -2.0, 1.0)
                rows.append(row / s["scale"])
        return np.vstack(rows) if rows else np.zeros((0, self.base.size))

    def _slices(self, w):
        k = self.width
        return [(w[s["offset"]:s["offset"] + k],
                 w[s["offset"] + k:s["offset"] + 2 * k],
                 w[s["offset"] + 2 * k:s["offset"] + 3 * k],
                 w[s["offset"] + 3 * k:s["offset"] + 3 * k + 1])
                for s in self.specs]

    def values(self, w):
        out = []
        for s, (W1, b1, W2, b2) in zip(self.specs, self._slices(w)):
            z = self.torch.tanh(s["xn"][:, None] * W1[None, :] + b1[None, :])
            # residual form: the correction vanishes for a zero last layer, so
            # a Zener warm start is reproduced exactly; softplus keeps the
            # correction bounded below and the final clamp keeps the node
            # values non-negative (the spline fit's projection does the same)
            corr = self.torch.nn.functional.softplus(z @ W2 + b2[0]) - self.log2
            out.append(self.torch.clamp(
                s["base"] + s["scale"] * corr, min=0.0))
        return self.torch.cat(out)

    def nodes(self, w):
        return self.values(w).detach().numpy().copy()

    def chain(self, theta):
        """(18*nb, nweights) Jacobian d(nodes)/d(weights)."""
        w = self.torch.tensor(np.asarray(theta, dtype=np.float64),
                              requires_grad=True)
        return self.torch.autograd.functional.jacobian(
            self.values, w).detach().numpy()

    def fit_to(self, target, steps=2000, lr=0.05):
        """Least-squares warm start, in per-function normalised units."""
        t = self.torch.tensor(np.asarray(target, dtype=np.float64))
        scale = self.torch.tensor(self.scales, dtype=self.torch.float64)
        # repeat each spec scale over its node count
        counts = [len(s["xn"]) for s in self.specs]
        scale = self.torch.repeat_interleave(scale,
                                             self.torch.tensor(counts))
        w = self.w0.clone().requires_grad_(True)
        opt = self.torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = self.torch.mean(((self.values(w) - t) / scale) ** 2)
            loss.backward()
            opt.step()
        with self.torch.no_grad():
            self.w0 = w.detach().clone()
        rms = float(self.torch.sqrt(self.torch.mean(
            ((self.values(self.w0) - t) / scale) ** 2)))
        return rms


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["eq", "neq"])
    ap.add_argument("--model", required=True, choices=["spline", "zener", "nn"])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="experiment keys (default: eq for mode eq, otherwise all training rates)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--nbranches", type=int, default=2,
                    help="number of Maxwell branches in the solver setup")
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--lambda0", type=float, default=1e-2)
    ap.add_argument("--max-rel-step", type=float, default=0.5)
    ap.add_argument("--line-search", type=int, default=0,
                    help="number of halvings tried on a rejected LM step "
                         "(0 = off, the historical behaviour)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set-eq-from", default=None,
                    help="directory whose equilibrium values are copied into interp_neq")
    ap.add_argument("--set-eq-nh", type=float, default=None,
                    help="write a neo-Hookean equilibrium with this mu into interp_neq")
    ap.add_argument("--init-zener", default=None,
                    help="mu1..munb,c1..cnb: write neo-Hookean Maxwell branches "
                         "before fitting (warm start, e.g. from a converged Zener fit)")
    ap.add_argument("--nn-width", type=int, default=8,
                    help="hidden width of the per-function tanh MLP (model nn)")
    ap.add_argument("--nn-seed", type=int, default=0)
    ap.add_argument("--nn-reg", type=float, default=0.0,
                    help="L2 penalty weight on the normalised node correction "
                         "(node-base)/scale; 0 = off")
    ap.add_argument("--nn-smooth", type=float, default=0.0,
                    help="L2 penalty weight on normalised second differences "
                         "of the node values; 0 = off")
    ap.add_argument("--nn-val-rate", default=None,
                    help="experiment key used for validation-based early "
                         "stopping (e.g. ldot3p75); only used by --model nn")
    ap.add_argument("--nn-val-every", type=int, default=3,
                    help="run the validation experiment every this many "
                         "accepted iterations")
    ap.add_argument("--nn-patience", type=int, default=0,
                    help="stop after this many validation checks without a "
                         "relative improvement of --nn-min-delta (0 = off)")
    ap.add_argument("--nn-min-delta", type=float, default=1e-3,
                    help="relative improvement required to reset the patience "
                         "counter")
    ap.add_argument("--nn-select", choices=["val", "total"], default="val",
                    help="checkpoint criterion: 'val' = validation cost only, "
                         "'total' = training cost + validation cost (the four "
                         "rates that are observed; the test rate is never used)")
    ap.add_argument("--init-nn", default=None,
                    help="theta.npy of a finished nn fit; skips the warm-start "
                         "regression and writes these weights instead")
    args = ap.parse_args()

    if args.mode == "eq" and args.model != "spline":
        raise SystemExit("the zener model is only defined for the neq mode")

    tree = os.path.abspath(args.tree)
    fit = os.path.abspath(args.fit_dir)
    solver = args.solver or os.path.join(tree, "fe_solver", "build", "variants", "platewithhole")
    calib_input = os.path.join(tree, "calibration", "input")
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(fit, "experiments.json")) as f:
        table = json.load(f)

    if args.mode == "eq":
        keys = args.experiments or ["eq"]
        interp = table["eq"]["interp"]
    else:
        keys = args.experiments or ["ldot0p15", "ldot1p5", "ldot15p0"]
        interp = table["ldot0p15"]["interp"]
        if args.set_eq_from:
            i1, i2 = read_eq_values(args.set_eq_from)
            write_eq_values(interp, i1, i2)
        if args.set_eq_nh is not None:
            x = i1_nodes(interp, kind="eq")
            i2 = np.loadtxt(os.path.join(interp, EQ_I2))
            write_eq_values(interp, 0.5 * args.set_eq_nh * (x - 3.0),
                            np.zeros_like(i2))
        if args.init_zener:
            vals = [float(v) for v in args.init_zener.split(",")]
            if len(vals) != 2 * args.nbranches:
                raise SystemExit(
                    f"--init-zener expects {2 * args.nbranches} values "
                    f"(mu1..mu{args.nbranches},c1..c{args.nbranches})")
            write_zener(interp, vals, args.nbranches)

    for k in keys:
        ensure_prepared(table[k], solver, calib_input, args.out)
    experiments = [Experiment(k, table[k], solver, calib_input) for k in keys]

    val_experiments = []
    if args.model == "nn" and args.nn_val_rate:
        if args.nn_val_rate in keys:
            raise SystemExit("--nn-val-rate must not be one of --experiments")
        ensure_prepared(table[args.nn_val_rate], solver, calib_input, args.out)
        val_experiments = [Experiment(args.nn_val_rate, table[args.nn_val_rate],
                                      solver, calib_input)]

    # ---- parameterisation ----
    if args.mode == "eq":
        i1, i2 = read_eq_values(interp)
        if args.init_scale != 1.0:
            i1, i2 = i1 * args.init_scale, i2 * args.init_scale
            write_eq_values(interp, i1, i2)
        columns = eq_columns()
        n_params = 17

        def get_theta():
            a, b = read_eq_values(interp)
            return np.concatenate([a, b])

        def put_theta(theta):
            write_eq_values(interp, theta[:12], theta[12:])

        def chain(theta):
            return np.eye(17)

    elif args.model == "spline":
        nb = args.nbranches
        columns = neq_columns(nb)
        n_params = 18 * nb
        if args.init_scale != 1.0:
            br = read_neq_values(interp, nb)
            for b in br:
                for k in br[b]:
                    br[b][k] = br[b][k] * args.init_scale
            write_neq_values(interp, br)

        def get_theta():
            return neq_vector(read_neq_values(interp, nb), nb)

        def put_theta(theta):
            write_neq_values(interp, neq_branches_from_vector(theta, nb))

        def chain(theta):
            return np.eye(18 * nb)

    elif args.model == "nn":
        import torch

        nb = args.nbranches
        columns = neq_columns(nb)
        n_params = 18 * nb
        width = args.nn_width
        scales = None
        base = None
        if args.init_nn:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.init_nn)),
                                    "nn_config.json")
            if not os.path.exists(cfg_path):
                cfg_path = os.path.splitext(args.init_nn)[0] + "_config.json"
            if not os.path.exists(cfg_path):
                raise SystemExit(f"{cfg_path} is required with --init-nn")
            with open(cfg_path) as f:
                cfg = json.load(f)
            width = int(cfg["width"])
            scales = cfg["scales"]
            base = cfg["base"]
        nn = NeuralConstitutive(torch, interp, nb, width=width,
                                seed=args.nn_seed, scales=scales, base=base)
        if args.init_nn:
            w = np.load(args.init_nn)
            if w.size != nn.nparams:
                raise SystemExit(
                    f"{args.init_nn} has {w.size} weights, expected "
                    f"{nn.nparams} (width {width}, {nb} branches)")
            nn.w0 = torch.tensor(np.asarray(w, dtype=np.float64))
            nodes = nn.nodes(nn.w0)
            print(f"nn resume: {nn.nparams} weights, node max "
                  f"{np.max(nodes):.4e}, min {np.min(nodes):.4e}")
        else:
            # the residual form starts exactly at the base function (typically
            # a converged Zener solution written by --init-zener); the warm
            # start is only needed when the base is not the target
            print(f"nn init: {nn.nparams} weights, base from current values")

        def get_theta():
            return nn.w0.detach().numpy().copy()

        def put_theta(theta):
            nodes = nn.nodes(torch.tensor(np.asarray(theta, dtype=np.float64)))
            write_neq_values(interp, neq_branches_from_vector(nodes, nb))

        def chain(theta):
            return nn.chain(theta)

        alpha = args.nn_reg
        beta = args.nn_smooth
        smooth = nn.smoothness_matrix() if beta > 0 else None

        def reg_residual(theta_vec):
            """Normalised regularisation residual at a weight vector."""
            if alpha == 0.0 and beta == 0.0:
                return np.zeros(0)
            nodes = nn.nodes(
                torch.tensor(np.asarray(theta_vec, dtype=np.float64)))
            parts = []
            if alpha:
                parts.append(
                    np.sqrt(alpha) * (nodes - nn.base) / nn.node_scale)
            if beta:
                parts.append(np.sqrt(beta) * (smooth @ nodes))
            return np.concatenate(parts)

        def reg_jacobian(C):
            """d(regularisation residual)/d(weights) from d(nodes)/d(weights)."""
            if alpha == 0.0 and beta == 0.0:
                return np.zeros((0, C.shape[1]))
            parts = []
            if alpha:
                parts.append(np.sqrt(alpha) * (C / nn.node_scale[:, None]))
            if beta:
                parts.append(np.sqrt(beta) * (smooth @ C))
            return np.vstack(parts)

    else:  # zener
        nb = args.nbranches
        columns = neq_columns(nb)
        n_params = 18 * nb
        # initial theta from the current branch values, per-branch node spacing
        br = read_neq_values(interp, nb)
        mu0 = []
        for b in range(nb):
            x = i1_nodes(interp, kind="neq", branch=b) - 3.0
            mu0.append(2.0 * br[b]["I1"][1] / x[1])
        theta0 = np.array(mu0 + [br[b]["creep"][1] for b in range(nb)])

        def get_theta():
            return theta0.copy()

        def put_theta(theta):
            write_zener(interp, theta, nb)

        def chain(theta):
            return zener_chain(interp, nb)

    # ---- initial run ----
    theta = get_theta()
    put_theta(theta)
    r, J, metrics, failed = evaluate(experiments, args.out, n_params, columns, args.jobs)
    if failed:
        raise RuntimeError(f"initial evaluation failed: {failed}")
    cost = float(r @ r)
    if args.model == "nn":
        _rreg = reg_residual(theta)
        reg_cost = float(_rreg @ _rreg)
    else:
        reg_cost = 0.0
    total = cost + reg_cost
    history = [{"iter": 0, "cost": cost, "reg_cost": reg_cost,
                "total": total, "accepted": True, "lambda": args.lambda0,
                "metrics": metrics, "theta": theta.tolist(), "val_cost": None}]
    print(f"iter 0 cost {cost:.6e} reg {reg_cost:.3e} total {total:.6e}")
    for k, m in metrics.items():
        print(f"  {k}: force rel(mean/absmax) {m['force_rel_mean']:+.3%}/{m['force_rel_max']:.3%}, "
              f"u mean {m['u_abs_mean']:.3f} mm")

    def run_val():
        rv, _, _, failedv = evaluate(val_experiments, args.out, n_params,
                                     columns, 1)
        return np.inf if failedv else float(rv @ rv)

    def select_score(data_cost, v_cost):
        # 'val' selects on the held-out rate alone; 'total' selects the
        # checkpoint that minimises the four observed rates (three training
        # rates + the validation rate).  The test rate is never used.
        return v_cost if args.nn_select == "val" else data_cost + v_cost

    val_cost = None
    best = None
    best_score = np.inf
    checks_without = 0
    if val_experiments:
        val_cost = run_val()
        history[0]["val_cost"] = val_cost
        best_score = select_score(cost, val_cost)
        best = {"theta": theta.copy(), "cost": cost, "reg_cost": reg_cost,
                "metrics": metrics, "iter": 0, "val": val_cost}
        print(f"iter 0 val {val_cost:.6e}")

    use_reg = args.model == "nn" and (args.nn_reg > 0 or args.nn_smooth > 0)
    lam = args.lambda0
    for it in range(1, args.iters + 1):
        C = chain(theta)
        Jt = J @ C                      # d r / d theta
        A = Jt.T @ Jt
        g = Jt.T @ r
        if use_reg:
            # augmented normal equations for data + regularisation
            rreg = reg_residual(theta)
            Jr = reg_jacobian(C)
            A = A + Jr.T @ Jr
            g = g + Jr.T @ rreg
        # damping must be relative to the curvature scale: the nn model starts
        # with a rank-deficient Jacobian (only the last layer is active), and
        # an absolute floor of one would put the damping far above the real
        # curvature and turn the step into numerical noise
        diag = np.diag(A).copy()
        floor = max(float(diag.max()), 1e-300) * 1e-8
        diag = np.maximum(diag, floor)
        step = -np.linalg.solve(A + lam * np.diag(diag), g)
        # trust region on the physical parameters
        scale = np.maximum(np.abs(theta), 0.05 * np.max(np.abs(theta)))
        ratio = np.max(np.abs(step) / scale)
        if ratio > args.max_rel_step:
            step *= args.max_rel_step / ratio
        predicted = -(2.0 * step @ g + step @ A @ step)
        # optional backtracking: a rejected trial halves the step before the
        # damping is increased (the nn model's first Gauss-Newton steps
        # overshoot badly and the pure lambda update wastes many iterations)
        trial = 1.0
        while True:
            theta_try = theta + trial * step
            if args.mode == "neq" and args.model == "zener":
                theta_try = np.maximum(theta_try, 1e-8)
            elif args.model != "nn":
                theta_try = np.maximum(theta_try, 0.0)
            put_theta(theta_try)
            r2, J2, m2, failed2 = evaluate(experiments, args.out, n_params,
                                           columns, args.jobs)
            if failed2:
                cost2 = np.inf
            else:
                cost2 = float(r2 @ r2)
            if use_reg:
                _rr2 = reg_residual(theta_try)
                reg2 = float(_rr2 @ _rr2)
            else:
                reg2 = 0.0
            total2 = cost2 + reg2
            if total2 < total or trial <= 0.5 ** args.line_search:
                break
            trial *= 0.5
        if total2 < total:
            rel = (cost - cost2) / max(cost, 1e-30)
            prev_cost = cost
            theta, r, J, cost, metrics = theta_try, r2, J2, cost2, m2
            reg_cost, total = reg2, total2
            lam = max(lam / 3.0, 1e-8)
            accepted = True
            print(f"iter {it} cost {cost:.6e} reg {reg_cost:.3e} "
                  f"total {total:.6e} (rel {rel:+.3e}) lambda {lam:.2e} accepted")
            for k, m in metrics.items():
                print(f"  {k}: force mean {m['force_rel_mean']:+.3%}, "
                      f"|max| {m['force_rel_max']:.3%}, u mean {m['u_abs_mean']:.3f} mm")
        else:
            lam = lam * 10.0
            accepted = False
            print(f"iter {it} rejected (total {total2:.6e} vs {total:.6e}), lambda {lam:.2e}")
        entry = {"iter": it, "cost": cost, "reg_cost": reg_cost,
                 "total": total, "trial_cost": cost2, "accepted": accepted,
                 "lambda": lam, "metrics": metrics, "theta": theta.tolist(),
                 "predicted": float(predicted), "val_cost": None}
        history.append(entry)
        with open(os.path.join(args.out, "fit_history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if val_experiments and accepted and it % args.nn_val_every == 0:
            val_cost = run_val()
            entry["val_cost"] = val_cost
            score = select_score(cost, val_cost)
            print(f"iter {it} val {val_cost:.6e} (best {best_score:.6e})")
            if score < best_score * (1.0 - args.nn_min_delta):
                best_score = score
                best = {"theta": theta.copy(), "cost": cost,
                        "reg_cost": reg_cost, "metrics": metrics,
                        "iter": it, "val": val_cost}
                checks_without = 0
            else:
                checks_without += 1
            with open(os.path.join(args.out, "fit_history.json"), "w") as f:
                json.dump(history, f, indent=1)
            if args.nn_patience and checks_without >= args.nn_patience:
                print(f"early stop after {checks_without} checks without "
                      "validation improvement")
                break
        if accepted and rel < 1e-5:
            print("converged")
            break

    # final state: restore the accepted theta (a rejected last trial may have
    # left its values in the interpolation files).  With a validation rate the
    # selectable state is the one with the best validation cost, not the last
    # accepted iterate.
    final_iter = history[-1]["iter"]
    selected_iter = final_iter
    if best is not None:
        selected_iter = best["iter"]
        if selected_iter != final_iter:
            print(f"selecting iteration {selected_iter} by validation "
                  f"({best['val']:.6e}); final iterate was {final_iter} "
                  f"(cost {cost:.6e})")
            theta = best["theta"]
            cost = best["cost"]
            reg_cost = best["reg_cost"]
            metrics = best["metrics"]
    put_theta(theta)
    np.save(os.path.join(args.out, "theta.npy"), theta)
    doc = {"model": args.model, "theta": theta.tolist(),
           "cost": cost, "reg_cost": reg_cost, "metrics": metrics,
           "final_iter": final_iter, "selected_iter": selected_iter,
           "selection": args.nn_select,
           "validation_rate": args.nn_val_rate,
           "validation_cost": None if best is None else best["val"]}
    if args.model == "nn":
        doc["nn_width"] = nn.width
        doc["nweights"] = int(theta.size)
        with open(os.path.join(args.out, "nn_config.json"), "w") as f:
            json.dump({"width": nn.width, "nb": nb, "kinds": list(NN_KINDS),
                       "scales": nn.scales, "base": nn.base.tolist(),
                       "reg": args.nn_reg, "smooth": args.nn_smooth}, f,
                      indent=1)
    with open(os.path.join(args.out, "final_metrics.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
