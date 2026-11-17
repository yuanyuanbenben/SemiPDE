"""The shared runner behind every simulation study.

Every case in ``cases.py`` differs in exactly three ways: which PDE it solves,
how its configuration is assembled, and which extra columns its tables report.
Everything else is the same work --

* enumerate the independent units of a *part* (``estimate``, ``inference``, ...)
  in a fixed order;
* shard them over any number of processes, each appending finished jobs to its
  own JSONL as it goes, so a killed shard restarts cheaply;
* rebuild the exact configuration a previous run used, so "already done" is a
  test rather than a guess;
* aggregate the JSONL into the tables the paper reports.

Before this module each case carried its own copy of that machinery
(``legacy/RD_main.py``, ``RD_coverage_out.py``, ``*_variance.py``, ...), and the
copies drifted: they disagreed on whether ``n`` meant ``n_traj`` or
``n_total``, whether the epoch budget was recorded, and which of them produced
intervals at all.  A study now supplies three hooks, and only the parts and
summary tables that are genuinely its own:

:attr:`Study.make_spec`
    ``(args, part) -> PDESpec``.  Depends on the part, because the forced case's
    ``calibration`` deliberately runs a different ``theta`` and no forcing.
:attr:`Study.make_config`
    ``(args, *, spec, sigma, M, pi) -> CaseConfig``.
:attr:`Study.score`
    ``(model, split, spec) -> dict`` of case-specific columns, merged into the
    record.  The common columns -- ``theta_error``, the mechanism error, the
    prediction error -- are computed here, for every case, by
    :func:`score_common`.

``scripts/exp_study.py`` wires the cases up and is the entry point;
``scripts/run_example2.sh`` (worth renaming to ``run_study.sh``) shards it.

Two conventions this module enforces, because getting them wrong is what made
the legacy tables hard to reconcile:

``n_total``
    Every record carries ``n_total = M * n_traj``, the size of the estimation
    sample and the ``n`` of Theorems 2--4.  ``n_traj`` alone is the paper's
    table convention and is *not* the sample size.
``config``
    Every record carries the full settings dict, and a restart compares it
    against the current options before skipping a job.  Without it a restart
    silently mixes a 200-epoch run with a 1000-epoch one in the same file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import CaseConfig
from .data import TRUTH, generate, split_trajectories
from .grids import TimeGrid
from .inference import estimate_covariance
from .metrics import (flat_param, interval_report, qq_points, studentized,
                      theta_error)
from .model import ProfilingModel
from .pde import PDESpec, parse_inner_inputs
from .sieve import SieveSpec, scaled_hidden
from .train import train

__all__ = [
    "Study",
    "aggregate",
    "build_parser",
    "enumerate_jobs",
    "fit",
    "flatten_observations",
    "forced_tables",
    "generate_split",
    "job_key",
    "main",
    "score",
    "score_common",
    "trace",
    "with_overrides",
]

#: Parts whose jobs are identical across cases plus the pieces that only the
#: forced (input-carrying) cases have.
BASE_PARTS = ("estimate", "inference")
FORCED_PARTS = ("ablation", "calibration")

#: Column headings for the metric columns of the tables.  Kept here so that a
#: new case can add a column without also having to grow the printer.
METRIC_LABELS = {
    "theta_error": "|dtheta|",
    "mech_error": "f rel.err",
    "q_error": "q rel.err",
    "u_approx": "u approx",
    "u_rms": "u rms",
    "R_n": "R_n",
}


# ---------------------------------------------------------------------------
# a study
# ---------------------------------------------------------------------------
@dataclass
class Study:
    """A case plus the parts and tables that are specific to it."""

    key: str
    title: str
    #: ``(args, part) -> PDESpec``.  Must depend on nothing else, so that
    #: :func:`resume_state` can rebuild a part's configuration without running
    #: a job of it.
    make_spec: Callable[[argparse.Namespace, str], PDESpec]
    #: ``(args, *, spec, sigma, M, pi) -> CaseConfig``.
    make_config: Callable[..., CaseConfig]
    #: Case-specific columns, merged over the common ones.
    score: Optional[Callable[..., Dict[str, float]]] = None
    #: Extra job types, by name.  Each takes ``(job, args)`` and returns a
    #: record.
    extra_parts: Mapping[str, Callable[[Mapping, argparse.Namespace], Dict]] = field(
        default_factory=dict
    )
    #: Extra tables for :func:`aggregate`, called after the common ones.
    extra_summary: Optional[Callable[[List[Dict], argparse.Namespace], Dict]] = None
    #: The common parts a case supports, unless it opts out of one.
    parts: Sequence[str] = BASE_PARTS
    #: Which columns the ``estimate`` table reports, in order.
    metrics: Sequence[str] = ("theta_error", "mech_error", "u_approx", "R_n")

    @property
    def supported_parts(self) -> Tuple[str, ...]:
        return tuple(self.parts) + tuple(self.extra_parts)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Study {self.key}: {self.title}>"


# ---------------------------------------------------------------------------
# generating, fitting, scoring
# ---------------------------------------------------------------------------
def generate_split(spec, cfg: CaseConfig, *, seed: int = 0, device=None,
                   method: str = "rk4", noise_seed: Optional[int] = None):
    """Reference trajectories, split into train/validation/test.

    The split is by *curve*, not by point: every trajectory is either fitted or
    held out, which is what makes the held-out prediction error an honest
    estimate of the approximation error and, in Algorithm 2, makes the
    covariance sample independent of the sample the fit saw.

    ``noise_seed`` redraws only the noise on the design ``seed`` fixes; see
    :func:`~semipde.data.generate`.
    """
    traj = generate(spec, cfg, n_curves=cfg.M + cfg.n_val + cfg.n_test,
                    seed=seed, device=device, method=method,
                    noise_seed=noise_seed)
    return split_trajectories(traj, cfg.M, cfg.n_val, cfg.n_test, seed=seed)


def fit(
    spec,
    cfg: CaseConfig,
    *,
    seed: int = 0,
    epochs: int = 200,
    steps: Optional[int] = None,
    lr: float = 1e-3,
    lambda_n: Optional[float] = None,
    stopping_tau: Optional[float] = None,
    batch_size: Optional[int] = None,
    network=None,
    device=None,
    solve_method: str = "rk4",
    select: str = "best",
    verbose: bool = False,
    noise_seed: Optional[int] = None,
    true_f: str = "truth",
):
    """Generate, split, fit.  Returns ``(model, split, result)``.

    ``noise_seed`` separates the observation noise from the design ``seed``
    fixes.  The *initialisation* stays on ``seed``, so holding ``seed`` and
    varying ``noise_seed`` leaves everything but the observations fixed.

    ``true_f`` chooses which equation *produced the data*: ``"truth"`` solves the
    case's own unknown mechanism, ``"zero"`` solves the ``F = 0`` model that the
    parametric family covers.  The estimator is unaffected -- the mechanism is
    still estimated, which is the point of the ``F = 0`` comparison: it measures
    what the proposal loses, on data it could have explained parametrically, by
    not knowing that.  The fitted numbers are not comparable between the two, so
    they belong in different output files.
    """
    traj = generate(spec, cfg, n_curves=cfg.M + cfg.n_val + cfg.n_test, seed=seed,
                    f=(None if true_f == "zero" else TRUTH),
                    device=device, method=solve_method, noise_seed=noise_seed)
    split = split_trajectories(traj, cfg.M, cfg.n_val, cfg.n_test, seed=seed)
    model = ProfilingModel(spec, cfg, network=network, seed=seed, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("always", RuntimeWarning)
        result = train(
            model,
            split.train,
            split.val,
            config=cfg,
            epochs=epochs,
            steps=steps,
            lr=lr,
            lambda_n=lambda_n,
            stopping_tau=stopping_tau,
            batch_size=batch_size,
            seed=seed,
            select=select,
            verbose=verbose,
        )
    if not result.fitted:
        print(f"  !! seed {seed}: fit diverged, returned the initial iterate",
              flush=True)
    return model, split, result


def flatten_observations(data) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stored trajectories to the model's rollout layout.

    ``Trajectories`` keeps time as an explicit axis -- ``(B, T, d_y, *grid)`` --
    because that is how the data are collected and plotted.  The drift and the
    mechanism, though, see one snapshot at a time, ``(B, d_y, *grid)``.  Folding
    the two leading axes is what lets ``f_hat`` and ``f_0`` be compared at the
    points the data actually cover, rather than on a probe grid that has to be
    chosen differently for every case.
    """
    states = data.states
    b, t = int(states.shape[0]), int(states.shape[1])
    u = states.reshape(b * t, *states.shape[2:])
    z = data.z
    if z.ndim == 1 and z.shape[0] == b:
        # ``reshape`` walks the batch first, so each curve's ``T`` snapshots are
        # contiguous and ``repeat_interleave`` reproduces that order exactly.
        z = z.repeat_interleave(t)
    return u, z


def score_common(model, split, spec, *, mech_reference=None) -> Dict[str, float]:
    """The columns every case reports, on the held-out curves.

    ``theta_error``
        ``||theta_hat - theta_0||`` over whatever keys the case parameterises,
        so a two-parameter case reports its Euclidean error rather than only its
        first coordinate.
    ``mech_error``
        ``||f_hat - f_0|| / ||f_0||`` at the **observed** points.  Evaluating at
        data rather than on a probe grid is what makes one formula work for
        every case: the forced case's mechanism reads the coordinate, so a state
        probe is wrong for it, while the RD and advection mechanisms read the
        state, so a coordinate probe is ``0/0`` at the origin for them.  At the
        observed points both are nondegenerate.
    ``u_rms``, ``u_approx``
        prediction error against the noisy and the clean held-out trajectories.
        The gap between them is the noise floor that no fit can cross.

    ``mech_reference`` replaces the mechanism the error is measured against.
    ``"zero"`` (the ``F = 0`` experiment) measures ``f_hat`` against the zero
    function, because there the fitted mechanism should vanish and comparing it
    with the case's nominal ``f_0`` would report ``f_0`` itself as the error.
    The denominator stays ``||f_0||``, so the column reads as the size of the
    spurious mechanism the estimator found, as a share of the one the case is
    named after.  A tensor is also accepted, for a reference that is neither.
    """
    with torch.no_grad():
        pred = model(split.test.initial, split.test.t, split.test.z)
        noisy = (pred - split.test.states).pow(2).mean()
        approx = (pred - split.test.clean).pow(2).mean()
        u, z = flatten_observations(split.test)
        f_hat = model.mechanism(u, z)
        f_0 = spec.truth_f(u, z)
        if mech_reference == "zero":
            reference = torch.zeros_like(f_0)
        elif mech_reference is None:
            reference = f_0
        else:
            reference = mech_reference
        scale = float(f_0.pow(2).mean().sqrt())
        mech = (float((f_hat - reference).pow(2).mean().sqrt() / scale)
                if scale > 0.0 else float("nan"))

    theta_hat = model.theta_detached()
    # Values stay in the shape the case parameterises them (a float, or a list
    # for Case 4's 3-vectors) so ``theta_error`` can compare coordinate by
    # coordinate, while the record carries the flattened vector as well.
    truth = {k: flat_param(v) for k, v in spec.truth().items()}
    values = [x for v in theta_hat.values() for x in flat_param(v)]
    return {
        # A scalar column for the single-parameter cases the paper reports, and
        # the full vector for everything else, so nothing is silently dropped.
        "theta_hat": values[0] if len(values) == 1 else float("nan"),
        "theta_vector": values,
        "theta_true": [x for v in truth.values() for x in v],
        "theta_error": theta_error(theta_hat, truth) if truth else float("nan"),
        "mech_error": mech,
        "u_rms": float(noisy.sqrt()),
        "u_approx": float(approx.sqrt()),
    }


def score(study: Study, model, split, spec, *, mech_reference=None) -> Dict[str, float]:
    """The common columns plus whatever the case adds."""
    out = score_common(model, split, spec, mech_reference=mech_reference)
    if study.score is not None:
        out.update(study.score(model, split, spec))
    return out


def trace(result) -> Dict[str, List[float]]:
    """The per-epoch training record, kept with the row.

    The radius is not a property of the *task*, it is a property of the
    *iterate*: ``J(F)`` starts at zero and grows as the centred network moves
    away from ``phi_0``, so a run stopped early reports a smaller ``R_n`` for no
    reason but the epoch count.  That makes a size study over ``n`` confounded
    by convergence -- the larger ``n`` is further from converged at the same
    budget -- unless the trajectory is visible.  Recording it costs one float
    per epoch and removes the confound: the slope can be read at matched
    convergence instead of at matched epochs, and the convergence itself can be
    checked rather than assumed.
    """
    keys = ("epoch", "val_loss", "val_mse", "penalty", "eta_grad_sq")
    return {
        "trace_" + k: [float(h[k]) for h in result.history] for k in keys
    }


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
def enumerate_jobs(study: Study, args: argparse.Namespace) -> List[Dict]:
    """Every independent unit of work of ``args.part``, in a fixed order.

    The order matters for one reason: consecutive jobs are dealt to *different*
    shards, so the slow jobs have to be spread out rather than clustered.  Every
    loop below therefore puts ``rep`` innermost -- the repetitions of one ``M``
    (or one width) go to different workers together.  Trailing ``rep`` also
    keeps the shards finishing at roughly the same time.

    ``sigma`` and ``M`` are written into *every* job, including the ablation's
    and the calibration's, so that :func:`job_setup` has one shape to read.  The
    benefit is not tidiness: it puts them in the job key, and a job key that
    does not name its own ``M`` cannot be resumed against a run with a different
    one.
    """
    if args.part not in study.supported_parts:
        raise ValueError(f"study {study.key!r} has no part {args.part!r}")
    jobs: List[Dict] = []
    part = args.part
    widths: List[Optional[int]] = list(args.widths) if args.widths else [None]
    # Which estimator each row is for.  ``method`` is part of the job dict, hence
    # of the job *key*, so a run that compares methods can be resumed without a
    # baseline row being read as a finished row of the proposal -- their keys
    # differ even when ``sigma``, ``M`` and ``rep`` coincide.
    methods: List[str] = [str(m) for m in (getattr(args, "methods", None) or ("ours",))]
    if part == "inference" and any(m != "ours" for m in methods):
        raise ValueError(
            "the `inference` part is the proposal's own Algorithm-2 interval; "
            f"no baseline in baselines.py produces a `se`, so {methods} cannot "
            "be run there.  Use the `estimate` part for the comparison table."
        )
    if part in ("estimate", "inference"):
        for method in methods:
            for sigma in args.sigmas:
                for M in args.sample_sizes:
                    for width in widths:
                        for rep in range(args.repeats):
                            job = {"part": part, "method": method, "sigma": sigma,
                                   "M": M, "rep": rep}
                            if width is not None:
                                job["width"] = int(width)
                            jobs.append(job)
    elif part == "ablation":
        for pi in (args.pis[:-1] if args.no_negative_control else args.pis):
            for rep in range(args.repeats):
                jobs.append({"part": "ablation", "pi": pi, "rep": rep,
                             "sigma": args.sigma_ablation, "M": args.M_ablation})
    elif part == "calibration":
        for pi in args.pis_cal:
            for rep in range(args.repeats):
                jobs.append({"part": "calibration", "pi": pi, "rep": rep,
                             "sigma": args.sigma_cal, "M": args.M_cal})
    return jobs


def job_setup(study: Study, job: Mapping, args: argparse.Namespace):
    """The ``(spec, config)`` a job runs under.

    One definition, read by the runner *and* by the resume check, which is the
    only way the two can be guaranteed to agree: a comparison that rebuilds the
    config by some second route will eventually disagree with the run it is
    meant to be describing, and then it silently recomputes good rows.
    """
    spec = study.make_spec(args, job["part"])
    cfg = study.make_config(args, spec=spec, sigma=job["sigma"], M=job["M"],
                            pi=job.get("pi", args.pi), width=job.get("width"))
    return spec, cfg


def job_key(job: Mapping) -> str:
    """A stable name for a job, used as the resume key.

    It names *what* the job is -- part, and whichever of ``sigma``/``M``/
    ``pi``/``rep`` it has -- but says nothing about the budget it was run with.
    That is what :func:`resume_state` adds by comparing the stored ``config``.
    """
    return "|".join(f"{k}={job[k]}" for k in sorted(job))


def run_job(study: Study, job: Mapping, args: argparse.Namespace) -> Dict:
    part = job["part"]
    if part in ("estimate", "inference"):
        return _job_estimate_or_inference(study, job, args)
    runner = study.extra_parts.get(part)
    if runner is None:
        raise ValueError(f"study {study.key!r} has no part {part!r}")
    return runner(job, args)


def _job_estimate_or_inference(study: Study, job: Mapping,
                               args: argparse.Namespace) -> Dict:
    rep, M = job["rep"], job["M"]
    spec, cfg = job_setup(study, job, args)
    # ``--design-policy`` decides what ``rep`` varies.  Under ``follow`` it moves
    # the design *and* the noise, which is an ordinary repetition and gives the
    # unconditional coverage the paper's C.R. column reports.  Under ``fixed``
    # the design and the initialisation are pinned to ``--seed`` and ``rep``
    # redraws only the observations, so the repetitions sample
    # ``Var(theta_hat | design)`` -- the quantity Algorithm 2's ``se`` is an
    # estimate of.  Running both and reporting them side by side is the only way
    # to tell a short interval from an estimator whose *centre* moves with the
    # design; see docs/size.md Sec. 5.
    fixed = args.design_policy == "fixed"
    fit_seed = args.seed if fixed else args.seed + rep
    t0 = time.time()
    method = job.get("method", "ours")
    true_f = getattr(args, "true_f", "truth")
    if method == "ours":
        model, split, result = fit(
            spec, cfg, seed=fit_seed, epochs=args.epochs,
            steps=getattr(args, "steps", None),
            batch_size=getattr(args, "batch_size", None), lr=args.lr,
            lambda_n=args.lambda_n, stopping_tau=args.stopping_tau,
            device=args.device, solve_method=args.solve_method,
            noise_seed=(rep if fixed else None), true_f=true_f,
        )
    else:
        # The same design and the same noise as the proposal gets for this
        # ``(sigma, M, rep)``, so the two rows of the table differ only in the
        # estimator.  Generating here rather than inside ``fit`` is what makes
        # that exact.
        traj = generate(spec, cfg, n_curves=cfg.M + cfg.n_val + cfg.n_test,
                        seed=fit_seed, device=args.device,
                        f=(None if true_f == "zero" else TRUTH),
                        method=args.solve_method,
                        noise_seed=(rep if fixed else None))
        split = split_trajectories(traj, cfg.M, cfg.n_val, cfg.n_test,
                                   seed=fit_seed)
        from .baselines import fit_baseline
        model, result = fit_baseline(
            method, spec, cfg, split,
            epochs=(args.baseline_epochs if args.baseline_epochs is not None
                    else args.epochs),
            lr=args.lr, seed=fit_seed,
            pinn_hidden=tuple(args.pinn_hidden) if args.pinn_hidden else (32, 128, 128, 32),
            pinn_batch=args.pinn_batch,
            refit_epochs=(args.refit_epochs if args.refit_epochs is not None
                          else (args.baseline_epochs if args.baseline_epochs is not None
                                else args.epochs)),
            theta_init=args.pinn_theta_init,
            pinn_select=args.pinn_select,
            cascading_lam=args.cascading_lam,
            cascading_mu=args.cascading_mu,
            cascading_modes=args.cascading_modes,
            cascading_basis=args.cascading_basis,
            cascading_sieve=args.cascading_sieve,
            cascading_iters=args.cascading_iters,
            verbose=args.verbose_inner,
        )
    t_fit = time.time() - t0
    rec = {
        "case": study.key, "part": job["part"], **job,
        "n_traj": int(cfg.n_traj), "n_total": M * cfg.n_traj,
        "fitted": bool(result.fitted), "seconds_fit": t_fit,
        "config": config_record(study, cfg, args, spec),
        # The sieve radius of the selected iterate.  ``J(F) = ||phi - phi_0||^2``
        # is what ``lambda_n`` penalises, so its square root *is* the
        # supplement's ``R_n``; recording it here is what lets the size
        # condition be *measured* across the ``n`` grid rather than only
        # asserted.
        "J": float(result.penalty), "R_n": float(result.radius),
        "width": int(max(cfg.hidden)), "hidden": list(cfg.hidden),
        "best_epoch": int(result.best_epoch),
        "best_val": float(result.best_val),
    }
    # A baseline records no per-epoch trace -- ``fit_pinn`` keeps its own history
    # in a different shape and Baseline 2 fits curve by curve -- so the trace
    # columns are the proposal's alone.  The other columns are computed for every
    # method by the same function, which is the point of the comparison.
    if method == "ours":
        rec.update(trace(result))
    # The ``F = 0`` experiment scores against zero, not against the mechanism
    # whose absence is the experiment; see ``score_common``.
    rec.update(score(study, model, split, spec,
                     mech_reference=("zero" if true_f == "zero" else None)))
    if method == "cascading":
        # The two smoothing parameters and the two basis sizes are what the
        # estimate is conditional on, so the row carries them: the same case can
        # be re-read from the record without the command line.
        rec["cascading"] = {
            "modes": int(model.fit.n_modes),
            "basis_t": int(model.fit.n_basis_t),
            "sieve_terms": int(model.fit.sieve.n_terms),
            "lam": float(model.fit.lam),
            "mu": float(model.fit.mu),
            "residual": float(model.fit.residual),
            "outer_evaluations": int(model.fit.outer_evaluations),
        }
    if job["part"] == "estimate":
        return rec

    # ---- inference: fit on the train curves, build Sigma_hat_eff on fresh ones
    print(f"    fit {t_fit:.0f}s, starting Algorithm 2 "
          f"(theta_hat={rec['theta_hat']:.4f})", flush=True)
    # The Algorithm-2 sample follows the same policy: ``fixed`` reuses one drawn
    # design for every repetition, which is what `probe_variance.py`'s group A
    # does.  Note the consequence recorded in docs/size.md -- under ``fixed``
    # ``se`` depends on the model and the one design and nothing else, so its
    # spread across repetitions is not evidence about the design.
    inf = generate(spec, cfg, n_curves=M,
                   seed=10_000 + args.seed + (0 if fixed else rep),
                   device=args.device)
    t0 = time.time()
    res = estimate_covariance(
        model, inf, delta=args.delta, epochs=args.inf_epochs, lr=args.inf_lr,
        n_est=cfg.M * cfg.n_traj, linear_term=not args.no_linear,
        lambda_n=args.inf_lambda, verbose=args.verbose_inner,
        inner_hidden=(tuple(args.inf_hidden) if args.inf_hidden else None),
        inner_solve=args.inner_solve,
        basis_degree=args.basis_degree,
        cov_estimator=args.inf_cov,
    )
    rec["seconds_inference"] = time.time() - t0
    iv = res.interval(alpha=args.alpha)
    est = np.asarray(iv["estimate"], dtype=np.float64).reshape(-1)
    se_all = np.asarray(iv["se"], dtype=np.float64).reshape(-1)
    rec.update({
        "n_inf": int(res.n_inf),
        "n_est": int(res.n_est),
        # Top level as well as in ``config``: the summary groups on it, and a
        # field buried in a nested dict cannot be a group key.
        "design_policy": args.design_policy,
        # Also top level: a size study has to be able to group by the inner
        # class, and ``inner_network_params`` is the only in-row evidence that
        # the requested class actually reached the network builder.  Read off
        # the spec for the same reason as in ``config_record`` -- the case
        # supplies a default, so ``args`` alone would report nothing at all for
        # a widened run.
        "mechanism_inputs": getattr(spec, "mechanism_inputs",
                                    getattr(args, "mechanism_inputs", None)),
        "inner_inputs": inner_inputs_label(spec),
        "inner_network_params": int(res.diagnostics.get("inner_network_params", 0)),
        # Top level so that a summary can group on it.  Under 'exact' the two
        # mismatch columns below are the class floor and its unregularised
        # value; under 'network' they are an upper bound on it, and nothing in
        # the row says by how much.
        "inner_solve": str(args.inner_solve),
        "inner_floor_rms": float(res.diagnostics.get("inner_floor_rms_0", np.nan)),
        "mismatch_rms": float(res.diagnostics.get("mismatch_rms_0", np.nan)),
        # Every coordinate is kept, and the scalar columns the tables read are
        # coordinate 0 -- which is the only coordinate when the case has one
        # parameter, as every case the paper reports an interval for does.
        "theta_names": list(res.names),
        "theta_hat_all": est.tolist(),
        "se_all": se_all.tolist(),
        "lower_all": np.asarray(iv["lower"], dtype=np.float64).reshape(-1).tolist(),
        "upper_all": np.asarray(iv["upper"], dtype=np.float64).reshape(-1).tolist(),
        "se": float(se_all[0]),
        "lower": float(np.asarray(iv["lower"]).reshape(-1)[0]),
        "upper": float(np.asarray(iv["upper"]).reshape(-1)[0]),
        "shrinkage": float(res.diagnostics["mismatch_shrinkage_0"]),
        "sigma_hat": float(res.sigma),
        "sigma_all_points": float(res.diagnostics.get("sigma_all_points",
                                                       res.sigma)),
        "noise_fraction": float(res.diagnostics.get("noise_fraction", 1.0)),
    })
    if len(res.names) > 1:
        print(f"    note: {len(res.names)} parameters {res.names}; the interval "
              f"table reports coordinate 0, the full vectors are in the record",
              flush=True)

    if int(getattr(args, "boot_reps", 0)) > 0:
        rec.update(_bootstrap_block(model, split, spec, cfg, args, rec))
    return rec


def _bootstrap_block(model, split, spec, cfg: CaseConfig, args,
                     rec: Mapping) -> Dict:
    """The score-free covariance, and how it compares with Algorithm 2's.

    Run only when asked for, because a bootstrap replicate is a full fit.  The
    comparison is the deliverable: ``boot_over_inf`` near 1 says Algorithm 2's
    plug-in score is estimating the variance the estimator actually has, and a
    factor well above 1 says the inner fit -- not the sampling noise -- is what
    sets the reported ``se``.  The two are computed on the *same* curves so that
    the only difference between them is the route taken to the covariance.
    """
    from .resample import curve_resample

    t0 = time.time()
    # The estimation curves are the ones the fit saw, so that the bootstrap's
    # sample size matches the one the reported ``se`` refers to.  The validation
    # curves are held fixed: letting them move would fold validation noise into
    # a spread that is meant to be sampling noise.
    boot = curve_resample(
        spec, cfg, split.train, val=split.val,
        n_rep=max(2, args.boot_reps), mode=args.boot_mode,
        n_groups=args.boot_groups, epochs=args.epochs, lr=args.lr,
        lambda_n=args.lambda_n, stopping_tau=args.stopping_tau,
        select="best", vary_init=bool(args.boot_vary_init),
        seed=args.seed, point=np.asarray(rec["theta_hat_all"], dtype=np.float64)
        if "theta_hat_all" in rec else None,
        device=args.device, verbose=args.verbose_inner,
    )
    iv = boot.interval(alpha=args.alpha)
    se = np.asarray(iv["se"], dtype=np.float64).reshape(-1)
    se_inf = float(rec.get("se", float("nan")))
    out = {
        "seconds_bootstrap": time.time() - t0,
        "boot_se_all": se.tolist(),
        "boot_se": float(se[0]),
        "boot_lower": float(np.asarray(iv["lower"]).reshape(-1)[0]),
        "boot_upper": float(np.asarray(iv["upper"]).reshape(-1)[0]),
        "boot_mode": boot.mode,
        "boot_n_replicates": int(boot.diagnostics["n_replicates"]),
        "boot_per_replicate_n": int(boot.diagnostics["per_replicate_n"]),
        # ``se`` from Algorithm 2 over this one: the headline diagnostic.  NaN
        # rather than an exception when the inference part did not run, so the
        # record stays readable.
        "boot_over_inf": float(se[0] / se_inf) if se_inf > 0 else float("nan"),
    }
    print(f"    bootstrap({boot.mode}, {int(boot.diagnostics['n_replicates'])}"
          f" reps, {time.time() - t0:.0f}s): se={se[0]:.4g} vs Algorithm 2's "
          f"{se_inf:.4g}  ->  {out['boot_over_inf']:.3f}x", flush=True)
    return out


# ---------------------------------------------------------------------------
# provenance and resume
# ---------------------------------------------------------------------------
#: What the record calls the paper's inner class -- ``h_j`` reading exactly what
#: ``f_hat`` reads.  ``None`` is the spec's spelling (a Python default has to be
#: falsy), the command line's is ``"state"`` (it needs a word to turn a widened
#: case back), and this is the record's: the matched class is *not* necessarily
#: the state, since the supplement's coordinate-reading cases match there too, so
#: naming it "state" would misname them.
MATCHED_INPUTS = "matched"


def inner_inputs_label(spec: Optional[PDESpec]) -> Optional[str]:
    """Algorithm 2's inner class, spelled for the record.

    ``None`` is written as ``"matched"`` rather than left null: in a JSONL column
    a null cannot be told from a row written before the field existed, and the
    whole point of the column is that a reader can group the size study by it.
    """
    if spec is None:
        return None
    value = getattr(spec, "inner_inputs", None)
    return MATCHED_INPUTS if value is None else value


def config_record(study: Study, cfg: CaseConfig, args: argparse.Namespace,
                  spec: Optional[PDESpec] = None) -> Dict:
    """The settings that determine a fit, stored with every record.

    Without this a run's provenance has to be reconstructed from its directory
    name and its approximation error.  ``runs/bias_ep1000`` was exactly that
    case: the epoch budget -- the one thing that run existed to vary -- appeared
    nowhere in its output, so the effect of the budget could only be inferred
    from ``q_error`` halving against the 200-epoch run.

    ``sigma``, ``M`` and ``rep`` are *not* here -- they are in the job key
    instead -- which is what lets :func:`resume_state` compare a stored record
    against the current options without running anything.  ``hidden`` is the
    exception and is recorded even though ``--widths`` makes it vary by job: the
    job key carries the width too, so the two agree, and a record that names the
    architecture it was fitted with is worth more than one rule held
    consistently.  The Algorithm-2 knobs are included because they move ``se``
    as much as the epoch budget moves ``theta_hat``: rows produced with a
    different ``delta`` are not comparable, and a restart must not skip over
    them.
    """
    return {
        "case": study.key,
        "epochs": int(args.epochs), "lr": float(args.lr),
        # ``fit`` takes the weight from ``--lambda-n`` and falls back to the
        # config default only when it is ``None``; recording ``cfg.lambda_n``
        # alone wrote 0.0 for every run, including the ``lam*`` directories that
        # existed to vary it.
        "lambda_n": float(args.lambda_n if args.lambda_n is not None
                          else cfg.lambda_n),
        "stopping_tau": args.stopping_tau,
        "ode_method": cfg.ode_method, "ode_step": float(cfg.ode_step),
        "data_step": float(cfg.data_step), "solve_method": args.solve_method,
        # Case 1 has two initial-condition families and they give different
        # statistical problems: on the legacy one the spread of ``theta_hat`` is
        # about a quarter of the supplement's and the bias is 20x smaller.  The
        # flag lives on ``args`` and never reached the record, so two runs that
        # agreed on every recorded field could disagree on this -- and on
        # ``runs/rd_legacy_ep200`` it is the *only* thing that separates them
        # from a supplement run at the same ``data_step``.
        "legacy_ic": bool(getattr(args, "legacy_ic", False)),
        # Read off the *spec*, not off ``args``: a case may supply a default, so
        # recording the command line would write the same thing for every run
        # that did not pass the flag and leave two runs that differ only in the
        # class looking identical -- exactly the failure ``resume_state`` exists
        # to prevent.
        "mechanism_inputs": getattr(spec, "mechanism_inputs",
                                   getattr(args, "mechanism_inputs", None)),
        "inner_inputs": (inner_inputs_label(spec) if spec is not None
                         else getattr(args, "inner_inputs", None)),
        "activation": cfg.activation, "repu_power": int(cfg.repu_power),
        "hidden": list(cfg.hidden), "normalize": bool(cfg.normalize),
        "init": cfg.init, "theta_init": cfg.theta_init,
        "seed": int(args.seed), "design_policy": args.design_policy,
        "true_f": getattr(args, "true_f", "truth"),
        "n_t": int(cfg.time.n_obs), "t_end": float(cfg.time.t1),
        "grid": list(cfg.grid.shape), "n_x": int(cfg.grid.shape[0]),
        "n_val": int(cfg.n_val), "n_test": int(cfg.n_test),
        "delta": float(args.delta), "inf_epochs": int(args.inf_epochs),
        "inf_lr": float(args.inf_lr), "inf_lambda": float(args.inf_lambda),
        "linear_term": not args.no_linear,
        # ``list(tuple(...))`` rather than the raw value: the record is JSON, and
        # ``None`` (the estimator's own widths) has to stay distinguishable from
        # an explicit width that happens to equal it.
        "inf_hidden": (list(args.inf_hidden) if args.inf_hidden else None),
        # ``inner_solve`` changes what ``mismatch_rms_j`` *means* -- the exact
        # solve reports the class floor, the network reports a fit that is
        # always above it -- so two runs differing only here must never be
        # averaged into one row.
        "inner_solve": str(getattr(args, "inner_solve", "network")),
        "basis_degree": int(getattr(args, "basis_degree", 8)),
        "inf_cov": str(args.inf_cov),
        "boot_reps": int(getattr(args, "boot_reps", 0)),
        "boot_mode": str(getattr(args, "boot_mode", "bootstrap")),
        "boot_groups": int(getattr(args, "boot_groups", 8)),
        "boot_vary_init": int(getattr(args, "boot_vary_init", 1)),
        "alpha": float(args.alpha), "sieve_kappa": float(args.sieve_kappa),
        "sieve_power": int(args.sieve_power),
        # The PINN's checkpoint rule decides *which iterate* its ``theta_hat``
        # is, so two rows that differ only here are not comparable.  Records
        # written before the rule existed do not carry the key, and
        # ``configs_agree`` skips absent keys by design.
        "pinn_select": str(getattr(args, "pinn_select", "train")),
        "pinn_theta_init": str(getattr(args, "pinn_theta_init", "zero")),
    }


def configs_agree(stored: Mapping, expected: Mapping) -> bool:
    """Whether a stored record came from the same options as this run.

    Only the keys the stored record actually has are compared, so records
    written before a field was added stay valid instead of being recomputed.
    """
    for key, want in expected.items():
        if key not in stored:
            continue
        got = stored[key]
        if isinstance(want, float) or isinstance(got, float):
            if not math.isclose(float(got), float(want), rel_tol=1e-12, abs_tol=0.0):
                return False
        elif got != want:
            return False
    return True


def resume_state(study: Study, args: argparse.Namespace,
                 jobs: Sequence[Mapping]) -> Tuple[Dict[str, Dict], set]:
    """The config each job will run under, and the keys already finished under it.

    ``job_key`` names a job (``M``, ``sigma``, ``rep``) but says nothing about
    the budget it was run with, so on its own it cannot tell a 200-epoch row
    from a 1000-epoch one.  Comparing the stored config first is what keeps a
    restart from silently skipping everything in an existing directory.

    The comparison is **per job**, not per run.  A study that varies a
    configuration axis across its jobs -- the ``--widths`` sweep does -- has
    jobs with different configs in one file, and a single run-level config would
    mark every row but the first as stale and recompute it on each restart,
    forever, without ever converging.  Reading the config from
    :func:`job_setup` rather than rebuilding it here also means the check cannot
    drift away from the thing it is checking.
    """
    def _expected(job: Mapping) -> Dict:
        spec, cfg = job_setup(study, job, args)
        return config_record(study, cfg, args, spec)

    expected = {job_key(j): _expected(j) for j in jobs}
    done, stale = set(), 0
    if args.out and os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                want = expected.get(row.get("key"))
                if want is not None and configs_agree(row.get("config") or {}, want):
                    done.add(row["key"])
                else:
                    stale += 1
    if stale:
        print(f"note: {stale} row(s) in {args.out} were written under a "
              f"different config and will be recomputed", flush=True)
    return expected, done


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
# what the paper's comparison table calls each row
METHOD_LABELS = {
    "ours": "SemiPDE",
    "para": "Baseline 1",
    "nonpara": "Baseline 2",
    "pinn": "Baseline 3",
}


def compare_table(est: List[Dict], args: argparse.Namespace) -> Dict:
    """The proposal-versus-baselines table of the main text.

    ``tab:simu:compared_baseline`` is laid out with the methods as rows and one
    column pair per ``(n, sigma)``, entries in units of ``1e-2``.  That layout is
    a *transpose* of the runner's grouping, so the quantisation is done here
    rather than at print time in the aggregation above.

    Two properties of the numbers are worth stating, because they are not
    accidents of the printing:

    * a missing entry (``nan``) is printed as ``--``.  Baseline 2 is a
      nonparametric fit of the trajectory and estimates no parameter, so its
      ``theta`` columns are not small, they are *undefined*; a reader would take
      a zero there for a perfect estimate.
    * the errors are means over whatever repetitions the file holds, and a file
      with two repetitions cannot support the paper's numbers.  The repetition
      count is therefore printed per row instead of being assumed.
    """
    out: Dict[str, object] = {}
    if not est:
        return out

    def cell(v: object) -> str:
        # not a zero but the absence of a definition: see the docstring
        return "      --  " if not np.isfinite(float(v)) else f"{1e2 * float(v):>10.3f}"

    methods = []
    for r in est:
        m = r.get("method", "ours")
        if m not in methods:
            methods.append(m)
    order = [m for m in METHOD_LABELS if m in methods]
    order += [m for m in methods if m not in METHOD_LABELS]

    print("\n== comparison with the baselines ==")
    print("  errors x 1e-2; `--` is not a zero but a quantity the method does "
          "not define")
    out["compare"] = []
    # ``n_total`` is part of the group, not just of the label: the table's two
    # columns per case are two *sample sizes*, and ``job_key`` does not name the
    # grid, so two grids can share a ``(sigma, M)``.  Grouping on the pair alone
    # would put both columns in one group and ``by_method`` would keep whichever
    # came last, silently dropping a column.
    for (sigma, M, n), g in sorted(_group(est, ("sigma", "M", "n_total")).items()):
        by_method = {r.get("method", "ours"): r for r in g}
        print(f"\n  sigma={sigma:g}  M={M}  n={n}  "
              f"reps={max(r['repeats'] for r in g)}")
        print(f"    {'method':<12}{'u_err':>11}{'theta_err':>13}{'f_err':>11}")
        for m in order:
            r = by_method.get(m)
            if r is None:
                continue
            u = r.get("u_approx", float("nan"))
            th = r.get("theta_error", float("nan"))
            f = r.get("mech_error", float("nan"))
            print(f"    {METHOD_LABELS.get(m, m):<12}{cell(u)}   {cell(th)}"
                  f"   {cell(f)}")
            out["compare"].append({
                "sigma": sigma, "M": M, "n_total": n, "method": m,
                "repeats": r["repeats"],
                "u_error_x1e2": float(1e2 * u) if np.isfinite(u) else None,
                "theta_error_x1e2": (float(1e2 * th) if np.isfinite(th)
                                     else None),
                "mech_error_x1e2": (float(1e2 * f) if np.isfinite(f)
                                    else None),
            })
    return out


def _group(records, keys):
    out: Dict[tuple, List[Dict]] = {}
    for r in records:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def _theta_coordinate0(rec: Mapping) -> float:
    """The first estimated coordinate of a fit, whichever way it was recorded.

    ``theta_vector`` is the flattened vector and is written on every record;
    ``theta_hat`` is the scalar column and is ``nan`` on a case with more than
    one parameter.  Reading the vector first keeps a vector case's inference row
    a measurement rather than a ``nan``, and the fallback covers records written
    before ``theta_vector`` existed.
    """
    vec = rec.get("theta_vector")
    if vec:
        return float(np.asarray(vec, dtype=np.float64).reshape(-1)[0])
    return float(rec.get("theta_hat", float("nan")))


def _ms(vals):
    """Mean and standard error of the mean, ignoring non-finite entries."""
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    sem = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem


def sieve_summary(est: List[Dict], args: argparse.Namespace) -> Dict:
    """How the fitted radius ``R_n`` behaves across the ``n`` grid.

    The supplement's premise is ``R_n <= C n^q`` for a *fixed* ``q``, so what
    has to be checked empirically is that ``R_n`` grows polynomially -- and the
    fitted exponent is the ``q`` to feed back into
    :class:`~semipde.sieve.SieveSpec`.  The radius ratio is the one part of the
    corollary that is checkable against a fitted model: it compares ``R_n`` with
    the ball the deployed width admits.

    A caveat the numbers make visible: ``R_n = ||phi - phi_0||`` is a
    *parameter-space* norm with no canonical scale.  For ReLU the two layers
    adjacent to the output can absorb a factor ``c`` in opposite directions
    (``W_L sigma(W_{L-1} h) = c W_L sigma((W_{L-1}/c) h)``) without changing the
    fitted function, and measured on Example 2 that moves ``R_n`` by 6x.  So
    ``q_hat`` is meaningful as a *trend* across ``n`` only if every run is in
    the same parametrisation -- which fixed initialisation and a fixed
    architecture do give -- while the absolute radius ratio is a convention.

    Two further measurements are reported because the condition as stated cannot
    be read off a single number:

    *Where the radius starts.*  ``J(F) = ||phi - phi_0||^2`` is a distance from
    the *initial* network, so ``R_n = 0`` at epoch 0 by construction and
    ``R_max(m)`` bounds how far training may travel, not how large the network
    is.  ``R_max`` balances two opposing powers and is therefore *not* monotone
    in ``m``: for ReLU ``R_max = sqrt(m)/(log m)^3`` is smallest at
    ``m = e^6 ~ 403`` (``0.093``) and needs ``m ~ 2.4e7`` to reach ``1``.  A
    wider network can tighten the restriction, which is worth knowing before
    reaching for width to satisfy it.

    *Whether one ``C`` covers every width.*  The theorem's constant is free, so
    the satisfiable statement is that ``R_n <= C R_max(m)`` holds for a *single*
    ``C`` across the widths being quantified over -- that is, that
    ``R_n/R_max(m)`` stays bounded, which is what the per-width table reports.
    """
    usable = [r for r in est if r.get("R_n") and np.isfinite(r["R_n"])]
    if not usable:
        return {}
    spec = SieveSpec(kappa=args.sieve_kappa, power=args.sieve_power)

    # ``q_hat`` is a slope *in n at fixed m*, and ``R_n`` moves with m too, so a
    # sweep over both has to be split before either trend is read: fitting one
    # line through a mixture would report the m-dependence as an n-dependence.
    # The widest group that still has two distinct n is the one to read.
    by_width = _group(usable, ("width",)) if all("width" in r for r in usable) else {}
    width_groups = {w[0]: g for w, g in by_width.items()} if by_width else {}
    per_width: List[Dict] = []
    for m, group in sorted(width_groups.items()):
        ng = np.array([r["n_total"] for r in group], dtype=float)
        Rg = np.array([r["R_n"] for r in group], dtype=float)
        q = float("nan")
        if np.unique(ng).size >= 2:
            n_mean = np.array([ng[ng == u].mean() for u in np.unique(ng)])
            R_mean = np.array([Rg[ng == u].mean() for u in np.unique(ng)])
            q = float(np.polyfit(np.log(n_mean), np.log(R_mean), 1)[0])
        per_width.append({
            "width": int(m), "n_widths": len(group), "q_hat": q,
            "R_n_max": float(Rg.max()), "R_n_mean": float(Rg.mean()),
            "radius_ratio": float(spec.radius_ratio(float(Rg.max()), int(m))),
            "max_radius": float(spec.max_radius(int(m))),
        })

    n = np.array([r["n_total"] for r in usable], dtype=float)
    R = np.array([r["R_n"] for r in usable], dtype=float)
    width = int(max(r.get("width", 64) for r in usable))

    # Repeated runs at the same ``n`` would let a single M dominate the fit, so
    # the slope is taken over the per-``n`` means.  When the sweep covers
    # several widths this uses the single focus width -- the one resolved over
    # the most distinct ``n`` -- because pooling them would fold the ``m``
    # dependence into ``q``, which is precisely the confound the split exists to
    # avoid.
    focus = (max(per_width, key=lambda row: row["n_widths"]) if per_width else None)
    if focus is not None and len(per_width) > 1:
        q_hat = float(focus["q_hat"])
        q_at = f"m={focus['width']}"
    else:
        q_hat = float("nan")
        if np.unique(n).size >= 2:
            n_mean = np.array([n[n == u].mean() for u in np.unique(n)])
            R_mean = np.array([R[n == u].mean() for u in np.unique(n)])
            q_hat = float(np.polyfit(np.log(n_mean), np.log(R_mean), 1)[0])
        q_at = f"m={width}"

    spec = SieveSpec(kappa=args.sieve_kappa, power=args.sieve_power)
    worst = max(usable, key=lambda r: r["R_n"])
    report = spec.check(
        n=int(worst["n_total"]), m=int(worst.get("width", width)),
        radius=float(worst["R_n"]),
        lambda_n=float(args.lambda_n if args.lambda_n is not None else 0.0),
    )
    print(f"\n== size control (R_n = sqrt(J(F)), width m = {worst.get('width', width)}) ==")
    print(f"  empirical growth exponent q_hat = {q_hat:.3f} at {q_at}"
          f"  ({'polynomial, consistent with the premise' if np.isfinite(q_hat) else 'n/a'})")
    print("  " + str(report).replace("\n", "\n  "))

    # ``R_n/R_max(m)`` is the whole content of the radius condition once the
    # constant ``C`` is left free, and ``R_max`` is *not* monotone in m, so the
    # question worth asking of a sweep is whether the ratio stays bounded --
    # that is what lets one ``C`` serve every width the theorem quantifies over.
    if len(per_width) > 1:
        print(f"\n  radius ratio R_n/R_max across width "
              f"(C is free; the condition is that one C covers every m)")
        print(f"  {'m':>7}{'R_max':>10}{'R_n(max)':>11}{'ratio':>9}{'q_hat':>9}")
        for row in per_width:
            print(f"  {row['width']:>7}{row['max_radius']:>10.4g}"
                  f"{row['R_n_max']:>11.4g}{row['radius_ratio']:>9.3f}"
                  f"{row['q_hat']:>9.3f}")
        ratios = [row["radius_ratio"] for row in per_width]
        print(f"  ratio spread max/min = {max(ratios) / min(ratios):.3g}"
              f"  ({'bounded -- one C suffices' if max(ratios) / min(ratios) < 10 else 'not bounded across this range'})")
    return {
        "q_hat": q_hat,
        "width": int(worst.get("width", width)),
        "R_n_max": float(worst["R_n"]),
        "radius_ratio": float(report.cond_radius),
        "max_radius": float(report.max_radius),
        "radius_ok": bool(report.radius_ok),
        "per_width": per_width,
        "kappa": float(args.sieve_kappa),
        "power": int(args.sieve_power),
    }


def aggregate(study: Study, records: List[Dict], args: argparse.Namespace) -> Dict:
    parts = {r["part"] for r in records}
    out: Dict[str, object] = {"case": study.key, "n_records": len(records)}

    # Rows produced under different budgets are not comparable, so report that
    # rather than quietly averaging across them.
    configs = sorted(
        {json.dumps(r["config"], sort_keys=True) for r in records if r.get("config")}
    )
    if configs:
        out["config"] = (json.loads(configs[0]) if len(configs) == 1
                         else [json.loads(c) for c in configs])
    if len(configs) > 1:
        print(f"\nWARNING: this file mixes {len(configs)} configs; the summaries "
              f"below aggregate across all of them")
        for c in configs:
            print(f"  {c}")

    if "estimate" in parts:
        rows = []
        est = [r for r in records if r["part"] == "estimate"]
        cols = [m for m in study.metrics
                if any(m in r for r in est)]
        # ``method`` joins the grouping key whenever the records name one, so a
        # file that compares the proposal against a baseline does not average
        # them into a single row -- the mean of a 3e-4 and a 3e-3 error is not a
        # number about either method.
        # ``n_total`` joins the key because one file may hold two *grids* at
        # the same ``(sigma, M)`` -- the baseline table's two sample sizes are
        # two grids, not two values of ``M``.  Grouping on the pair alone would
        # average the two columns into one row and ``compare_table`` would then
        # print only the grid that happened to come last.
        est_keys = ("sigma", "M", "n_total")
        if any("method" in r for r in est):
            est_keys = ("method", "sigma", "M", "n_total")
        for grp, g in sorted(_group(est, est_keys).items()):
            labels = dict(zip(est_keys, grp))
            sigma, M, n = labels["sigma"], labels["M"], labels["n_total"]
            row = {"sigma": sigma, "M": M, "n_total": n, "repeats": len(g)}
            if "method" in labels:
                row["method"] = labels["method"]
            for key in cols:
                row[key], row[key + "_sem"] = _ms([r.get(key, float("nan"))
                                                   for r in g])
            rows.append(row)
        out["estimate"] = rows
        out["estimate_columns"] = cols
        has_method = "method" in est_keys
        print("\n== estimation and prediction ==")
        tag = f"{'method':>8} " if has_method else ""
        head = f"{tag}{'sigma':>6} {'M':>5} {'n':>9} " + " ".join(
            f"{METRIC_LABELS.get(c, c):>11}" for c in cols)
        print(head)
        for r in rows:
            body = " ".join(f"{r[c]:>11.3e}" if c != "R_n" else f"{r[c]:>11.3f}"
                            for c in cols)
            pre = f"{r['method']:>8} " if has_method else ""
            print(f"{pre}{r['sigma']:>6} {r['M']:>5} {r['n_total']:>9} {body}")
        out["sieve"] = sieve_summary(est, args)
        if has_method:
            out["compare"] = compare_table(rows, args)

    if "inference" in parts:
        rows = []
        inf_records = [r for r in records if r["part"] == "inference"]
        # When every row names its design policy the summary splits on it, so
        # concatenating a ``--design-policy follow`` run with a ``fixed`` one
        # prints the two coverage readings side by side instead of averaging
        # them into a single number that describes neither.  A file written
        # before the flag existed names no policy and is grouped as before.
        #
        # Algorithm 2's inner class splits for a stronger reason still: ``se``
        # *is* the class.  On Case 1 the widened class is the one calibrated to
        # nominal coverage and the paper's matched class reads 0.75 at the same
        # nominal, so a file mixing the two would average 0.75 and 0.95 into a
        # number that describes neither run -- see ``docs/size.md`` Sec. 5.3e.
        group_keys = ["sigma", "M"]
        for key in ("design_policy", "inner_inputs"):
            # ``is not None`` rather than ``in``: a null must not become a group
            # value, both because it is not one and because sorting the keys
            # would then compare it against the names of the real classes.
            if all(r.get(key) is not None for r in inf_records):
                group_keys.append(key)
        for grp, g in sorted(_group(inf_records, group_keys).items()):
            labels = dict(zip(group_keys, grp))
            sigma, M = labels["sigma"], labels["M"]
            truth = g[0]["theta_true"][0]
            # ``theta_vector`` rather than the scalar ``theta_hat``: the scalar
            # column is ``nan`` whenever the case has more than one parameter
            # (it is there for the single-parameter cases the paper tabulates),
            # so reading it here would make ``bias``, ``std`` and the Q--Q
            # points ``nan`` on exactly the cases whose intervals are hardest to
            # calibrate -- while ``coverage``, which is built from ``lower`` and
            # ``upper``, would still print a number.  The row would then look
            # measured with only part of it missing, which is worse than either.
            # Coordinate 0 is the coordinate ``lower``/``upper``/``se`` already
            # refer to -- the note printed at fit time says so -- so taking it
            # here makes the four columns describe one quantity.
            est0 = [_theta_coordinate0(r) for r in g]
            rep = interval_report(
                estimates=est0,
                lowers=[r["lower"] for r in g], uppers=[r["upper"] for r in g],
                ses=[r["se"] for r in g], truth=truth, alpha=args.alpha,
            )
            piv = studentized(est0, [r["se"] for r in g], truth)
            piv = piv[np.isfinite(piv)]
            theo, emp = qq_points(piv) if piv.size > 1 else (np.array([]),
                                                             np.array([]))
            row = {"sigma": sigma, "M": M, "n_est": g[0]["n_est"],
                   "n_inf": g[0]["n_inf"], "repeats": len(g), **rep,
                   "sigma_hat": _ms([r["sigma_hat"] for r in g])[0],
                   "shrinkage": _ms([r["shrinkage"] for r in g])[0],
                   "qq_theoretical": theo.tolist(), "qq_empirical": emp.tolist()}
            for key in group_keys:
                if key not in ("sigma", "M"):
                    row[key] = labels[key]
            row["mean_se_over_std"] = (
                row["mean_se"] / row["std"] if row["std"] else float("nan")
            )
            rows.append(row)
        out["inference"] = rows
        # The split keys lead the table, one column each, so the reading that
        # follows is attached to the rows it describes.
        # 14 is ``"location,state"``, the widest of the class names, so the
        # column never silently widens the table by wrapping.
        EXTRA_COLS = (("design_policy", "design", 8), ("inner_inputs", "inner", 14))
        lead_cols = [c for c in EXTRA_COLS if c[0] in group_keys]
        print("\n== inference: Algorithm-2 standard error vs the empirical spread ==")
        print("".join(f"{name:>{w}}" for _, name, w in lead_cols) +
              f"{'sigma':>6} {'M':>5} {'reps':>5} {'bias':>11} "
              f"{'std':>11} {'mean_se':>11} {'ratio':>7} {'len':>11} {'cov':>6}")
        for r in rows:
            print("".join(f"{str(r[key]):>{w}}" for key, _, w in lead_cols) +
                  f"{r['sigma']:>6} {r['M']:>5} {r['repeats']:>5} "
                  f"{r['bias']:>11.3e} {r['std']:>11.3e} {r['mean_se']:>11.3e} "
                  f"{r['mean_se_over_std']:>7.3f} {r['mean_length']:>11.3e} "
                  f"{r['coverage']:>6.2f}")
        print("  `ratio` is `mean_se / std`, and the two are not the same kind "
              "of object:\n  `se` is conditional on the design, while `std` "
              "spans designs as well as noises.\n  So `ratio < 1` is not by "
              "itself evidence that `se` is short -- part of it is\n  `std` "
              "containing a component `se` never claimed to cover. "
              "`probe_variance.py`\n  separates the two by holding one fixed; "
              "see docs/size.md.")
        if "design_policy" in group_keys:
            print("  The two `design` blocks are the same question asked twice:"
                  "\n  `fixed` holds one design and redraws the noise, so its"
                  " `std` is the spread\n  `se` claims to cover and its `cov`"
                  " is the interval's own reading; `follow`\n  redraws the"
                  " design as well, so its `std` also contains how far the"
                  " centre\n  moves, and its `cov` is what a user drawing one"
                  " design actually gets.")
        if "inner_inputs" in group_keys:
            print("  `inner` is Algorithm 2's class, i.e. what `h_j` reads, and"
                  " it is not a detail:\n  `se` *is* the class.  `matched` is"
                  " the paper's own map and is what every published\n  number"
                  " used; the rest are calibrated widenings (`docs/size.md`"
                  " Sec. 5.3e), so\n  rows under different `inner` values are"
                  " different experiments and must not be read\n  as one.")

    if study.extra_summary is not None:
        out.update(study.extra_summary(records, args) or {})

    return out


def forced_tables(records: List[Dict], args: argparse.Namespace) -> Dict:
    """The two tables only an input-carrying case has.

    Shared by every forced study rather than copied into each, because these
    two are what the *design* contributes and have to be comparable across
    cases: the loss swing along the non-identified direction, and the
    zero-background covariance against the supplement's analytic ``I_eff``.
    """
    out: Dict[str, object] = {}
    if any(r["part"] == "ablation" for r in records):
        rows = []
        for pi, g in sorted(_group([r for r in records if r["part"] == "ablation"],
                                   ("pi",)).items()):
            row = {"pi": pi[0], "repeats": len(g)}
            for key in ("theta_error", "q_error", "mech_error", "loss_swing",
                        "loss_swing_rel", "theta_hat"):
                if any(key in r for r in g):
                    row[key], row[key + "_sem"] = _ms([r.get(key, float("nan"))
                                                       for r in g])
            rows.append(row)
        out["ablation"] = rows
        print("\n== input-design ablation (fixed total sample size) ==")
        print(f"{'pi':>6} {'|dtheta|':>11} {'+-':>9} {'q rel.err':>10} "
              f"{'loss swing':>11} {'rel':>8}")
        for r in rows:
            print(f"{r['pi']:>6} {r.get('theta_error', float('nan')):>11.3e} "
                  f"{r.get('theta_error_sem', float('nan')):>9.1e} "
                  f"{r.get('q_error', r.get('mech_error', float('nan'))):>10.3f} "
                  f"{r.get('loss_swing', float('nan')):>11.3e} "
                  f"{r.get('loss_swing_rel', float('nan')):>8.2e}")

    if any(r["part"] == "calibration" for r in records):
        rows = []
        for pi, g in sorted(_group([r for r in records
                                    if r["part"] == "calibration"], ("pi",)).items()):
            row = {"pi": pi[0], "repeats": len(g)}
            for key in ("sigma_hat", "analytic_var", "ratio", "shrinkage"):
                row[key], row[key + "_sem"] = _ms([r[key] for r in g])
            rows.append(row)
        out["calibration"] = rows
        print("\n== zero-background calibration against the analytic I_eff ==")
        print(f"{'pi':>6} {'Sigma_hat':>12} {'analytic':>12} {'ratio':>8}")
        for r in rows:
            print(f"{r['pi']:>6} {r['sigma_hat']:>12.3e} "
                  f"{r['analytic_var']:>12.3e} {r['ratio']:>8.3f}")
    return out


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def with_overrides(cfg: CaseConfig, args: argparse.Namespace,
                   width: Optional[int] = None) -> CaseConfig:
    """Apply the command line's overrides to a case's own configuration.

    Every study builds the config ``cases.py`` prescribes and then passes it
    through here, so an override is honoured in exactly one place and cannot be
    applied by one case and forgotten by another -- the failure mode that lets
    two runs of "the same setting" differ.  Options left at ``None`` mean "what
    the case prescribes", which keeps the case layer the single source of truth
    for the paper's settings.

    ``--width m`` is the override that matters most for the size condition: it
    rescales ``hidden`` so its widest layer has ``m`` units, keeping depth and
    the layer-to-layer ratios, which is exactly the knob the corollary's
    ``m``-dependence is about.  A run with a fixed ``hidden`` for every ``n`` is
    ``C_m = 0`` in that statement, and that is where every published run sits.
    An explicit ``--hidden`` wins over ``--width``: it says what the widths are,
    not just what the widest one should be.

    ``width`` is that same knob, passed in for *one job* by :func:`job_setup`
    when ``--widths`` sweeps it; it takes precedence over ``--width`` because
    ``--width`` is then just the sweep's fallback.
    """
    kwargs: Dict = {
        "ode_method": args.ode_method if args.ode_method is not None else cfg.ode_method,
        "ode_step": (float(args.ode_step) if args.ode_step is not None
                     else float(cfg.ode_step)),
        "data_step": (float(args.data_step) if args.data_step is not None
                      else float(cfg.data_step)),
    }
    job_width = args.width if width is None else width
    if args.hidden is not None:
        kwargs["hidden"] = tuple(int(h) for h in args.hidden)
    elif job_width is not None:
        kwargs["hidden"] = scaled_hidden(cfg.hidden, job_width)
    if args.activation is not None:
        kwargs["activation"] = args.activation
    if args.repu_power is not None:
        kwargs["repu_power"] = int(args.repu_power)
    if args.init is not None:
        kwargs["init"] = args.init
    if args.theta_init is not None:
        kwargs["theta_init"] = args.theta_init
    if args.normalize is not None:
        kwargs["normalize"] = bool(args.normalize)
    if args.lambda_n is not None:
        # ``train`` receives the weight separately, but the config default is
        # what the record falls back to and what the sieve report reads.
        kwargs["lambda_n"] = float(args.lambda_n)
    if args.t_end is not None:
        cfg = cfg.with_(
            time=TimeGrid(cfg.time.t0, float(args.t_end), cfg.time.n_obs,
                          include_endpoint=True)
        )
    return cfg.with_(**kwargs)


def build_parser(study: Study, description: Optional[str] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("part", choices=study.supported_parts)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--methods", nargs="+", default=["ours"],
                   choices=("ours", "para", "nonpara", "pinn", "cascading"),
                   help="which estimators to run for `estimate`: 'ours' is the "
                        "proposal, 'para'/'nonpara'/'pinn' are the paper's "
                        "three baselines, and 'cascading' is parameter "
                        "cascading (baselines.fit_baseline).  Only 'ours' has an "
                        "interval, so `inference` accepts it alone.  Each value "
                        "becomes its own column of `tab:simu:compared_baseline`, "
                        "and the runs share sigma/M/rep so the comparison is "
                        "paired")
    p.add_argument("--baseline-epochs", type=int, default=None,
                   help="epoch budget for the baselines; the default is the "
                        "`--epochs` of the proposal, which is what makes the "
                        "comparison a comparison rather than a budget sweep")
    p.add_argument("--refit-epochs", type=int, default=None,
                   help="epochs to fit a fresh solution network per held-out "
                        "curve for the PINN's prediction; defaults to "
                        "`--baseline-epochs`")
    p.add_argument("--pinn-hidden", type=int, nargs="+", default=None,
                   help="PINN solution-network widths; the legacy "
                        "PINN_model.py default is 32 128 128 32")
    p.add_argument("--pinn-batch", type=int, default=None,
                   help="PINN trajectories per backward pass; `None` is one "
                        "full-batch step, and a smaller value only bounds peak "
                        "memory (the gradient is unchanged -- see fit_pinn)")
    p.add_argument("--pinn-theta-init", default="zero",
                   choices=("zero", "truth"),
                   help="PINN's initial theta: 'zero' reproduces the legacy "
                        "PINN_model.py; 'truth' starts at the oracle and is not "
                        "a fair baseline, only a diagnostic")
    p.add_argument("--pinn-select", default="train",
                   choices=("train", "val", "final"),
                   help="PINN checkpoint rule: 'train' is the legacy rule "
                        "(minimum training loss) and the default -- the "
                        "held-out curves have no trained solution network, so "
                        "a held-out loss selects the initialisation; 'val' and "
                        "'final' are diagnostics")
    p.add_argument("--cascading-lam", type=float, default=100.0,
                   help="weight of the PDE-residual penalty in parameter "
                        "cascading's inner criterion (the method's smoothing "
                        "parameter)")
    p.add_argument("--cascading-mu", type=float, default=1e-4,
                   help="weight of the second-difference penalty on the "
                        "mechanism sieve's coefficients (the other smoothing "
                        "parameter; the ridge is what keeps a free ``f`` from "
                        "absorbing the parameter)")
    p.add_argument("--cascading-modes", type=int, default=8,
                   help="Fourier modes per spatial axis for cascading's state "
                        "basis, capped per axis at (n - 1) // 2; the method's "
                        "smoothness is carried by --cascading-lam, not by "
                        "cutting the basis")
    p.add_argument("--cascading-basis", type=int, default=10,
                   help="number of cubic B-spline functions in time for the "
                        "state basis")
    p.add_argument("--cascading-sieve", type=int, default=6,
                   help="B-spline functions per axis in the sieve for the "
                        "unknown mechanism; n^d terms, d the width of the "
                        "case's mechanism inputs")
    p.add_argument("--cascading-iters", type=int, default=150,
                   help="LBFGS iterations of each state-coefficient pass "
                        "inside cascading's inner solve; the solve converges "
                        "long before this on every case")
    p.add_argument("--n-x", type=int, default=None,
                   help="grid points along x; the paper's size for this "
                        "study otherwise")
    p.add_argument("--n-t", type=int, default=None,
                   help="observation times; the paper's size for this study "
                        "otherwise (25 for `estimate`, 10 for `inference`)")
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[16, 64, 256],
                   help="training-curve counts for estimate/inference")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3])
    p.add_argument("--pi", type=float, default=0.5,
                   help="P(Z=1) used by estimate/inference; the ablation and "
                        "calibration parts sweep --pis instead")
    p.add_argument("--pis", type=float, nargs="+", default=[0.5, 0.8, 0.95, 1.0])
    p.add_argument("--pis-cal", type=float, nargs="+", default=[0.5, 0.8, 0.95])
    p.add_argument("--no-negative-control", action="store_true",
                   help="drop pi = 1 from the ablation")
    p.add_argument("--M-ablation", type=int, default=64)
    p.add_argument("--sigma-ablation", type=float, default=0.1)
    p.add_argument("--M-cal", type=int, default=64)
    p.add_argument("--sigma-cal", type=float, default=0.1)
    p.add_argument("--c-max", type=float, default=0.25)
    p.add_argument("--n-c", type=int, default=11)
    p.add_argument("--epochs", type=int, default=1000,
                   help="training budget.  `1000` is the budget at which the "
                        "interval's estimated standard error matches the spread "
                        "of `theta_hat`, i.e. at which coverage reaches its "
                        "nominal level; see docs/divergences.md Sec. 10")
    p.add_argument("--steps", type=int, default=None,
                   help="optimizer steps instead of epochs: each step is one "
                        "gradient on a freshly drawn minibatch of `--batch-size` "
                        "curves.  The large-grid cases (np, ns) cost thousands "
                        "of times more per sweep than a gradient needs, so the "
                        "minibatch estimator is the only affordable protocol "
                        "there; a size ladder has to be run with the same flags "
                        "for its coverage to describe this estimator")
    p.add_argument("--batch-size", type=int, default=None,
                   help="curves per gradient step when `--steps` is given; the "
                        "default is every training curve")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-n", type=float, default=None)
    p.add_argument("--width", type=int, default=None,
                   help="rescale hidden so its widest layer has this many "
                        "units, keeping depth and the layer ratios (see "
                        "semipde.sieve.scaled_hidden).  This is the m of the "
                        "supplement's width condition.")
    p.add_argument("--widths", type=int, nargs="+", default=None,
                   help="cross the sweep with several widths, one job each.  The "
                        "size condition is a statement about m as much as about "
                        "n, and --width alone varies m only between runs -- "
                        "which is what makes the radius ratio R_n/R_max(m) "
                        "impossible to read as a trend, since a single m gives a "
                        "single point.  Only estimate/inference are swept.")
    p.add_argument("--hidden", type=int, nargs="+", default=None,
                   help="explicit layer widths, overriding --width and the "
                        "case default")
    p.add_argument("--activation", default=None)
    p.add_argument("--repu-power", type=int, default=None)
    p.add_argument("--init", default=None)
    p.add_argument(
        "--theta-init", choices=("zero", "truth"), default=None,
        help="where the optimisation of theta starts; the case's own otherwise, "
             "which is 'zero'.  'zero' is the legacy convention and the setting "
             "the published tables were produced under -- ``theta_init='truth'`` "
             "starts the estimator at zero error, so it is for looking at the "
             "local regime and never for a bias or coverage study.",
    )
    p.add_argument(
        "--theta-true", type=float, default=None,
        help="override the case's true theta.  Only wired for Case 1 (rd), "
             "whose factory takes a scalar; its default is 1e-2, so pass 3e-3 "
             "to reproduce the paper's setting.  Write the run to its own "
             "--out directory: the resume key is the job index and the recorded "
             "config does not include theta_true.",
    )
    p.add_argument("--normalize", type=int, choices=(0, 1), default=None,
                   help="Eq. (5) scaling on (1) or off (0); off reproduces the "
                        "legacy networks")
    p.add_argument(
        "--stopping-tau", type=float, default=None,
        help="tau of Eq. (16); the run stops at the first epoch with "
             "eta*|grad L|^2 <= tau.  The default disables the rule, which is "
             "what the paper's implementation details do in practice (a fixed "
             "epoch budget with the best-validation iterate returned).  Do not "
             "reach for a small constant: eta*|grad L|^2 is 6.0e-04 at the "
             "initial iterate on the forced case, so the old config default of "
             "1e-3 fired at epoch 0 and returned an untrained model.  Scale tau "
             "to the problem -- the initial eta*|grad L|^2 is the natural unit.",
    )
    p.add_argument(
        "--sieve-kappa", type=float, default=1.0,
        help="the supplement's tightness exponent kappa for the size-control "
             "report.  The paper leaves it unspecified, so it is a reporting "
             "choice: every sieve bound is monotone in it, and the radius "
             "condition R(1+R)^kappa (log m)^kappa <= m^{1/4} tightens fast "
             "with it (R_max = 0.46 at kappa=1 against 0.13 at kappa=2, m=64).",
    )
    p.add_argument(
        "--sieve-power", type=int, default=1, choices=(1, 2),
        help="activation power l of the size-control report: 1 is the ReLU the "
             "code actually uses, 2 is the RePU for which the paper states the "
             "corollary.  Only l>=2 has a closed-form width exponent and an "
             "H_m bound; l=1 reports the radius condition only.",
    )
    p.add_argument(
        "--data-step", type=float, default=None,
        help="the *reference* solve's internal step; the case's own otherwise "
             "(2e-2 for every case).  It never enters the estimator, so it only "
             "has to be accurate, not small: with rk4, 2e-2 already matches "
             "1e-3 to 3.2e-09 on the forced case and is 20x cheaper.  Only "
             "lower it if you also pass --solve-method euler.",
    )
    p.add_argument(
        "--solve-method", choices=("rk4", "euler"), default="rk4",
        help="the reference solve's scheme.  'rk4' is the published setting; "
             "'euler' at the model's own --ode-step removes the discretisation "
             "mismatch between data and model, which is how its contribution to "
             "the theta bias is measured rather than assumed.",
    )
    p.add_argument(
        "--ode-step", type=float, default=None,
        help="the *model's* rollout step; the case's own otherwise (2e-2 for "
             "every case).  Its error is a bias in theta_hat, so this -- not "
             "--data-step -- is the knob to spend integration time on.  With "
             "the rk4 default, 2e-2 is already converged.",
    )
    p.add_argument(
        "--ode-method", choices=("euler", "rk4"), default=None,
        help="the *model's* rollout scheme; the case's own otherwise (rk4 for "
             "every case).  'rk4' costs 4 drift evaluations per step against "
             "Euler's 1 but is O(h^4) rather than O(h); at ode-step 2e-2 it "
             "matches the supplement's Euler-at-1e-3 accuracy (1008 evaluations "
             "per forward pass) for 192.  'euler' reproduces the legacy "
             "*_model.py setting, whose discretisation bias is what --ode-step "
             "was measuring.",
    )
    p.add_argument("--t-end", type=float, default=None,
                   help="observation window; the case's own default otherwise")
    p.add_argument("--legacy-ic", action="store_true",
                   help="Case 1 (rd) only: draw the initial condition from the "
                        "legacy profile instead of the supplement's.  The "
                        "published Case 1 tables were produced with the legacy "
                        "one, so a run meant to be compared against them needs "
                        "this flag.")
    p.add_argument("--mechanism-inputs", default=None,
                   choices=("state", "location,state"),
                   help="Case 1 (rd) only: inputs of the fitted mechanism "
                        "f_hat.  'state' is the paper's construction; "
                        "'location,state' lets f_hat vary with x, which makes "
                        "the interval wider but biases theta_hat (see "
                        "docs/size.md).")
    p.add_argument("--inner-inputs", default=None,
                   type=parse_inner_inputs,
                   help="Inputs of Algorithm 2's inner h_j, independently of "
                        "f_hat.  A deliberate deviation from the paper (h_0 = "
                        "d_theta f_hat, so h_j should read what f_hat reads): "
                        "it makes se conservative by an amount that depends "
                        "only on the class width, never on the data.  'matched' "
                        "('state' is an alias) is the paper's class and is the "
                        "default everywhere.  Any combination of "
                        "'state', 'location', 'derivatives' is accepted; a "
                        "wider class lowers the class floor, and where the "
                        "floor falls below the rms(M) that would calibrate se, "
                        "the gap is closed by --inf-lambda instead.  "
                        "See docs/size.md Sec. 5.3e-5.3f.")
    p.add_argument("--delta", type=float, default=1e-4,
                   help="finite-difference step of Algorithm 2")
    p.add_argument("--inf-epochs", type=int, default=40)
    p.add_argument("--inf-lr", type=float, default=1e-2)
    p.add_argument("--inf-lambda", type=float, default=0.0,
                   help="the tilde-lambda_n of Algorithm 2: the penalty on "
                        "J(h_j) inside the inner fit.  Default 0 leaves the "
                        "inner sieve radius uncontrolled, which is what every "
                        "published run used; raising it pulls h toward 0 and "
                        "so raises `se`.  Recorded in the config, because rows "
                        "produced under a different value are not comparable")
    p.add_argument("--no-linear", action="store_true",
                   help="disable the inner linear term")
    p.add_argument("--inf-hidden", type=int, nargs="+", default=None,
                   help="widths of Algorithm 2's inner h_j network.  Defaults "
                        "to the estimator's own --hidden, which is what the "
                        "paper's S_n prescribes; setting it separates the "
                        "inner sieve from the outer one and is the direct test "
                        "of whether the inner class, rather than its fit, is "
                        "what limits `se`")
    p.add_argument("--inner-solve", default="network",
                   choices=("network", "exact"),
                   help="how Algorithm 2's step 1 is solved.  'network' is the "
                        "paper's gradient descent on h_j; 'exact' replaces h_j "
                        "with a Chebyshev basis in the inner features and "
                        "solves the linear least squares in closed form, one "
                        "perturbed PDE solve per basis column.  The class is "
                        "the same and the exact solve reaches its floor, so it "
                        "removes the optimiser gap and the inner-optimiser "
                        "noise from `se` -- and costs one rollout per column "
                        "instead of `inf-epochs` passes")
    p.add_argument("--basis-degree", type=int, default=8,
                   help="total degree of the 'exact' basis, jointly in every "
                        "column of the inner feature tensor: at "
                        "--inner-inputs location the cross terms of the state "
                        "and the coordinate are what the widened inner map buys")
    p.add_argument("--inf-cov", default="efficient",
                   choices=("efficient", "robust"),
                   help="middle matrix of the sandwich behind `se`.  "
                        "'efficient' is the paper's sigma_hat^2 * gram^{-1}, "
                        "which assumes one residual variance at every point; "
                        "'robust' is the Eicker-Huber-White form "
                        "gram^{-1} mean(r_i^2 M_i M_i^T) gram^{-1}, which keeps "
                        "the pointwise weights.  The two differ by exactly the "
                        "systematic misfit, so the gap between their `se` is a "
                        "direct measure of it (see docs/size.md)")
    p.add_argument("--boot-reps", type=int, default=0,
                   help="also estimate the variance by refitting on "
                        "bootstrap-resampled curves, which never touches the "
                        "score `M_hat` (see semipde/resample.py).  0 (the "
                        "default) leaves it off: it costs one full fit per "
                        "replicate.  Its `se` is recorded alongside Algorithm "
                        "2's, so the two can be compared -- they agree only if "
                        "the inner fit really reaches its class optimum")
    p.add_argument("--boot-mode", default="bootstrap",
                   choices=("bootstrap", "partition"),
                   help="'bootstrap' resamples M curves with replacement and "
                        "refits at full sample size; 'partition' cuts them into "
                        "--boot-groups disjoint blocks, which costs one fit's "
                        "worth of data but carries the m-out-of-n bias")
    p.add_argument("--boot-groups", type=int, default=8,
                   help="blocks for --boot-mode partition")
    p.add_argument("--boot-vary-init", type=int, choices=(0, 1), default=1,
                   help="let each bootstrap replicate start from its own "
                        "initialisation (the estimator is randomised, so this "
                        "is part of its variance) or pin it (isolating the data)")
    p.add_argument("--true-f", default="truth", choices=("truth", "zero"),
                   help="which equation generates the data.  'truth' (the "
                        "default) solves the case's own unknown mechanism; "
                        "'zero' solves F = 0, the table of "
                        "tab:compared_baseline_paramodel, where the true model "
                        "is inside the parametric family and the proposal's "
                        "efficiency loss is what is being measured.  Every "
                        "estimator is still run unchanged, so the rows are not "
                        "comparable with a 'truth' run: use its own --out")
    p.add_argument("--design-policy", default="follow",
                   choices=("follow", "fixed"),
                   help="what a repetition varies.  'follow' (the default, and "
                        "what every recorded run used) redraws the design and "
                        "the observations together, so the spread is the "
                        "unconditional one the paper's C.R. column reports.  "
                        "'fixed' pins the design and the initialisation to "
                        "--seed and redraws only the noise, sampling "
                        "Var(theta_hat | design) -- the quantity `se` estimates "
                        "-- and needs its own --out, since the two policies' "
                        "rows share a job key")
    p.add_argument("--verbose-inner", action="store_true",
                   help="print the Algorithm-2 inner-fit progress")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--out", default=None, help="JSONL of finished records")
    p.add_argument("--aggregate", default=None,
                   help="read this JSONL and print the tables")
    return p


def main(study: Study, argv: Optional[Sequence[str]] = None,
         description: Optional[str] = None) -> int:
    args = build_parser(study, description).parse_args(argv)
    # The CPU is the default **on purpose**, and it is not a fallback.  Both the
    # reference solve and the model rollout are Python loops over substeps, each
    # one a few operations on a small tensor; the accelerator only ever sees
    # kernel launches.  On the shared host this was developed on (16 GPUs, all of
    # them at 100% from another user) a launch costs far more than the
    # arithmetic: one training epoch at n_x=32, n_t=25, M=64 measured 4.7 s on
    # the CPU against 14.6 s on the GPU, and the reference solve 2.7 s against
    # 177 s.  Pass --device cuda to put the *model* on a card (the reference
    # solve stays on the CPU regardless; see ``semipde.data.generate``).
    if args.device is None:
        args.device = "cpu"

    if args.aggregate:
        with open(args.aggregate) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        summary = aggregate(study, records, args)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(summary, fh, indent=2, default=float)
            print(f"\nwrote {args.out}")
        return 0

    jobs = enumerate_jobs(study, args)
    if not jobs:
        raise SystemExit(f"no jobs for part {args.part!r}")
    expected, done = resume_state(study, args, jobs)
    mine = [j for k, j in enumerate(jobs) if k % args.shards == args.shard]
    todo = [j for j in mine if job_key(j) not in done]

    # A run whose jobs span several configs (a ``--widths`` sweep) cannot be
    # described by one of them, so say how many there are rather than print the
    # first and let it stand for all.
    distinct = len({json.dumps(c, sort_keys=True) for c in expected.values()})
    shown = (json.dumps(expected[job_key(mine[0])], sort_keys=True) if mine
             else "(no jobs on this shard)")
    suffix = f"  [{distinct} distinct configs]" if distinct > 1 else ""
    print(f"case {study.key}: {shown}{suffix}", flush=True)
    print(f"shard {args.shard}/{args.shards}: part={args.part} "
          f"{len(todo)} of {len(mine)} jobs to run "
          f"({len(mine) - len(todo)} already done)", flush=True)

    # ``--out`` may name a directory that does not exist yet, and a shard that
    # dies after an hour of training because of a missing parent is a bad trade
    # for one ``mkdir``.
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
    # Unbuffered: the whole resume scheme rests on finished rows being *on disk*
    # when the process dies.  Through the default buffer a killed shard would
    # redo every job still sitting in the stream.
    fh = open(args.out, "a", buffering=1) if args.out else None
    try:
        for n, job in enumerate(todo, 1):
            t0 = time.time()
            try:
                rec = run_job(study, job, args)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{n}/{len(todo)}] FAILED {job}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            rec["key"] = job_key(job)
            rec["seconds"] = time.time() - t0
            print(f"  [{n}/{len(todo)}] {rec['key']}  "
                  f"{rec['seconds']:.0f}s  "
                  f"theta_error={rec.get('theta_error', float('nan')):.3e}",
                  flush=True)
            if fh is not None:
                fh.write(json.dumps(rec, default=float) + "\n")
                fh.flush()
    finally:
        if fh is not None:
            fh.close()
    print("shard done", flush=True)
    return 0
