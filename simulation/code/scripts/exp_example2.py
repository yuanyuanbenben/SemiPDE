#!/usr/bin/env python
"""Example 2 in simulation: forced Burgers with a switched input ``Z``.

The paper's Example 2 is

.. math::

    \\partial_t u_z - \\nu \\partial_{xx} u_z + u_z \\partial_x u_z
        = \\theta z g(x) + q(x), \\qquad Z \\in \\{-1, +1\\},

with ``nu`` and ``g`` known, ``theta`` estimated, and ``q`` a shared unknown
background that must be recovered from data at **both** input values.

The work is split into independent *jobs* so that it can be sharded over any
number of workers, and every finished job is appended to a JSONL file as it
completes, so a restart skips what is already done.  ``scripts/run_example2.sh``
does the sharding::

    # 42 CPU workers on the inference part, then the tables
    scripts/run_example2.sh 42 inference --repeats 50 --epochs 200

    # equivalently, by hand, one process per shard
    python scripts/exp_example2.py inference --repeats 50 --shard 0 --shards 16 \\
        --out runs/example2_inference.jsonl

The device is the CPU by default and that is deliberate -- see ``main`` and
``scripts/run_example2.sh`` for the measurements.  On this shared host a GPU
kernel launch costs far more than the handful of operations behind it.

Parts
-----
``estimate``
    Balanced design (``pi = 0.5``), a few sample sizes, two noise levels:
    whether ``theta``, ``q`` and ``u_z`` are recovered together.
``inference``
    Repetitions at a fixed design, comparing the mean Algorithm-2 standard error
    against the empirical spread of ``theta_hat``, plus coverage, mean interval
    length and the studentized Q--Q points.  ``theta_hat`` is fitted on the
    training curves and ``Sigma_hat_eff`` is built on curves the fit never saw;
    ``n_est`` is the size of the estimation sample, and is passed explicitly.
``ablation``
    Fixed total sample size, input frequency ``pi`` in ``{0.5, 0.8, 0.95,
    1.0}``.  The last is the deliberate negative control: at a single input the
    map ``(theta, q) -> (theta + c, q - c z_0 g)`` leaves every observation
    unchanged, so ``theta`` is not identified at all.  The fitted loss is traced
    along that direction, where it is *exactly* flat.
``calibration``
    Zero background, ``(theta_0, q_0) = (0, 0)``, where the supplement's
    analytic information is exact:
    ``I_eff = 4 pi (1 - pi) sigma^{-2} E_X |K g(X)|^2``.  The PDE is then linear,
    so ``K g`` is read straight off the theta-perturbation rollout and the
    analytic value can be compared with the Algorithm-2 covariance numerically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semipde.config import CaseConfig, InputSpec  # noqa: E402
from semipde.data import generate, split_trajectories  # noqa: E402
from semipde.grids import Grid, TimeGrid  # noqa: E402
from semipde.inference import estimate_covariance  # noqa: E402
from semipde.metrics import interval_report, qq_points, studentized  # noqa: E402
from semipde.model import ProfilingModel, coordinates  # noqa: E402
from semipde.pde import ForcedBurgers  # noqa: E402
from semipde.sieve import SieveSpec  # noqa: E402
from semipde.train import train  # noqa: E402

#: ``[0, 2 pi)`` with the paper's own profiles, so that ``g`` and ``q_0`` share
#: their ``sin x`` component.  That overlap is not incidental -- it is exactly
#: what makes ``theta`` identifiable only through the *difference* between the
#: two forcing directions, and therefore what the negative control probes.
DOMAIN = (0.0, 2 * math.pi)

Q_MODES = ((0.3, "sin"), (0.2, "cos2"), (0.1, "sin3"))

#: The paper states Example 2 but publishes no simulation settings -- no ``nu``,
#: no time window -- so both have to be chosen here, and the window is **not**
#: free.  With ``u_z(0, x) = 0`` the forcing drives the solution up, the
#: advective term steepens it, and the shock that forms at ``t ~ 1.5`` sharpens
#: without bound: at ``nu = 0.05`` the reference solve at ``n_x = 32`` is finite
#: to ``t = 2.0`` and diverges before ``t = 2.5``, so the obvious window of
#: ``[0, 2.5]`` (the one Cases 1--2 use) produces NaN data and a NaN objective.
#:
#: ``t_end = 1.0`` is the largest window that stays clear of that, and it is
#: converged at the working resolution: refining ``n_x = 32`` against
#: ``n_x = 1024`` moves the solution by ``4.7e-3`` (``1.8e-2`` at ``t = 1.5``,
#: ``1.1e+0`` at ``t = 2.0``, as the front forms).
#:
#: **The two step sizes do different jobs and are tuned in opposite
#: directions.**  ``data_step`` sets how accurate the *reference* trajectory
#: is; it never enters the estimator.  RK4 at ``2e-2`` is already accurate to
#: ``3.2e-09`` here -- ``max|clean(1e-3) - clean(2e-2)|`` on this case -- so the
#: old ``1e-3`` cost 20x the integration time of the dominant part of data
#: generation and bought a change below float noise.  The default is
#: therefore ``2e-2``.
#:
#: ``ode_step`` sets the *model's* rollout, which is what the estimator fits the
#: data *with*, so its discretisation error is a bias in ``theta_hat`` rather
#: than an accuracy loss -- it is the one worth spending on.  The supplement
#: specifies Euler at ``1e-3`` there; the legacy code used Euler at ``2e-2``,
#: which leaves a ``-5.25e-04`` bias at ``M = 256`` against a standard error of
#: ``5.37e-04``.  One full standard error is enough on its own to pull 95%
#: coverage to ~0.85.  ``ode_method="rk4"`` at the same ``2e-2`` reaches
#: ``1.7e-16`` on the same comparison for 192 drift evaluations per forward pass
#: against the supplement's 1008, so the default is RK4 and ``ode_step`` no
#: longer has to buy accuracy with a small step.  See ``runs/ode_trade``.
T_END = 1.0


# ---------------------------------------------------------------------------
# the case
# ---------------------------------------------------------------------------
def make_spec(
    n_x: int = 32,
    *,
    theta_true: float = 0.5,
    q_modes: Optional[Sequence] = None,
    nu: float = 0.05,
):
    grid = Grid.from_ranges([DOMAIN], [n_x], periodic=True)
    return ForcedBurgers(
        grid,
        nu=nu,
        theta_true=theta_true,
        q_modes=tuple(Q_MODES if q_modes is None else q_modes),
    )


def make_config(
    spec,
    *,
    sigma: float = 0.1,
    n_t: int = 25,
    M: int = 64,
    n_val: int = 16,
    n_test: int = 16,
    pi: float = 0.5,
    #: Both steps must agree with ``cases.py`` and the ``--data-step`` default;
    #: ``1e-3`` here was the last surviving copy of the pre-measurement value.
    data_step: float = 2e-2,
    ode_step: float = 2e-2,
    t_end: float = T_END,
    **extra,
) -> CaseConfig:
    inputs = InputSpec.from_pi(pi) if pi < 1.0 else InputSpec((1.0,), (1.0,))
    return CaseConfig(
        name=f"burg-nx{spec.grid.shape[0]}-pi{pi}-s{sigma}",
        grid=spec.grid,
        time=TimeGrid(0.0, t_end, n_t, include_endpoint=True),
        d_y=1,
        sigma=sigma,
        inputs=inputs,
        M=M,
        n_val=n_val,
        n_test=n_test,
        data_step=data_step,
        ode_step=ode_step,
        **extra,
    )


# ---------------------------------------------------------------------------
# fitting and scoring
# ---------------------------------------------------------------------------
def fit(
    spec,
    cfg: CaseConfig,
    *,
    seed: int = 0,
    epochs: int = 200,
    lr: float = 1e-3,
    lambda_n: Optional[float] = None,
    stopping_tau: Optional[float] = None,
    batch_size: Optional[int] = None,
    network=None,
    device=None,
    solve_method: str = "rk4",
    verbose: bool = False,
):
    """Generate, split, fit.  Returns ``(model, split, result)``."""
    total = cfg.M + cfg.n_val + cfg.n_test
    traj = generate(spec, cfg, n_curves=total, seed=seed, device=device,
                    method=solve_method)
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
            lr=lr,
            lambda_n=lambda_n,
            stopping_tau=stopping_tau,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
        )
    if not result.fitted:
        print(f"  !! seed {seed}: fit diverged, returned the initial iterate", flush=True)
    return model, split, result


def score(model, split, spec) -> Dict[str, float]:
    """Theta error, ``q`` error and prediction error on the held-out curves."""
    with torch.no_grad():
        pred = model(split.test.initial, split.test.t, split.test.z)
        noisy = (pred - split.test.states).pow(2).mean()
        # ``clean`` is the noiseless solution: the approximation error of the
        # fit, as opposed to the noise, which no model can fit.
        approx = (pred - split.test.clean).pow(2).mean()
    theta_hat = float(model.theta_detached()["theta"])
    return {
        "theta_hat": theta_hat,
        "theta_error": abs(theta_hat - float(spec.theta_true)),
        "q_error": _q_error(model, spec),
        "u_rms": float(noisy.sqrt()),
        "u_approx": float(approx.sqrt()),
    }


def _q_error(model, spec) -> float:
    """``||q_hat - q_0||_2 / ||q_0||_2`` on the grid."""
    cfg = model.config
    dev = next(model.parameters()).device
    coords = coordinates(cfg, 1, dtype=model.dtype, device=dev)
    with torch.no_grad():
        feats = spec.features(
            torch.zeros(1, spec.d_y, *cfg.grid.shape, dtype=model.dtype, device=dev),
            coords,
            torch.ones(1, dtype=model.dtype, device=dev),
        )
        q_hat = model.network(feats.reshape(-1, feats.shape[-1]))
    q_hat = q_hat.reshape(*feats.shape[:-1], -1)[0, ..., 0].cpu().numpy()
    q_true = spec.q_profile(coords)[0].cpu().numpy()
    return float(np.sqrt(np.mean((q_hat - q_true) ** 2)) / np.sqrt(np.mean(q_true**2)))


# ---------------------------------------------------------------------------
# the equivalence direction of the input-design ablation
# ---------------------------------------------------------------------------
class _Shift(torch.nn.Module):
    """The ``-c z_0 g`` half of ``(theta, q) -> (theta + c, q - c z_0 g)``.

    Added to a frozen fitted ``q_hat`` through the model's ``base_mechanism``
    slot, so the pair ``(theta_hat + c, q_hat - c z_0 g)`` is evaluated exactly,
    with no refitting.  ``z_0`` is ``E[Z]``, so the direction moves ``theta``
    while holding the *average* forcing ``theta E[Z] g + q`` fixed: at a single
    input it cancels identically, and at a balanced design it changes exactly the
    part of the drift that ``Z`` identifies.

    ``apply_pointwise`` hands the module the flattened coordinate matrix, so the
    ``sqrt`` is a no-op here and the result is reshaped back to a trailing axis
    of length one, matching ``mechanism_out_dim``.
    """

    def __init__(self, spec, c: float, z0: float, dtype=torch.float64):
        super().__init__()
        self.spec = spec
        self.c = float(c)
        self.z0 = float(z0)
        self._dtype = dtype

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return (-self.c * self.z0 * self.spec.g_profile(coords)).unsqueeze(-1)

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=self._dtype)


def _loss_along_equivalence_direction(model, spec, cfg, data, cs: Sequence[float]):
    """Fitted loss at ``(theta_hat + c, q_hat - c z_0 g)`` for each ``c``.

    At a single input every observation carries the same ``z_0``, so ``theta + c``
    and ``q - c z_0 g`` cancel in the drift exactly and this curve is
    *constant* -- measured swing ``0.0`` at ``pi = 1`` against ``0.18`` at
    ``pi = 0.5``.  The flat direction is a property of the design, not of the
    optimiser, so no amount of fitting removes it.
    """
    theta_hat = float(model.theta_detached()["theta"])
    z0 = float(np.mean(data.z.cpu().numpy()))
    fitted = model.network
    out: List[float] = []
    n_diverged = 0
    with torch.no_grad():
        for c in cs:
            shifted = ProfilingModel(
                spec, cfg,
                theta={"theta": torch.tensor(theta_hat + c, dtype=model.dtype)},
                network=_Shift(spec, c, z0, dtype=model.dtype),
                base_mechanism=fitted,
                dtype=model.dtype,
            )
            pred = shifted(data.initial, data.t, data.z)
            val = float((pred - data.states).pow(2).mean().sqrt())
            if not math.isfinite(val):
                # The shifted system carries a stronger forcing, and past a
                # certain ``c`` it shocks and the rollout diverges -- the loss
                # then says nothing about identification.  Count it and drop it
                # rather than let one infinite entry turn the whole statistic
                # into NaN without a trace.
                n_diverged += 1
                val = float("nan")
            out.append(val)
    return out, n_diverged


def _expected_kg_squared(model, spec, data, delta: float) -> float:
    """``E_X |K g(X)|^2`` from the theta-perturbation rollout.

    At the zero background the solution is *affine* in ``theta``, so
    ``(u(theta + delta) - u(theta)) / delta`` reproduces ``K g`` up to the ODE
    solver's own error.  Averaging its square over every observed point and
    curve is the ``E_X`` of the analytic formula.
    """
    with torch.no_grad():
        base = model(data.initial, data.t, data.z)
        model.theta["theta"].add_(delta)
        bumped = model(data.initial, data.t, data.z)
        model.theta["theta"].sub_(delta)
    return float(((bumped - base) / delta).pow(2).mean())


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
def enumerate_jobs(args) -> List[Dict]:
    """Every independent unit of work, in a fixed order."""
    jobs: List[Dict] = []
    if args.part in ("estimate", "inference"):
        for sigma in args.sigmas:
            for M in args.sample_sizes:
                for rep in range(args.repeats):
                    jobs.append({"part": args.part, "sigma": sigma, "M": M, "rep": rep})
    elif args.part == "ablation":
        for pi in args.pis[: -1] if args.no_negative_control else args.pis:
            for rep in range(args.repeats):
                jobs.append({"part": "ablation", "pi": pi, "rep": rep})
    elif args.part == "calibration":
        for pi in args.pis_cal:
            for rep in range(args.repeats):
                jobs.append({"part": "calibration", "pi": pi, "rep": rep})
    return jobs


def job_key(job: Dict) -> str:
    return "|".join(f"{k}={job[k]}" for k in sorted(job))


def run_job(job: Dict, args) -> Dict:
    part = job["part"]
    if part in ("estimate", "inference"):
        return _job_estimate_or_inference(job, args)
    if part == "ablation":
        return _job_ablation(job, args)
    if part == "calibration":
        return _job_calibration(job, args)
    raise ValueError(part)


def _spec_for(args, theta_true: float = 0.5, q_modes=None):
    """The PDE a job of this part solves.

    The calibration part deliberately runs the ``theta_true = 0``, no-forcing
    case; every other part runs the forced Burgers case the paper reports.
    """
    if args.part == "calibration":
        return make_spec(args.n_x, theta_true=0.0, q_modes=())
    return make_spec(args.n_x, q_modes=Q_MODES if q_modes is None else q_modes)


def _make_config(args, *, spec, sigma: float, M: int, pi: float = 0.5) -> CaseConfig:
    """The case configuration a job runs under, from the parsed options.

    Every job goes through here so that :func:`_resume_state` can rebuild the
    exact same configuration without running a job, which is what makes the
    "already done" test meaningful rather than a guess.
    """
    return make_config(
        spec, sigma=sigma, n_t=args.n_t, M=M, pi=pi,
        data_step=args.data_step, ode_step=args.ode_step,
        ode_method=args.ode_method, t_end=args.t_end,
    )


def _job_estimate_or_inference(job: Dict, args) -> Dict:
    rep, sigma, M = job["rep"], job["sigma"], job["M"]
    spec = _spec_for(args)
    cfg = _make_config(args, spec=spec, sigma=sigma, M=M)
    t0 = time.time()
    model, split, result = fit(
        spec, cfg, seed=args.seed + rep, epochs=args.epochs, lr=args.lr,
        lambda_n=args.lambda_n, stopping_tau=args.stopping_tau, device=args.device,
        solve_method=args.solve_method,
    )
    t_fit = time.time() - t0
    rec = {"part": job["part"], **job, "n_total": M * cfg.n_traj,
           "fitted": bool(result.fitted), "theta_true": float(spec.theta_true),
           "seconds_fit": t_fit, "config": _config_record(cfg, args),
           # The sieve radius of the selected iterate.  ``J(F) = ||phi-phi_0||^2``
           # is what lambda_n penalises, so its square root *is* the supplement's
           # R_n; recording it here is what lets the size condition be measured
           # across the n grid rather than only asserted.
           "J": float(result.penalty), "R_n": float(result.radius),
           "width": int(max(cfg.hidden)), "best_epoch": int(result.best_epoch)}
    rec.update(score(model, split, spec))
    if job["part"] == "estimate":
        return rec

    # ---- inference: fit on train curves, build Sigma_hat_eff on fresh ones ----
    print(f"    fit {t_fit:.0f}s, starting Algorithm 2 "
          f"(theta_hat={rec['theta_hat']:.4f})", flush=True)
    inf = generate(spec, cfg, n_curves=M, seed=10_000 + args.seed + rep,
                   device=args.device)
    t0 = time.time()
    res = estimate_covariance(
        model, inf, delta=args.delta, epochs=args.inf_epochs, lr=args.inf_lr,
        n_est=cfg.M * cfg.n_traj, linear_term=not args.no_linear,
        verbose=args.verbose_inner,
    )
    rec["seconds_inference"] = time.time() - t0
    iv = res.interval(alpha=args.alpha)
    rec.update({
        "n_inf": int(res.n_inf),
        "n_est": int(res.n_est),
        "se": float(iv["se"][0]),
        "lower": float(iv["lower"][0]),
        "upper": float(iv["upper"][0]),
        "shrinkage": float(res.diagnostics["mismatch_shrinkage_0"]),
        "sigma_hat": float(res.sigma),
    })
    return rec


def _job_ablation(job: Dict, args) -> Dict:
    rep, pi = job["rep"], job["pi"]
    spec = _spec_for(args)
    cfg = _make_config(args, spec=spec, sigma=args.sigma_ablation,
                       M=args.M_ablation, pi=pi)
    model, split, _ = fit(
        spec, cfg, seed=args.seed + rep, epochs=args.epochs, lr=args.lr,
        lambda_n=args.lambda_n, stopping_tau=args.stopping_tau, device=args.device,
    )
    sc = score(model, split, spec)
    cs = np.linspace(-args.c_max, args.c_max, args.n_c)
    losses, n_diverged = _loss_along_equivalence_direction(
        model, spec, cfg, split.test, cs
    )
    finite = np.asarray(losses, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    mid = float(np.min(finite)) if finite.size else float("nan")
    spread = float(np.ptp(finite)) if finite.size else float("nan")
    return {
        "part": "ablation", **job,
        "n_total": args.M_ablation * cfg.n_traj,
        "theta_hat": sc["theta_hat"],
        "theta_error": sc["theta_error"],
        "q_error": sc["q_error"],
        # The swing of the fitted loss along the direction that is exactly flat
        # at a single input.  Zero is the non-identifiability; the scale-free
        # ratio makes it comparable across runs.
        "loss_swing": spread,
        "loss_swing_rel": spread / mid if mid and mid > 0 else float("nan"),
        "n_diverged": n_diverged,
        "losses": [float(v) for v in losses],
    }


def _job_calibration(job: Dict, args) -> Dict:
    rep, pi = job["rep"], job["pi"]
    spec = _spec_for(args, theta_true=0.0, q_modes=())
    cfg = _make_config(args, spec=spec, sigma=args.sigma_cal,
                       M=args.M_cal, pi=pi)
    model, split, _ = fit(
        spec, cfg, seed=args.seed + rep, epochs=args.epochs, lr=args.lr,
        lambda_n=args.lambda_n, stopping_tau=args.stopping_tau, device=args.device,
    )
    inf = generate(spec, cfg, n_curves=args.M_cal, seed=20_000 + args.seed + rep,
                   device=args.device)
    res = estimate_covariance(
        model, inf, delta=args.delta, epochs=args.inf_epochs, lr=args.inf_lr,
        n_est=cfg.M * cfg.n_traj, linear_term=not args.no_linear,
    )
    kg2 = _expected_kg_squared(model, spec, inf, args.delta)
    info = _efficiency_information(pi, kg2, cfg.sigma)
    var_theory = 1.0 / info if info > 0 else float("inf")
    return {
        "part": "calibration", **job,
        "sigma_hat": float(res.cov[0, 0]),
        "analytic_var": var_theory,
        "ratio": float(res.cov[0, 0]) / var_theory if var_theory > 0 else float("nan"),
        "shrinkage": float(res.diagnostics["mismatch_shrinkage_0"]),
    }


def _efficiency_information(pi: float, kg2: float, sigma: float) -> float:
    """``E[M_0^2] / sigma^2`` for the two-arm design, from the sensitivity only.

    At the zero background the sensitivity is ``du/dtheta = Z K g``, so with
    ``a = K g`` the two arms carry ``a_+ = a`` and ``a_- = -a``.  The nuisance is
    the *same* ``q(x)`` in both arms, so ``h_0`` minimises

        pi ||a - K'h||^2 + (1 - pi) ||a + K'h||^2

    and, writing ``P`` for the projection onto the reachable set ``{K'h}``,

        E[M_0^2] = ||a_perp||^2 + 4 pi (1 - pi) ||P a||^2,      (C)

    with ``pi = P(Z = 1)``.  Both checks that matter are in that line: at
    ``pi = 1/2`` the coefficient of ``||Pa||^2`` is one, so the bound coincides
    with the *oracle* ``sigma^2 / E[(du/dtheta)^2]`` -- the variance an estimator
    that is **told** ``q`` would get -- and for ``pi != 1/2`` the estimator is
    strictly *better* than the oracle, because it also knows ``q`` is shared.
    That is the Neyman orthogonality doing work, not a paradox.

    ``a`` is a smooth Fourier mode and the mechanism reads the coordinate, so
    the network class represents it and ``P a = a``: the ``||a_perp||^2`` term is
    dropped, which makes the reported bound the tightest one (any genuine
    ``a_perp`` would only raise it).

    The first version of this function wrote ``2 L (1 - pi)`` with ``L = 2 pi``
    the domain length -- numerically ``4 pi_c (1 - pi)``, where ``pi_c`` is the
    *circle* constant.  Wrong twice over: it swapped the constant ``pi_c`` for
    the probability ``pi``, and it fed the bound a variance *below* the oracle.
    Measured on this case (``M = 64``, ``sigma = 0.1``, 50 repetitions): the
    ratio ``Sigma_hat / analytic`` came out at 6.10, 3.98, 4.11 for
    ``pi = 0.5, 0.8, 0.95``.  Under (C) the predicted ratios are
    ``2 pi_c (1 - pi) / (4 pi (1 - pi)) = pi_c / (2 pi)`` times
    ``Sigma_hat / Sigma_oracle`` -- and ``Sigma_hat / Sigma_oracle`` is measured
    independently at 0.95, 1.58, 6.54 for the same three inputs, giving
    ``5.97, 3.97, 4.11``.  The bound is the remaining unknown; ``Sigma_hat`` is
    not.
    """
    return 4.0 * pi * (1.0 - pi) * kg2 / (sigma ** 2)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _group(records, keys):
    out: Dict[tuple, List[Dict]] = {}
    for r in records:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def _ms(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    sem = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem


def _config_record(cfg, args) -> Dict:
    """The settings that determine a fit, stored with every record.

    Without this a run's provenance has to be reconstructed from its directory
    name and its approximation error.  ``runs/bias_ep1000`` was exactly that
    case: the epoch budget -- the one thing that run existed to vary -- appeared
    nowhere in its output, so the effect of the budget on the bias could only be
    inferred from ``q_error`` halving against the 200-epoch run.

    Every field here is independent of the job's ``(sigma, M, rep)`` -- those are
    in the job key instead -- which is what lets :func:`_resume_state` compare a
    stored record against the current options without running anything.  The
    Algorithm-2 knobs are included because they move ``se`` as much as the epoch
    budget moves ``theta_hat``: rows produced with a different ``delta`` are not
    comparable and a restart must not skip over them.
    """
    return {
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
        "activation": cfg.activation, "repu_power": int(cfg.repu_power),
        "hidden": list(cfg.hidden), "normalize": bool(cfg.normalize),
        "init": cfg.init, "seed": int(args.seed),
        "n_t": int(cfg.time.n_obs), "n_x": int(cfg.grid.shape[0]),
        "n_val": int(cfg.n_val), "n_test": int(cfg.n_test),
        "delta": float(args.delta), "inf_epochs": int(args.inf_epochs),
        "inf_lr": float(args.inf_lr), "linear_term": not args.no_linear,
        "alpha": float(args.alpha), "sieve_kappa": float(args.sieve_kappa),
        "sieve_power": int(args.sieve_power),
    }


def _configs_agree(stored: Dict, expected: Dict) -> bool:
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


def _resume_state(args) -> Tuple[Dict, set]:
    """The expected config plus the set of job keys already finished under it.

    ``job_key`` names a job (``M``, ``sigma``, ``rep``) but says nothing about
    the budget it was run with, so on its own it cannot tell a 200-epoch row
    from a 1000-epoch one.  Comparing the stored config first is what keeps a
    restart from silently skipping everything in an existing directory.
    """
    # Any job's parameters will do: every field of the record is job-independent
    # (``sigma``, ``M``, ``pi`` and ``rep`` live in the job key instead).
    cfg = _make_config(args, spec=_spec_for(args), sigma=args.sigmas[0],
                       M=args.sample_sizes[0])
    expected = _config_record(cfg, args)
    done, stale = set(), 0
    if args.out and os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if _configs_agree(row.get("config") or {}, expected):
                    done.add(row["key"])
                else:
                    stale += 1
    if stale:
        print(f"note: {stale} row(s) in {args.out} were written under a "
              f"different config and will be recomputed", flush=True)
    return expected, done


def _sieve_summary(est: List[Dict], args) -> Dict:
    """How the fitted radius ``R_n`` behaves across the ``n`` grid.

    The supplement's premise is ``R_n <= C n^q`` for a *fixed* ``q``, so what
    has to be checked empirically is that ``R_n`` grows polynomially -- and the
    fitted exponent is the ``q`` to feed back into :class:`~semipde.sieve.SieveSpec`.
    The radius ratio is the one part of the corollary that is checkable against
    a fitted model: it compares ``R_n`` with the ball the deployed width admits.
    """
    usable = [r for r in est if r.get("R_n") and np.isfinite(r["R_n"])]
    if not usable:
        return {}
    n = np.array([r["n_total"] for r in usable], dtype=float)
    R = np.array([r["R_n"] for r in usable], dtype=float)
    width = int(max(r.get("width", 64) for r in usable))

    q_hat = float("nan")
    if np.unique(n).size >= 2:
        slope = np.polyfit(np.log(n), np.log(R), 1)
        q_hat = float(slope[0])

    spec = SieveSpec(kappa=args.sieve_kappa, power=args.sieve_power)
    worst = max(usable, key=lambda r: r["R_n"])
    report = spec.check(
        n=int(worst["n_total"]), m=width, radius=float(worst["R_n"]),
        lambda_n=float(args.lambda_n),
    )
    print(f"\n== size control (R_n = sqrt(J(F)), width m = {width}) ==")
    print(f"  empirical growth exponent q_hat = {q_hat:.3f}"
          f"  ({'polynomial, consistent with the premise' if np.isfinite(q_hat) else 'n/a'})")
    print("  " + str(report).replace("\n", "\n  "))
    return {
        "q_hat": q_hat,
        "width": width,
        "R_n_max": float(worst["R_n"]),
        "radius_ratio": float(report.cond_radius),
        "max_radius": float(report.max_radius),
        "radius_ok": bool(report.radius_ok),
        "kappa": float(args.sieve_kappa),
        "power": int(args.sieve_power),
    }


def aggregate(records: List[Dict], args) -> Dict:
    parts = {r["part"] for r in records}
    out: Dict[str, object] = {"n_records": len(records)}

    # Rows produced under different budgets are not comparable, so report that
    # rather than quietly averaging across them.
    configs = sorted(
        {json.dumps(r["config"], sort_keys=True) for r in records if r.get("config")}
    )
    if configs:
        out["config"] = json.loads(configs[0]) if len(configs) == 1 else [
            json.loads(c) for c in configs
        ]
    if len(configs) > 1:
        print(f"\nWARNING: this file mixes {len(configs)} configs; the summaries "
              f"below aggregate across all of them")
        for c in configs:
            print(f"  {c}")

    if "estimate" in parts:
        rows = []
        est = [r for r in records if r["part"] == "estimate"]
        for (sigma, M), g in sorted(_group(est, ("sigma", "M")).items()):
            row = {"sigma": sigma, "M": M, "n_total": g[0]["n_total"], "repeats": len(g)}
            for key in ("theta_error", "q_error", "u_approx", "u_rms", "R_n"):
                row[key], row[key + "_sem"] = _ms([r[key] for r in g])
            rows.append(row)
        out["estimate"] = rows
        print("\n== estimation and prediction ==")
        print(f"{'sigma':>6} {'M':>5} {'n':>9} {'|dtheta|':>11} {'q rel.err':>10} "
              f"{'u approx':>11} {'R_n':>9}")
        for r in rows:
            print(f"{r['sigma']:>6} {r['M']:>5} {r['n_total']:>9} "
                  f"{r['theta_error']:>11.3e} {r['q_error']:>10.3f} {r['u_approx']:>11.3e} "
                  f"{r['R_n']:>9.3f}")
        out["sieve"] = _sieve_summary(est, args)

    if "inference" in parts:
        rows = []
        for (sigma, M), g in sorted(_group([r for r in records if r["part"] == "inference"],
                                           ("sigma", "M")).items()):
            truth = g[0]["theta_true"]
            rep = interval_report(
                estimates=[r["theta_hat"] for r in g],
                lowers=[r["lower"] for r in g], uppers=[r["upper"] for r in g],
                ses=[r["se"] for r in g], truth=truth, alpha=args.alpha,
            )
            piv = studentized([r["theta_hat"] for r in g], [r["se"] for r in g], truth)
            piv = piv[np.isfinite(piv)]
            theo, emp = qq_points(piv) if piv.size > 1 else (np.array([]), np.array([]))
            row = {"sigma": sigma, "M": M, "n_est": g[0]["n_est"],
                   "n_inf": g[0]["n_inf"], "repeats": len(g), **rep,
                   "sigma_hat": _ms([r["sigma_hat"] for r in g])[0],
                   "shrinkage": _ms([r["shrinkage"] for r in g])[0],
                   "qq_theoretical": theo.tolist(), "qq_empirical": emp.tolist()}
            row["mean_se_over_std"] = (
                row["mean_se"] / row["std"] if row["std"] else float("nan")
            )
            rows.append(row)
        out["inference"] = rows
        print("\n== inference: Algorithm-2 standard error vs the empirical spread ==")
        print(f"{'sigma':>6} {'M':>5} {'reps':>5} {'bias':>11} {'std':>11} "
              f"{'mean_se':>11} {'ratio':>7} {'len':>11} {'cov':>6}")
        for r in rows:
            print(f"{r['sigma']:>6} {r['M']:>5} {r['repeats']:>5} "
                  f"{r['bias']:>11.3e} {r['std']:>11.3e} {r['mean_se']:>11.3e} "
                  f"{r['mean_se_over_std']:>7.3f} {r['mean_length']:>11.3e} "
                  f"{r['coverage']:>6.2f}")

    if "ablation" in parts:
        rows = []
        for pi, g in sorted(_group([r for r in records if r["part"] == "ablation"],
                                   ("pi",)).items()):
            row = {"pi": pi[0], "repeats": len(g)}
            for key in ("theta_error", "q_error", "loss_swing", "loss_swing_rel",
                        "theta_hat"):
                row[key], row[key + "_sem"] = _ms([r[key] for r in g])
            rows.append(row)
        out["ablation"] = rows
        print("\n== input-design ablation (fixed total sample size) ==")
        print(f"{'pi':>6} {'|dtheta|':>11} {'+-':>9} {'q rel.err':>10} "
              f"{'loss swing':>11} {'rel':>8}")
        for r in rows:
            print(f"{r['pi']:>6} {r['theta_error']:>11.3e} {r['theta_error_sem']:>9.1e} "
                  f"{r['q_error']:>10.3f} {r['loss_swing']:>11.3e} "
                  f"{r['loss_swing_rel']:>8.2e}")

    if "calibration" in parts:
        rows = []
        for pi, g in sorted(_group([r for r in records if r["part"] == "calibration"],
                                   ("pi",)).items()):
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
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("part", choices=("estimate", "inference", "ablation", "calibration"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-x", type=int, default=32)
    p.add_argument("--n-t", type=int, default=25)
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[16, 64, 256],
                   help="training-curve counts for estimate/inference")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3])
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-n", type=float, default=None)
    p.add_argument(
        "--stopping-tau", type=float, default=None,
        help="tau of Eq. (16); the run stops at the first epoch with "
             "eta*|grad L|^2 <= tau.  The default disables the rule, which is "
             "what the paper's implementation details do in practice (a fixed "
             "epoch budget with the best-validation iterate returned).  Do not "
             "reach for a small constant: eta*|grad L|^2 is 6.0e-04 at the "
             "initial iterate here, so the old config default of 1e-3 fired at "
             "epoch 0 and returned an untrained model.  Scale tau to the "
             "problem -- the initial eta*|grad L|^2 is the natural unit.",
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
        "--data-step", type=float, default=2e-2,
        help="the *reference* solve's internal step.  It never enters the "
             "estimator, so it only has to be accurate, not small: with rk4, "
             "2e-2 already matches 1e-3 to 3.2e-09 on this case and is 20x "
             "cheaper.  Only lower it if you also pass --solve-method euler.",
    )
    p.add_argument(
        "--solve-method", choices=("rk4", "euler"), default="rk4",
        help="the reference solve's scheme.  'rk4' is the published setting; "
             "'euler' at the model's own --ode-step removes the discretisation "
             "mismatch between data and model, which is how its contribution to "
             "the theta bias is measured rather than assumed.",
    )
    p.add_argument(
        "--ode-step", type=float, default=2e-2,
        help="the *model's* rollout step.  Its error is a bias in theta_hat, "
             "so this -- not --data-step -- is the knob to spend integration "
             "time on.  With the rk4 default, 2e-2 is already converged.",
    )
    p.add_argument(
        "--ode-method", choices=("euler", "rk4"), default="rk4",
        help="the *model's* rollout scheme.  'rk4' costs 4 drift evaluations "
             "per step against Euler's 1 but is O(h^4) rather than O(h); at "
             "ode-step 2e-2 it matches the supplement's Euler-at-1e-3 "
             "accuracy (1008 evaluations per forward pass) for 192.  "
             "'euler' reproduces the legacy *_model.py setting, whose "
             "discretisation bias is what --ode-step was measuring.",
    )
    p.add_argument("--t-end", type=float, default=T_END,
                   help="observation window; past ~1.5 the reference solve shocks")
    p.add_argument("--delta", type=float, default=1e-4,
                   help="finite-difference step of Algorithm 2")
    p.add_argument("--inf-epochs", type=int, default=40)
    p.add_argument("--inf-lr", type=float, default=1e-2)
    p.add_argument("--no-linear", action="store_true",
                   help="disable the inner linear term")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
        summary = aggregate(records, args)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(summary, fh, indent=2, default=float)
            print(f"\nwrote {args.out}")
        return 0

    expected, done = _resume_state(args)
    jobs = enumerate_jobs(args)
    mine = [j for k, j in enumerate(jobs) if k % args.shards == args.shard]
    todo = [j for j in mine if job_key(j) not in done]

    print(f"config {json.dumps(expected, sort_keys=True)}", flush=True)

    print(f"shard {args.shard}/{args.shards}: part={args.part} "
          f"{len(todo)} of {len(mine)} jobs to run ({len(mine) - len(todo)} already done)",
          flush=True)

    fh = open(args.out, "a") if args.out else None
    try:
        for n, job in enumerate(todo, 1):
            t0 = time.time()
            try:
                rec = run_job(job, args)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{n}/{len(todo)}] FAILED {job}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            rec["key"] = job_key(job)
            rec["seconds"] = time.time() - t0
            print(f"  [{n}/{len(todo)}] {rec['key']}  "
                  f"{rec['seconds']:.0f}s  theta_error={rec.get('theta_error', float('nan')):.3e}",
                  flush=True)
            if fh is not None:
                fh.write(json.dumps(rec, default=float) + "\n")
                fh.flush()
    finally:
        if fh is not None:
            fh.close()
    print("shard done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
