#!/usr/bin/env python
"""Every simulation case of the paper, on the shared runner.

``semipde/study.py`` holds the machinery -- sharding, resume, provenance, the
tables -- and this module is the wiring: which case each study runs, what its
configuration is, and the extra parts and columns that only some cases have.

Usage::

    # the forced case, 42 CPU workers, then the tables
    scripts/run_study.sh 42 inference --case burg --repeats 50 --epochs 200

    # the size study: does R_n = sqrt(J(F)) grow with n?
    scripts/run_study.sh 32 estimate --case rd \\
        --sample-sizes 16 64 256 --repeats 5 --epochs 200

    # by hand, one shard
    python scripts/exp_study.py estimate --case advz --shard 0 --shards 16 \\
        --out runs/advz/estimate.jsonl

``--case`` selects the study and defaults to ``burg``.  The five others are
``rd``, ``fhn``, ``np``, ``ns`` and ``advz``; ``cases.py`` is the single source
of truth for their settings and placement in the paper.

Two columns are worth naming, because the old per-case drivers called them
different things:

``mech_error``
    ``||f_hat - f_0|| / ||f_0||`` at the observed points, computed the same way
    for every case.  On the forced case ``f_0 = q(x)`` does not depend on time
    and the observation grid is uniform, so this is *identical* to the
    ``q_error`` the earlier forced runs recorded -- the rows stay comparable.
``R_n``
    ``sqrt(J(F)) = ||phi - phi_0||_2``, the supplement's radius.  It is what
    ``lambda_n`` penalises and what the width condition constrains, so it is
    recorded on every job rather than only in the runs that were about size.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semipde import cases  # noqa: E402
from semipde.config import CaseConfig  # noqa: E402
from semipde.data import generate  # noqa: E402
from semipde.inference import estimate_covariance  # noqa: E402
from semipde.model import ProfilingModel  # noqa: E402
from semipde.study import (  # noqa: E402
    Study,
    fit,
    forced_tables,
    job_setup,
    main as run_study,
    score,
    with_overrides,
)

#: ``case key -> (n_grid, n_t)``: the sizes the paper actually runs, which are
#: not the same for its two experiments.  For the multi-dimensional cases
#: ``n_grid`` is the count *per axis*, as the case factories build isotropic
#: grids.  ``--n-x`` and ``--n-t`` override either entry.
#:
#: ``supp.tex:523`` (SemiPDE against the baselines, ``sigma in {0.1, 0.5}``,
#: 50 repetitions) fixes ``n_t = 25`` and two sizes per case, of which the
#: smaller is taken here.  ``supp.tex:525`` (the size study, ``sigma = 0.1``,
#: 200 repetitions since 2026-10-14 and 500 before that) fixes ``n_t = 10`` and
#: ``n_x = 16`` for Case 1, ``n_y = 8``
#: for Case 2 -- ``n = 160`` and ``640``, the sizes ``main.tex:887`` states.
#: The two ``Z`` cases have no published size, so they take Case 1's: the
#: point of adding them is to be read alongside it.
#:
#: These are *labels* for the design, not residual counts.  The ``n`` that
#: enters an interval is ``n_total = M n_traj`` and is carried per record --
#: see ``config.py`` and ``docs/divergences.md`` Sec. 11 for why the two differ
#: and why Case 2's field count is a live question.
PAPER_SIZES: Dict[str, Dict[str, tuple]] = {
    "estimate": {
        "rd": (32, 25), "fhn": (8, 25), "np": (32, 25),
        "ns": (8, 25), "burg": (32, 25), "advz": (32, 25),
    },
    "inference": {
        "rd": (16, 10), "fhn": (8, 10), "np": (16, 10),
        "ns": (8, 10), "burg": (16, 10), "advz": (16, 10),
    },
    # The ablation and the calibration are Case-1/Example-2 scale probes, not
    # tables of their own, so they follow the estimate settings.
    "ablation": {"rd": (32, 25), "fhn": (8, 25), "np": (32, 25),
                 "ns": (8, 25), "burg": (32, 25), "advz": (32, 25)},
    "calibration": {"rd": (32, 25), "fhn": (8, 25), "np": (32, 25),
                    "ns": (8, 25), "burg": (32, 25), "advz": (32, 25)},
}

#: The cases whose law carries the auxiliary variable ``Z``.
HAS_INPUT = ("burg", "advz")

#: Cases whose factory takes ``inner_inputs`` (Algorithm 2's inner class).
#: Every case now does: ``np``/``ns`` gained the argument so that the two
#: coordinate-reading supplement cases can be sized by the same rule as the
#: four main ones.
INNER_INPUT_CASES = ("rd", "fhn", "burg", "advz", "np", "ns")

DEFAULT_CASE = "burg"



# ---------------------------------------------------------------------------
# the case layer
# ---------------------------------------------------------------------------
def _case(key: str, args, *, part: str = "estimate", sigma: float = 0.1,
          M: int = 64, pi: float = 0.5) -> cases.Case:
    """The ``Case`` a job runs, assembled by its factory in ``cases.py``.

    Everything the command line can vary is routed through the factory's own
    named arguments, so a setting the case prescribes is never silently lost:
    ``data_step``, ``ode_step`` and the network are applied afterwards by
    :func:`~semipde.study.with_overrides`, which is also where their ``None``
    default means "what the case prescribes".
    """
    # ``make_config`` calls this without a part, so read the real one off
    # ``args``; the sizes differ between the two experiments.
    part = getattr(args, "part", part)
    grid_default, n_t_default = PAPER_SIZES.get(
        part, PAPER_SIZES["estimate"])[key]
    overrides: Dict = {
        "n_grid": args.n_x if args.n_x is not None else grid_default,
        "n_t": args.n_t if args.n_t is not None else n_t_default,
        "sigma": sigma,
        "M": M,
    }
    if key in HAS_INPUT:
        overrides["pi"] = pi
    if key == "rd" and getattr(args, "legacy_ic", False):
        # The published Case 1 tables were produced with the legacy profile, so
        # a run meant to be compared against them has to ask for it.
        overrides["legacy_ic"] = True
    # Both are ``None``/default on a normal run; they exist so a size study can
    # widen ``f_hat``'s inputs and Algorithm 2's inner class independently.
    # Only the factories that take them are listed: anything else would land in
    # the case's ``**extra`` and then in ``CaseConfig``, which is a TypeError in
    # a place that says nothing about the real mistake.
    if getattr(args, "inner_inputs", None) and key in INNER_INPUT_CASES:
        overrides["inner_inputs"] = args.inner_inputs
    if getattr(args, "mechanism_inputs", None) and key == "rd":
        overrides["mechanism_inputs"] = args.mechanism_inputs
    if key == "burg" and part == "calibration":
        # The supplement's analytic I_eff is exact only at the zero background:
        # (theta_0, q_0) = (0, 0) with a common design.  Anywhere else the PDE
        # is nonlinear in theta and the formula is simply not a statement about
        # the fitted covariance.
        overrides.update(theta_true=0.0, q_modes=())
    if getattr(args, "theta_true", None) is not None:
        # The factory's own keyword is used so that the generator, the
        # estimator and the scoring all read the same truth.  Case 1 now
        # defaults to 1e-2, so the flag's uses are putting a run back on the
        # paper's 3e-3 or on the 3e-2 this case carried on 2026-10-22/23.  Only
        # Case 1 is wired; the other factories parameterise theta differently
        # (a tuple, or no scalar at all).
        if key != "rd":
            raise SystemExit("--theta-true is only wired for --case rd")
        overrides["theta_true"] = float(args.theta_true)
    return cases.build(key, **overrides)


def make_spec(key: str):
    def _make(args, part: str):
        return _case(key, args, part=part).spec

    return _make


def make_config(key: str):
    def _make(args, *, spec, sigma: float, M: int, pi: float = 0.5,
              width: int | None = None) -> CaseConfig:
        # ``spec`` is unused: the case factory builds the grid and the spec
        # together, so re-deriving the grid from ``args`` is what keeps the two
        # consistent.  The parameter is kept because every study's hook is
        # called with it.
        case = _case(key, args, sigma=sigma, M=M, pi=pi)
        return with_overrides(case.config, args, width=width)

    return _make


def plain_study(key: str, title: str, **kwargs) -> Study:
    """A study for a case with no input-carrying parts of its own."""
    return Study(key=key, title=title, make_spec=make_spec(key),
                 make_config=make_config(key),
                 metrics=("theta_error", "mech_error", "u_approx", "R_n"),
                 **kwargs)


# ---------------------------------------------------------------------------
# the forced case: the design ablation and the zero-background calibration
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


def _ablation_job(key: str):
    def _job(job: Mapping, args) -> Dict:
        rep, pi = job["rep"], job["pi"]
        spec, cfg = job_setup(STUDIES[key], job, args)
        model, split, _ = fit(
            spec, cfg, seed=args.seed + rep, epochs=args.epochs, lr=args.lr,
            lambda_n=args.lambda_n, stopping_tau=args.stopping_tau,
            device=args.device, solve_method=args.solve_method,
        )
        sc = score_run(key, model, split, spec)
        cs = np.linspace(-args.c_max, args.c_max, args.n_c)
        losses, n_diverged = _loss_along_equivalence_direction(
            model, spec, cfg, split.test, cs
        )
        finite = np.asarray(losses, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        mid = float(np.min(finite)) if finite.size else float("nan")
        spread = float(np.ptp(finite)) if finite.size else float("nan")
        return {
            "case": key, "part": "ablation", **job,
            "n_total": args.M_ablation * cfg.n_traj,
            "theta_hat": sc["theta_hat"],
            "theta_error": sc["theta_error"],
            # The swing of the fitted loss along the direction that is exactly
            # flat at a single input.  Zero is the non-identifiability; the
            # scale-free ratio makes it comparable across runs.
            "loss_swing": spread,
            "loss_swing_rel": spread / mid if mid and mid > 0 else float("nan"),
            "n_diverged": n_diverged,
            "losses": [float(v) for v in losses],
        }

    return _job


def _calibration_job(key: str):
    def _job(job: Mapping, args) -> Dict:
        rep, pi = job["rep"], job["pi"]
        spec, cfg = job_setup(STUDIES[key], job, args)
        model, _, _ = fit(
            spec, cfg, seed=args.seed + rep, epochs=args.epochs, lr=args.lr,
            lambda_n=args.lambda_n, stopping_tau=args.stopping_tau,
            device=args.device, solve_method=args.solve_method,
        )
        inf = generate(spec, cfg, n_curves=args.M_cal,
                       seed=20_000 + args.seed + rep, device=args.device)
        res = estimate_covariance(
            model, inf, delta=args.delta, epochs=args.inf_epochs,
            lr=args.inf_lr, n_est=cfg.M * cfg.n_traj,
            linear_term=not args.no_linear,
        )
        kg2 = _expected_kg_squared(model, spec, inf, args.delta)
        info = _efficiency_information(pi, kg2, cfg.sigma)
        var_theory = 1.0 / info if info > 0 else float("inf")
        return {
            "case": key, "part": "calibration", **job,
            "sigma_hat": float(res.cov[0, 0]),
            "analytic_var": var_theory,
            "ratio": (float(res.cov[0, 0]) / var_theory
                      if var_theory > 0 else float("nan")),
            "shrinkage": float(res.diagnostics["mismatch_shrinkage_0"]),
        }

    return _job


def score_run(key: str, model, split, spec) -> Dict[str, float]:
    """``semipde.study.score`` for a study looked up by key."""
    return score(STUDIES[key], model, split, spec)


def forced_study(key: str, title: str) -> Study:
    return Study(
        key=key,
        title=title,
        make_spec=make_spec(key),
        make_config=make_config(key),
        # The two forced parts come in through ``extra_parts``, which is what
        # makes them appear in the parser's choices as well.
        extra_parts={
            "ablation": _ablation_job(key),
            "calibration": _calibration_job(key),
        },
        extra_summary=forced_tables,
        metrics=("theta_error", "mech_error", "u_approx", "R_n"),
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
STUDIES: Dict[str, Study] = {
    "rd": plain_study("rd", "Case 1, Fisher--KPP reaction--diffusion"),
    "fhn": plain_study("fhn", "Case 2, FitzHugh--Nagumo"),
    "np": plain_study("np", "Case 3, Nernst--Planck (supplement)"),
    "ns": plain_study("ns", "Case 4, 3-D Navier--Stokes (supplement)"),
    "advz": plain_study("advz", "the Z-switched advection--diffusion case"),
    "burg": forced_study("burg", "Example 2, forced Burgers with input Z"),
}

CASE_HELP = "\n".join(f"  {k:<6} {v.title}" for k, v in sorted(STUDIES.items()))


def _select_case(argv: Sequence[str]) -> Tuple[str, List[str]]:
    """Pop ``--case`` out of ``argv``, defaulting to :data:`DEFAULT_CASE`.

    ``--case`` cannot be a normal option: which study it is decides the parser
    -- its parts, and the help text -- and the parser is what would parse it.
    """
    rest: List[str] = []
    key = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--case":
            if i + 1 >= len(argv):
                raise SystemExit("--case needs a value; known cases:\n" + CASE_HELP)
            key = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--case="):
            key = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1
    return (key or DEFAULT_CASE), rest


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    key, argv = _select_case(argv)
    if key not in STUDIES:
        raise SystemExit(f"unknown case {key!r}; known cases:\n" + CASE_HELP)
    study = STUDIES[key]
    description = (
        f"{study.title}\n\n"
        f"Everything is reported on the shared runner in ``semipde.study``; see\n"
        f"``semipde/cases.py`` for this case's settings and the tables of the\n"
        f"paper, and ``--help`` for the options.\n"
    )
    return run_study(study, argv, description=description)


if __name__ == "__main__":
    raise SystemExit(main())
