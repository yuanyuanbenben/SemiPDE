"""An explicit covariance for ``theta_hat`` -- refits, never resampling.

Algorithm 2's ``Sigma_eff`` is a *plug-in* covariance: it reads the fitted
model's ``theta``-derivative ``M_hat`` and assumes the estimator responds to a
perturbation of the observations the way a linear (Gauss--Newton) estimator
would.  On Case 1 that assumption is measurably false -- ``docs/size.md``
Sec. 5a -- and the whole shortfall of ``se`` lives in the single number
``rms(M_hat)``: it is about ``2.3x`` larger than the value that would reproduce
the spread ``theta_hat`` actually has.  This module computes the response
directly instead of modelling it.

The identity
------------
Write ``theta_hat = F(Y)`` for the *whole* estimator -- generation, split,
optimisation, epoch selection -- and let ``a = dF/dY`` be its gradient with
respect to the observations.  Then to first order

    Var(theta_hat) = sigma^2 E[||a(Y)||^2],

which is exact for any estimator smooth enough to have ``a`` and makes no
reference to ``M_hat``, ``h_j``, ``Sigma_eff`` or the tangent space.  ``a`` is a
single fixed vector per data draw, and its norm comes back from random
Rademacher directions ``g`` through ``E[(a . g)^2] = ||a||^2``; each ``a . g``
is one central difference of two refits on the *same* design::

    d_r = (F(Y + t g_r) - F(Y - t g_r)) / (2 t)

Nothing here is resampled.  Every fit is an ordinary fit on an ordinary data
set; the perturbation is a numerical derivative of the estimator map, not a
pretend-sample, and the replicate spread is never used as a stand-in for
anything.

Why ``E`` before ``sqrt``
-------------------------
``||a||`` belongs to the *draw*, not to the estimator alone: the clean-base and
noisy-base ladders in ``docs/size.md`` are each stable in ``t``, yet differ by
``2.3x``, so one base point cannot stand in for the averaged quantity.
:func:`influence_covariance` therefore takes a *sequence* of base draws (noise
redrawn under a fixed design, i.e. group ``A`` of
:mod:`scripts.probe_variance`) and averages the squared norms, taking the root
once at the end.  Averaging ``sigma ||a||`` over draws instead would bias the
result down by Jensen.

The perturbation rides on every curve, and its first slice is left alone
(``g[0] = 0``) because the generator puts no noise there either -- perturbing a
point that carries no noise would measure the derivative of a map that does not
depend on it.

This module owns the *mathematics* of the probe -- the direction seeds, the
central differences, the draw averaging and the ``WaldIntervals`` mapping -- and
nothing else.  The fit plumbing stays with the caller, which passes ``refit`` as
a callable; ``scripts/probe_influence.py`` supplies a process-pool dispatcher,
and a single-fit caller can supply a plain closure.  That seam is deliberate:
the arithmetic above is the part that is easy to get wrong, and it must be
identical wherever it is used.

``refit`` takes a *batch* of trajectories and returns one estimate each, rather
than taking them one at a time.  That is not an interface preference: a
one-at-a-time callable would be invoked serially by the loop below, so a caller
whose parallelism lives in a process pool (the probe's does) would silently lose
all of it.  Handing over ``2 * directions + 1`` problems in one call is what lets
the caller submit them together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .data import Trajectories
from .inference import WaldIntervals

__all__ = ["InfluenceResult", "influence_covariance", "perturb"]


def perturb(
    base: Trajectories,
    *,
    step: float,
    seed: int,
    sign: float = 1.0,
) -> Trajectories:
    """``base`` shaken by ``sign * step * g``, with ``g`` Rademacher.

    The direction ``g`` is drawn from ``seed`` alone, so ``+`` and ``-`` share
    it and their difference isolates the odd part of the estimator's response.
    The initial slice is left at zero -- see the module docstring.
    """
    states = base.states.clone()
    mask = torch.ones(
        (states.shape[0], *states.shape[1:]), dtype=torch.bool
    )
    mask[:, 0] = False
    rng = np.random.default_rng(seed)
    for i in range(states.shape[0]):
        g = rng.choice(np.array([-1.0, 1.0]), size=tuple(states[i].shape))
        g[0] = 0.0
        states[i] += torch.as_tensor(sign * step * g, dtype=states.dtype)
    return Trajectories(
        states=states,
        clean=base.clean,
        z=base.z,
        t=base.t,
        initial=base.initial,
        sigma=base.sigma,
        noise_mask=mask,
    )


def direction_seeds(directions: int, seed: int = 1234) -> List[int]:
    """Seeds for the Rademacher directions, drawn once and reused per draw.

    Reusing them across base draws is common random numbers: it makes the
    per-draw values comparable and cuts the variance of their mean without
    biasing it.
    """
    if directions < 1:
        raise ValueError(f"directions must be >= 1, got {directions}")
    gen = np.random.default_rng(seed)
    return [int(gen.integers(1 << 30)) for _ in range(directions)]


@dataclass
class InfluenceResult(WaldIntervals):
    """``sigma^2 E[a a^T]`` rescaled so ``interval()`` reports the sd of ``theta_hat``.

    ``cov`` is ``n_est * sigma^2 * E[a a^T]``: :meth:`WaldIntervals.interval`
    divides by ``n_est``, so the half-width comes out as ``z * sigma * ||a||``,
    the estimator's own standard deviation, with no appeal to ``Sigma_eff``.
    """

    names: List[str]
    theta: np.ndarray                 # (p,) point estimate, first base draw
    cov: np.ndarray                   # (p, p) n_est * sigma^2 * E[a a^T]
    sigma: float
    n_est: int
    aa: np.ndarray                    # (p, p) E[a a^T], the raw second moment
    per_draw: List[Dict[str, object]] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def measured(self) -> np.ndarray:
        """``sigma * ||a_i||`` per coordinate: the estimator's own sd."""
        return self.sigma * np.sqrt(np.diag(self.aa))

    @property
    def ratio(self) -> float:
        """``sigma ||a|| / se`` on the first coordinate.

        The diagnostic the whole construction turns on: a value near ``1``
        acquits ``Sigma_eff``; a value near ``sd_A / se`` from
        :mod:`scripts.probe_variance` indicts ``M_hat``.
        """
        claimed = float(self.diagnostics.get("algorithm2_se", 0.0))
        if claimed <= 0.0:
            return float("nan")
        return float(self.measured[0] / claimed)


def influence_covariance(
    refit: Callable[[Sequence[Trajectories]], Sequence[np.ndarray]],
    bases: Sequence[Trajectories],
    *,
    names: Sequence[str],
    sigma: float,
    n_est: int,
    step: Optional[float] = None,
    directions: int = 16,
    seed: int = 1234,
    algorithm2_se: Optional[float] = None,
) -> InfluenceResult:
    """Explicit ``Var(theta_hat)`` from central differences of ``refit``.

    ``refit`` takes a batch of trajectories and returns one estimate per input,
    in order -- see the module docstring for why it is batched.  It must be
    *deterministic given the observations*: every other input -- split seed,
    initialisation, epoch budget, learning rate, batch order -- is the caller's
    responsibility to pin, or the central difference measures that randomness
    rather than ``a``.  ``bases`` are independent noise draws under one fixed
    design; the squared norms are averaged over them before the root is taken.

    ``step`` defaults to ``0.2 * sigma``.  ``algorithm2_se`` is recorded for the
    ``measured / claimed`` diagnostic and changes no number computed here.
    """
    bases = list(bases)
    if not bases:
        raise ValueError("at least one base draw is required")
    t = 0.2 * float(sigma) if step is None else float(step)
    if t <= 0.0:
        raise ValueError(f"step must be positive, got {t}")
    seeds = direction_seeds(directions, seed)

    per_draw: List[Dict[str, object]] = []
    for base in bases:
        # One batch: the base, then every ``+t g``, then every ``-t g``.  The
        # order is the caller's contract, so ``+`` and ``-`` for direction ``r``
        # share the Rademacher draw and their difference is exactly the odd part.
        batch = [base]
        batch += [perturb(base, step=t, seed=s, sign=+1.0) for s in seeds]
        batch += [perturb(base, step=t, seed=s, sign=-1.0) for s in seeds]
        fits = [np.asarray(x, dtype=np.float64).reshape(-1) for x in refit(batch)]
        if len(fits) != len(batch):
            raise ValueError(
                f"refit returned {len(fits)} estimates for {len(batch)} inputs"
            )
        base_theta = fits[0]
        plus = np.asarray(fits[1:1 + len(seeds)])              # (R, p)
        minus = np.asarray(fits[1 + len(seeds):])              # (R, p)
        d = (plus - minus) / (2.0 * t)                         # a . g_r
        c = (plus + minus - 2.0 * base_theta) / (2.0 * t * t)  # curvature
        # ``odd`` carries the derivative and ``even`` the only readable
        # curvature; the ratio is dimensionless and *does* depend on ``t`` (the
        # quadratic contamination of the derivative is O(t^2) against a linear
        # term), so it should fall roughly linearly in ``t``.  Comparing an
        # absolute theta-scale number against ``sigma |a| / sqrt(n)`` instead
        # cancels ``t`` exactly and is structurally incapable of testing the
        # step -- an error this probe has already made once.
        per_draw.append({
            "theta_hat": base_theta,
            "aa": d.T @ d / len(d),
            "odd": t * float(np.sqrt((d ** 2).mean())),
            "even": t * t * float(np.sqrt((c ** 2).mean())),
            "directions": d.tolist(),
            "curvatures": c.tolist(),
        })

    aa = np.mean([r["aa"] for r in per_draw], axis=0)
    cov = n_est * float(sigma) ** 2 * aa
    odd = float(per_draw[0]["odd"])
    even = float(per_draw[0]["even"])
    diagnostics: Dict[str, float] = {
        "step": t,
        "directions": float(directions),
        "base_draws": float(len(bases)),
        "even_over_odd": even / odd if odd > 0.0 else float("inf"),
    }
    if algorithm2_se is not None:
        diagnostics["algorithm2_se"] = float(algorithm2_se)
    return InfluenceResult(
        names=list(names),
        theta=np.asarray(per_draw[0]["theta_hat"], dtype=np.float64),
        cov=cov,
        sigma=float(sigma),
        n_est=int(n_est),
        aa=aa,
        per_draw=per_draw,
        diagnostics=diagnostics,
    )
