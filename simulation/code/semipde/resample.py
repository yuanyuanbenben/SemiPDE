"""Variance for ``theta_hat`` from the estimator's own replicates, not the score.

Algorithm 2 estimates the covariance by *plugging in an estimated efficient
score*: ``Sigma_hat_eff = sigma_hat^2 {mean_i M_hat_i M_hat_i^T}^{-1}``.  That is
a consistent estimator of the paper's ``Sigma_eff``, but only if ``h_hat`` is a
near-minimiser of the projection problem over the *right* class.  When it is
not, the failure is invisible from the outside: a ``M_hat`` that is too large
makes ``Sigma_hat_eff`` too small and the interval too narrow, and nothing in
the Algorithm-2 output says so.

The estimators here never touch ``M_hat``.  They re-run the *whole* profiling
M-estimator on resampled data and read the covariance off the spread of the
resulting ``theta_hat`` -- the classical bootstrap/subsampling route, which
assumes only that the curves are exchangeable.  Used next to
:func:`~semipde.inference.estimate_covariance` it is an audit: the two agree
when the inner fit really does reach its class optimum, and the gap between them
localises the failure when it does not.

Two resampling schemes, with a real trade-off:

``"bootstrap"``
    Draw ``M`` curves with replacement from the ``M`` estimation curves, refit,
    repeat.  Each replicate sees a full-size sample, so it estimates the
    variance *at the design sample size* with no extrapolation.  Cost is
    ``n_rep`` full fits.
``"partition"``
    Cut the estimation curves into ``n_groups`` disjoint blocks and fit each.
    Each replicate sees ``M / n_groups`` curves, i.e. ``M / n_groups`` times
    less data, and the resulting spread is rescaled by that factor.  Cost is
    ``n_groups`` fits on ``1 / n_groups`` of the data, so roughly one fit's
    worth of data but ``n_groups`` times the wall clock.  The catch is the
    classical ``m``-out-of-``n`` bias: a fit on a quarter of the curves is not
    a quarter-sample version of the same estimator, because the penalty
    ``lambda_n`` and the sieve width were tuned for the full sample, so the
    rescaled spread is biased and the direction is not fixed in advance.

Both return a covariance of ``sqrt(n) (theta_hat - theta_0)``, the same object
:func:`~semipde.inference.estimate_covariance` returns, so
:meth:`ResampleResult.interval` and ``InferenceResult.interval`` are directly
comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .config import CaseConfig
from .data import Trajectories
from .inference import WaldIntervals, parameter_names
from .model import ProfilingModel
from .train import train

__all__ = ["ResampleResult", "curve_resample"]


@dataclass
class ResampleResult(WaldIntervals):
    """A covariance read off repeated fits, plus the replicates behind it."""

    names: List[str]
    theta: np.ndarray                 # (p,) estimate on the whole sample
    cov: np.ndarray                   # (p, p) covariance of sqrt(n) (theta - theta_0)
    n_est: int                        # sample size each replicate ran at
    estimates: np.ndarray             # (n_rep, p) the replicates themselves
    mode: str
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.estimates.ndim != 2 or self.estimates.shape[0] < 2:
            raise ValueError(
                "a covariance needs at least two replicates, got "
                f"{self.estimates.shape}"
            )


def _replicate_theta(
    spec,
    cfg: CaseConfig,
    train_data: Trajectories,
    val_data: Trajectories,
    *,
    seed: int,
    epochs: int,
    lr: float,
    lambda_n: Optional[float],
    stopping_tau: Optional[float],
    batch_size: Optional[int],
    select: str,
    device,
) -> np.ndarray:
    """One refit of the profiling estimator, returned as a flat vector."""
    model = ProfilingModel(spec, cfg, seed=seed, device=device)
    train(
        model,
        train_data,
        val_data,
        config=cfg,
        epochs=epochs,
        lr=lr,
        lambda_n=lambda_n,
        stopping_tau=stopping_tau,
        batch_size=batch_size,
        seed=seed,
        select=select,
    )
    with torch.no_grad():
        flat = [p.detach().reshape(-1).cpu().numpy() for _, p in model.theta.items()]
    return np.concatenate(flat).astype(np.float64)


def curve_resample(
    spec,
    cfg: CaseConfig,
    curves: Trajectories,
    *,
    val: Optional[Trajectories] = None,
    n_rep: int = 32,
    mode: str = "bootstrap",
    n_groups: int = 8,
    epochs: int = 200,
    lr: float = 1e-3,
    lambda_n: Optional[float] = None,
    stopping_tau: Optional[float] = None,
    batch_size: Optional[int] = None,
    select: str = "best",
    vary_init: bool = True,
    seed: int = 0,
    point: Optional[np.ndarray] = None,
    device=None,
    fitter: Optional[Callable[..., np.ndarray]] = None,
    verbose: bool = False,
) -> ResampleResult:
    """Covariance of ``sqrt(n)(theta_hat - theta_0)`` by repeated refitting.

    Parameters
    ----------
    curves
        The **estimation** sample -- the same curves
        :func:`~semipde.inference.estimate_covariance` would be run on.  It is
        resampled, never the inference sample: this estimator is meant to be
        comparable with Algorithm 2's, and the two only are if they see the same
        data.
    val
        Validation curves, held fixed across replicates.  Varying them would put
        the validation noise into the spread, where it does not belong.
    mode
        ``"bootstrap"`` or ``"partition"``; see the module docstring.
    vary_init
        ``True`` (default) gives each replicate its own initialisation, which is
        what the estimator is: a fit from a random start is a randomised
        estimator, so its initialisation is part of its variance.  ``False``
        pins the start, isolating the part of the spread that comes from the
        data.  Reporting the first and checking the second is the cheap way to
        see whether the bootstrap is measuring sampling noise or optimiser
        noise.
    point
        The full-sample estimate, if the caller already has one.  Otherwise it
        is fit here, on ``curves`` with the same settings.
    fitter
        Overrides the refit entirely: ``fitter(train_curves, seed) -> theta``.
        It exists so the resampling logic can be tested without paying for a
        PDE solve per replicate.

    Returns
    -------
    A :class:`ResampleResult` whose ``cov`` is on the same scale as
    ``Sigma_eff``, i.e. ``cov`` is the asymptotic covariance, not the variance
    of ``theta_hat`` itself.  Divide by ``n_est`` for the latter.
    """
    if mode not in ("bootstrap", "partition"):
        raise ValueError(f"mode must be 'bootstrap' or 'partition', got {mode!r}")
    if mode == "bootstrap" and n_rep < 2:
        raise ValueError("bootstrap needs n_rep >= 2")
    if mode == "partition" and n_groups < 2:
        raise ValueError("partition needs n_groups >= 2")

    n_curves = len(curves)
    if n_curves < 2:
        raise ValueError(f"need at least two curves, got {n_curves}")

    def run(train_data: Trajectories, seed_b: int) -> np.ndarray:
        """One replicate: the caller's fitter, or a full Algorithm-1 refit."""
        if fitter is not None:
            return np.asarray(fitter(train_data, seed_b), dtype=np.float64)
        return _replicate_theta(
            spec, cfg, train_data, val,
            seed=seed_b, epochs=epochs, lr=lr, lambda_n=lambda_n,
            stopping_tau=stopping_tau, batch_size=batch_size, select=select,
            device=device,
        )

    def init_seed(b: int) -> int:
        # ``vary_init`` decides whether the replicate's initialisation is part
        # of the spread.  It is part of the estimator, so it is on by default;
        # pinning it isolates the data contribution.
        return seed + 1 + b if vary_init else seed

    rng = np.random.default_rng(seed)
    estimates: List[np.ndarray] = []

    if mode == "bootstrap":
        for b in range(n_rep):
            idx = rng.integers(0, n_curves, size=n_curves)
            train_b = curves.subset(torch.as_tensor(idx, dtype=torch.long))
            estimates.append(run(train_b, init_seed(b)))
            if verbose:
                print(f"  bootstrap {b + 1}/{n_rep}: theta={estimates[-1]}",
                      flush=True)
        per_rep = float(n_curves)
    else:
        order = rng.permutation(n_curves)
        blocks = np.array_split(order, n_groups)
        if any(len(b) < 2 for b in blocks):
            raise ValueError(
                f"n_groups={n_groups} leaves a block of fewer than 2 curves of "
                f"{n_curves}; the block's own variance would be undefined"
            )
        for g, block in enumerate(blocks):
            train_g = curves.subset(torch.as_tensor(block, dtype=torch.long))
            estimates.append(run(train_g, init_seed(g)))
            if verbose:
                print(f"  group {g + 1}/{n_groups} ({len(block)} curves): "
                      f"theta={estimates[-1]}", flush=True)
        per_rep = n_curves / n_groups

    est = np.stack(estimates, axis=0)
    if est.shape[1] < 1:  # pragma: no cover - defensive
        raise ValueError("the estimator returned no parameters")
    # ``ddof=1``: these are replicate draws of a random quantity, not a
    # population.  One replicate would make this a division by zero, which is
    # why ``n_rep >= 2`` is enforced above.
    spread = np.cov(est, rowvar=False, ddof=1).reshape(est.shape[1], est.shape[1])
    n_rep_sample = int(round(per_rep * cfg.n_traj))
    cov = spread * n_rep_sample

    if point is None:
        point = run(curves, seed)

    names = parameter_names(ProfilingModel(spec, cfg, device=device))

    return ResampleResult(
        names=names,
        theta=np.asarray(point, dtype=np.float64),
        cov=cov,
        n_est=n_rep_sample,
        estimates=est,
        mode=mode,
        diagnostics={
            "per_replicate_n": float(n_rep_sample),
            "curves_per_replicate": float(per_rep),
            "n_replicates": float(est.shape[0]),
            "vary_init": float(bool(vary_init)),
            "bootstrap_mean": float(est.mean()),
            "bootstrap_sd_of_mean": float(
                math.sqrt(spread[0, 0] / est.shape[0]) if spread.size else 0.0
            ),
        },
    )
