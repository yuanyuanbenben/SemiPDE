"""Algorithm 1: gradient flow on the penalised profiling objective.

The objective is Eq. (3) of the paper,

.. math::

    L_n(\\theta, F) + \\lambda_n J(F),
    \\qquad
    L_n = \\frac{1}{n}\\sum_i \\|Y_i - \\tilde u(X_i; \\theta, F)\\|^2,

minimised from ``(theta_init, phi_0)`` -- the true parameter and the reference
network -- along the gradient flow

.. math::

    \\dot{\\hat w}_t = -\\eta \\nabla L_n(\\hat w_t)

with the stopping rule of Eq. (16),

.. math::

    T = \\inf\\{t \\ge 0 : \\eta \\|\\nabla L_n(\\hat w_t)\\|_2^2 \\le \\tau\\}.

The reported output is the iterate at ``T``, except that the paper's
implementation details select the epoch with the best validation loss; both
criteria are recorded so the two can be compared (they are not the same rule,
and the legacy code only ever implemented the second one).

Two divergences from the legacy scripts are deliberate and are worth stating,
because both change the estimator rather than just its speed:

* the legacy ``_forward`` has the penalty ``lambda_n * J(F)`` **commented out**,
  so ``lambda_n`` never entered the fit;
* the legacy loop has no gradient-norm stopping rule at all, only epoch
  checkpointing;
* the legacy internal Euler step is ``0.02``, while the paper's implementation
  details state ``0.001``.

This module implements the paper's version.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .config import CaseConfig
from .data import Trajectories
from .model import ProfilingModel

__all__ = [
    "TrainResult",
    "objective",
    "mse",
    "evaluate",
    "train",
    "select_lambda",
    "lambda_grid",
]


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def _residuals(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pred - target


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over every observed scalar."""
    return _residuals(pred, target).pow(2).mean()


def objective(
    model: ProfilingModel,
    batch: Trajectories,
    lambda_n: float,
) -> torch.Tensor:
    """``L_n(theta, F) + lambda_n * J(F)`` on one batch of trajectories."""
    pred = model(batch.initial, batch.t, batch.z)
    loss = mse(pred, batch.states)
    if lambda_n:
        loss = loss + lambda_n * model.penalty()
    return loss


@torch.no_grad()
def evaluate(
    model: ProfilingModel,
    batch: Trajectories,
    lambda_n: float = 0.0,
) -> Dict[str, float]:
    """Losses on a batch, without touching the graph."""
    pred = model(batch.initial, batch.t, batch.z)
    out = {
        "mse": float(mse(pred, batch.states)),
        "penalty": float(model.penalty()),
    }
    if lambda_n:
        out["loss"] = out["mse"] + lambda_n * out["penalty"]
    else:
        out["loss"] = out["mse"]
    return out


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Outcome of one fit.

    ``history`` has one entry per validation check, with the training loss, the
    validation loss and the squared gradient norm the stopping rule reads.  The
    model itself is left holding the *best* iterate (by validation loss);
    ``theta`` and ``state_dict`` describe that iterate.
    """

    theta: Dict[str, float]
    best_epoch: int
    best_val: float
    stopped_epoch: Optional[int] = None
    stopped_by: Optional[str] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    n_rollouts: int = 0
    #: Which epoch ``theta`` and ``penalty`` were read from.  It equals
    #: ``best_epoch`` under the default ``select="best"`` and the last checked
    #: epoch under ``select="final"``; ``best_epoch`` keeps describing the
    #: validation curve either way.  See the ``select`` parameter of
    #: :func:`train` for why the two are worth telling apart.
    selected_epoch: int = -1
    #: ``theta`` at the *last* epoch, alongside :attr:`theta` at the selected
    #: one.  Recording both is what makes the cost of the validation rule
    #: measurable for free: the two are read off a single fit, so their
    #: difference is paired and carries no fit-to-fit noise.  ``None`` when the
    #: run was already selecting the final iterate.
    final_theta: Optional[Dict[str, float]] = None
    final_penalty: float = 0.0
    #: ``J(F)`` at the selected iterate.  Its square root is the supplement's
    #: weight radius ``R_n`` -- the quantity the sieve conditions bound -- so it
    #: is reported next to ``theta`` rather than left in ``history``, where it
    #: would be indexed by the wrong epoch once ``check_every > 1``.
    penalty: float = 0.0

    @property
    def radius(self) -> float:
        """``R_n = ||phi - phi_0||_2``, the radius of the fitted sieve element."""
        return math.sqrt(max(self.penalty, 0.0))

    @property
    def converged(self) -> bool:
        return self.stopped_by == "gradient"

    @property
    def fitted(self) -> bool:
        """``False`` when no epoch beat the initial iterate.

        The selected iterate is the best by validation loss, and the initial
        state is the fallback, so a run in which *every* epoch lost to the
        initial one returns an **untrained** model with no error -- silently, if
        the caller only reads ``theta``.  In practice this means the optimiser
        diverged (a step too large for the grid, or a NaN), which is worth
        finding out about rather than absorbing.
        """
        return self.best_epoch >= 0


def _batch_iter(trajectories: Trajectories, batch_size: Optional[int], shuffle: bool,
                generator: Optional[torch.Generator]):
    """Yield (index-tensor) batches; a single full batch when size is None."""
    n = len(trajectories)
    if batch_size is None or batch_size >= n:
        yield torch.arange(n)
        return
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def train(
    model: ProfilingModel,
    train_data: Trajectories,
    val_data: Trajectories,
    *,
    config: Optional[CaseConfig] = None,
    lambda_n: Optional[float] = None,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    inner_steps: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    stopping_tau: Optional[float] = None,
    check_every: int = 1,
    patience: Optional[int] = None,
    theta_lr: Optional[float] = None,
    theta_warmup: int = 0,
    grad_clip: Optional[float] = None,
    lr_final: Optional[float] = None,
    shuffle: bool = True,
    seed: int = 0,
    select: str = "best",
    verbose: bool = False,
    clock: Optional[Callable[[], float]] = None,
    max_seconds: Optional[float] = None,
) -> TrainResult:
    """Run Algorithm 1 and return the selected iterate.

    Parameters
    ----------
    model
        A :class:`~semipde.model.ProfilingModel`.  It is modified in place: on
        return it holds the selected iterate.
    lambda_n
        The penalty weight.  ``None`` means "read it from ``config``".
    batch_size
        ``None`` uses the full training set, which is what the theory's
        ``L_n`` is; a positive value minibatches, as the legacy scripts did.
    steps
        Number of optimizer steps.  ``None`` (the default) keeps the epoch
        semantics: one sweep over every minibatch, so a run with ``batch_size``
        b costs ``epochs * ceil(M / b)`` steps.  A positive value makes each
        outer iteration a *single* step on a freshly drawn minibatch of
        ``batch_size`` trajectories, so the run costs exactly ``steps`` steps --
        the knob the large-grid cases need, where one full sweep is thousands of
        times the work a gradient estimate requires.  ``steps`` replaces
        ``epochs`` as the length of the learning-rate schedule; validation and
        checkpointing still run every ``check_every`` outer iterations, so the
        selected iterate is the argmin over ``steps`` checks.
    stopping_tau
        The threshold ``tau`` of Eq. (16).  ``None`` means "read it from
        ``config``", whose default is ``None`` as well -- so the rule is off
        unless asked for, and validation checkpointing alone selects the
        iterate.  Pass ``math.inf`` to keep the rule recorded but never firing.
        See :attr:`~semipde.config.CaseConfig.stopping_tau` for why a constant
        ``tau`` is not transferable between cases.
    patience
        Stop early after this many consecutive validation checks without
        improvement.  ``None`` never stops early.
    select
        Which iterate to return: ``"best"`` keeps the argmin of the validation
        loss, ``"final"`` keeps the last one.  ``"best"`` is the paper's
        protocol ("the epoch with the minimum test loss", ``supp.tex:523``) and
        the default, but it is a *selection*, and the selection is invisible to
        the interval.  Algorithm 2 linearises the estimator's dependence on the
        **data**; it has nothing to say about a stopping rule that ranks a
        thousand noisy validation losses and keeps the winner, so whatever
        spread the ranking adds to ``theta_hat`` appears as under-coverage that
        no amount of Algorithm-2 tuning removes.  ``"final"`` exists to measure
        that: see ``scripts/probe_variance.py``.
    theta_lr
        A learning rate for ``theta`` alone.  ``None`` (the default) puts
        ``theta`` and the mechanism in one Adam group at ``lr``, which is the
        paper's setup and preserves every existing run bit for bit.  It is
        worth separating because the two live on wildly different scales: Adam's
        step is ``~lr`` in *absolute* terms, and ``theta`` is ``1e-3``, so a
        single shared ``lr = 1e-3`` moves it by 100% of its own size per
        iteration.  A ``theta`` that never settles -- or that freezes on the
        wrong value while the loss keeps falling along the mechanism -- is what
        that looks like, and it is invisible in the loss whenever the diffusion
        signal is absorbable by a smooth function of the state.
    theta_warmup
        Hold ``theta`` at its initial value for this many epochs, by setting its
        group's learning rate to zero.  Fitting the mechanism first is the other
        half of the same story: with a centred ``f`` (``f = 0`` at
        initialisation) and ``theta = 0``, the drift starts empty, so an early
        ``theta`` feeds on nothing but the misfit and can grab the truth's
        place before ``f`` is expressive enough to take it over.
    grad_clip
        Clip the global gradient norm to this value before every step.
        ``None`` (the default) does not clip, which is what the paper and the
        legacy scripts do.  A trajectory loss is a *rollout* loss: one stiff
        iterate sends the integrated state -- and with it the loss and the
        gradient -- far away within a single step, and the update that follows
        pushes the mechanism further into the same region.  A run that
        descends, bottoms out on the validation loss and then climbs for the
        remaining epochs is that loop, and clipping is what breaks it.
    lr_final
        Decay every group's learning rate from its base value to this one along
        a cosine over the run (the schedule is applied as a common factor, so a
        group's own base is preserved).  ``None`` keeps the constant rate the
        paper and the legacy scripts use.  Adam's step is ``~lr`` *per
        coordinate* no matter how large the gradient is, so with a constant
        ``lr = 1e-3`` and ``theta`` of size ``1e-3`` the estimate takes a random
        walk of a full parameter-length per iteration whose drift toward the
        truth is a small bias on top; decaying the rate is what lets that bias
        win.  Measured on FitzHugh--Nagumo, one epoch (five inner steps) is
        already enough to put ``theta_1`` at ``4.4e-3`` against a truth of
        ``1e-3`` -- four times over, before the mechanism has learned anything.
    clock, max_seconds
        Optional wall-clock guard; useful for the large repeated experiments.
    """
    config = model.config if config is None else config
    lambda_n = config.lambda_n if lambda_n is None else float(lambda_n)
    epochs = config.epochs if epochs is None else int(epochs)
    # ``steps`` reinterprets the outer loop: one optimizer step per iteration on
    # a random minibatch, instead of a sweep over every minibatch.  The length of
    # the outer loop is what the learning-rate schedule and the checkpoint rule
    # count, so it is resolved once here and used everywhere below.
    n_outer = epochs if steps is None else int(steps)
    if n_outer < 0:
        raise ValueError(f"steps must be non-negative, got {steps!r}")
    inner_steps = config.inner_steps if inner_steps is None else int(inner_steps)
    lr = config.lr if lr is None else float(lr)
    stopping_tau = config.stopping_tau if stopping_tau is None else float(stopping_tau)
    if select not in ("best", "final"):
        raise ValueError(f"select must be 'best' or 'final', got {select!r}")
    theta_warmup = max(0, int(theta_warmup))
    if theta_lr is not None:
        theta_lr = float(theta_lr)
    if grad_clip is not None:
        grad_clip = float(grad_clip)
        if grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {grad_clip!r}")
    lr_final = None if lr_final is None else float(lr_final)

    params = [p for p in model.parameters() if p.requires_grad]
    theta_group: Optional[Dict[str, object]] = None
    if theta_lr is None and theta_warmup == 0:
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        # The mechanism keeps ``lr``; ``theta`` gets its own, and its warmup is
        # a learning rate of zero rather than ``requires_grad_(False)`` so the
        # recorded ``grad_sq`` keeps reporting the honest joint gradient.
        theta_ids = {id(p) for p in model.theta.parameters()}
        rest = [p for p in params if id(p) not in theta_ids]
        theta_group = {
            "params": [p for p in params if id(p) in theta_ids],
            "lr": lr if theta_lr is None else theta_lr,
        }
        optimizer = torch.optim.Adam([{"params": rest, "lr": lr}, theta_group], lr=lr)
    # One common factor over every group, so a group's base rate stays a
    # property of the group rather than of the schedule.
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    def _lr_factor(epoch: int) -> float:
        if lr_final is None or n_outer < 2 or lr == 0.0:
            return 1.0
        t = epoch / (n_outer - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        return (lr_final + (lr - lr_final) * cosine) / lr

    generator = torch.Generator().manual_seed(int(seed) + 7919)

    history: List[Dict[str, float]] = []
    best_val = math.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    best_theta = model.theta_detached()
    best_penalty = float(model.penalty())
    n_rollouts = 0
    stale = 0
    stopped_epoch: Optional[int] = None
    stopped_by: Optional[str] = None

    for epoch in range(n_outer):
        factor = _lr_factor(epoch)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        if theta_group is not None and epoch < theta_warmup:
            theta_group["lr"] = 0.0
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        if steps is None:
            batches = _batch_iter(train_data, batch_size, shuffle, generator)
        else:
            # One freshly drawn minibatch per outer iteration.  The draw is from
            # the same ``generator`` the epoch path uses, so the minibatch
            # stream is part of the fit's seed and a repetition is reproducible.
            n_curves = len(train_data)
            k = n_curves if batch_size is None else min(int(batch_size), n_curves)
            draw = (torch.randperm(n_curves, generator=generator)[:k]
                    if shuffle else torch.arange(k))
            batches = (draw,)
        for idx in batches:
            batch = train_data.subset(idx)
            for _ in range(inner_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, batch, lambda_n)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
                optimizer.step()
                n_rollouts += 1
                epoch_loss += float(loss.detach())
                n_batches += 1

        if (epoch + 1) % check_every:
            continue

        train_metrics = evaluate(model, train_data, lambda_n)
        val_metrics = evaluate(model, val_data, lambda_n)
        grad_sq = _gradient_norm_sq(model, train_data, lambda_n)
        n_rollouts += 1

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "penalty": float(model.penalty()),
            "grad_sq": grad_sq,
            "eta_grad_sq": lr * grad_sq,
            "theta": model.theta_detached(),
        }
        history.append(record)

        if verbose:
            print(
                f"  epoch {epoch:4d}  train {record['train_loss']:.6e}  "
                f"val {record['val_loss']:.6e}  J {record['penalty']:.3e}  "
                f"eta*|g|^2 {record['eta_grad_sq']:.3e}"
            )

        if val_metrics["loss"] < best_val - 1e-12:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_theta = model.theta_detached()
            best_penalty = record["penalty"]
            stale = 0
        else:
            stale += 1

        if stopping_tau is not None and record["eta_grad_sq"] <= stopping_tau:
            stopped_epoch = epoch
            stopped_by = "gradient"
            break
        if patience is not None and stale >= patience:
            stopped_epoch = epoch
            stopped_by = "patience"
            break
        if (
            clock is not None
            and max_seconds is not None
            and clock() > max_seconds
        ):
            stopped_epoch = epoch
            stopped_by = "time"
            break

    if best_epoch < 0 and n_outer > 0 and select == "best":
        # Under ``select="final"`` the returned iterate is not the initial one,
        # so the "returned model is untrained" half of this warning would be
        # false -- the divergence is still worth reporting, but as a property of
        # the validation curve rather than of what is handed back.
        warnings.warn(
            "no epoch improved on the initial iterate: the fit diverged and the "
            "returned model is untrained (check lr, ode_step and the grid)",
            RuntimeWarning,
            stacklevel=2,
        )

    # Read before the restore below overwrites it, so that a ``select="best"``
    # fit still reports where the optimisation ended up.
    final_theta = model.theta_detached()
    final_penalty = float(model.penalty())

    if select == "final":
        # Leave the model where the optimisation actually stopped.  ``best_*``
        # is *not* overwritten: it still reports the validation curve, which is
        # the diagnostic it is meant to be, and ``selected_epoch`` says which
        # iterate ``theta`` came from.  It is an *epoch number*, like
        # ``best_epoch`` and ``history[*]["epoch"]`` -- not an index into
        # ``history``, which only holds every ``check_every``-th epoch.
        selected_epoch = history[-1]["epoch"] if history else -1
        return TrainResult(
            theta=model.theta_detached(),
            best_epoch=best_epoch,
            best_val=best_val,
            stopped_epoch=stopped_epoch,
            stopped_by=stopped_by,
            history=history,
            n_rollouts=n_rollouts,
            selected_epoch=selected_epoch,
            penalty=float(model.penalty()),
        )

    model.load_state_dict(best_state)
    return TrainResult(
        theta=best_theta,
        best_epoch=best_epoch,
        best_val=best_val,
        stopped_epoch=stopped_epoch,
        stopped_by=stopped_by,
        history=history,
        n_rollouts=n_rollouts,
        selected_epoch=best_epoch,
        penalty=best_penalty,
        final_theta=final_theta,
        final_penalty=final_penalty,
    )


def _gradient_norm_sq(
    model: ProfilingModel,
    data: Trajectories,
    lambda_n: float,
) -> float:
    """``||grad L_n(w)||^2`` at the current iterate, as Eq. (16) reads it."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        objective(model, data, lambda_n), params, allow_unused=True
    )
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.pow(2).sum())
    return total


# ---------------------------------------------------------------------------
# lambda_n selection
# ---------------------------------------------------------------------------
def lambda_grid(
    n_traj: int,
    spec_smoothness: float,
    order: float,
    dim: int,
    factors: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
) -> List[float]:
    """The paper's tuning ``lambda_n`` (Eq. 17) times a range of constants.

    ``lambda_n = n^{-2(a+r)/(D_s)} (log n)^{(a+r)/(s+r)}`` with
    ``D_s = 2(s + r) + d``; the exact constant is unknown, so the paper selects
    it on validation data.  This returns the grid that selection searches.
    """
    a = float(spec_smoothness)
    r = float(order)
    d = int(dim)
    D_s = 2.0 * (spec_smoothness + order) + d
    base = n_traj ** (-2.0 * (a + r) / D_s) * math.log(n_traj) ** (
        (a + r) / (spec_smoothness + order)
    )
    return [float(f * base) for f in factors]


def select_lambda(
    model_factory: Callable[[], ProfilingModel],
    train_data: Trajectories,
    val_data: Trajectories,
    candidates: Sequence[float],
    **train_kwargs,
) -> Dict[str, object]:
    """Fit once per candidate ``lambda_n`` and keep the best validation loss.

    ``model_factory`` must return a *fresh* model, so that each candidate starts
    from ``(theta_init, phi_0)`` as Algorithm 1 requires.
    """
    results = []
    best = None
    for lam in candidates:
        model = model_factory()
        res = train(model, train_data, val_data, lambda_n=float(lam), **train_kwargs)
        entry = {"lambda_n": float(lam), "result": res, "model": model}
        results.append(entry)
        if best is None or res.best_val < best["result"].best_val:
            best = entry
    return {"best": best, "all": results}
