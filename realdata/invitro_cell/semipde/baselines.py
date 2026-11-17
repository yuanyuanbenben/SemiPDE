"""The paper's baselines, plus the low-dimensional nuisance baseline.

Baseline 4
    Parameter cascading (Ramsay et al. 2007; Xun et al. 2013): a parametric
    state on a Fourier-in-space, B-spline-in-time basis, with the parameter
    fitted by an inner/outer cascade.  See :mod:`semipde.cascading`.  It lives
    here because it is one more column of the same table, but it is the one
    baseline the paper only *cites* -- it is run on the ``F = 0`` experiment,
    where a correctly specified parametric estimator is the natural competitor.

Baseline 1
    Purely parametric, ``L(u; theta) = 0``: the unknown mechanism is deleted and
    ``theta`` is the only free quantity.  It is not a separate implementation --
    it is :class:`~semipde.model.ProfilingModel` with ``use_f=False``, so it
    shares the estimator, the solver and the rollout with the proposed method and
    the comparison isolates the effect of the unknown mechanism.

Baseline 2
    Purely data-driven, ``u in S_{L,W}`` fitted to the observations with no PDE
    and no ``theta`` -- hence the ``--`` entries in the parameter columns of
    ``tab:simu:compared_baseline``.

Baseline 3
    The joint PINN of the main text: ``theta``, ``F`` and ``u`` are minimised
    simultaneously against a data-fit term and a PDE residual.

Both of the last two have to answer a question the paper never faces, because
its generative model (main text: ``Y_i = u(X_i; theta, F) + eps_i``) uses a
**single** solution ``u``: what is ``u`` when the sample pools several
trajectories?  Our sample pools ``M`` trajectories, each with its own randomly
generated initial condition, so a single ``(t, x) -> u`` map cannot represent it.
Rather than quietly fitting one trajectory -- which is what the legacy scripts do
and what makes them look worse than they are -- each baseline is given one
solution network *per trajectory*, with ``theta`` and ``F`` shared.  That is the
direct generalisation of the paper's estimator to a pooled sample, it uses every
observation, and it is the version of the baseline that is hardest to beat.

The legacy implementations, for the record:

* ``RD_nonpara.py`` trains on ``shelve[...][str(0)]`` -- one trajectory -- and
  compares against the **noiseless** copy of it, and its penalty and ``lambda``
  scaling are commented out.
* ``PINN_model.py`` carries hard-coded constants from the deepxde tutorial it was
  copied from (``1/(1+0.1(20-x)^2)``, ``u_t`` formed as ``U(t+0.1) - U(t)``
  *without* dividing by ``0.1``, ``lap_u`` scaled by an unexplained ``10``), and
  ``PINN_Net.__init__`` calls ``torch.manual_seed`` *inside* the layer loop, so
  every layer of every copy is initialised from the same RNG state.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import CaseConfig
from .data import Trajectories
from .model import apply_pointwise, build_network, coordinates
from .networks import FNN
from .pde import PDESpec

__all__ = [
    "SolutionNet",
    "NonparametricFit",
    "fit_nonparametric",
    "PINN",
    "fit_pinn",
    "FourierMechanism",
    "fourier_features",
    "ParametricModel",
    "BaselineModel",
    "BaselineResult",
    "fit_baseline",
]


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------
class SolutionNet(nn.Module):
    """``(t, x, ..., z) -> u``: one trajectory's solution, with no PDE.

    ``init="xavier"`` draws from the *global* RNG (see
    :meth:`~semipde.networks.FNN.reset_parameters`), so the seed argument only
    affects the Gaussian scheme; callers that need reproducibility should set
    ``torch.manual_seed`` around construction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (16, 64, 64, 16),
        activation: str = "tanh",
        normalize: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.net = FNN(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            power=1,
            activation=activation,
            normalize=normalize,
            init="xavier",
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _solution_inputs(
    config: CaseConfig,
    t: torch.Tensor,
    z_value: float,
    dtype: torch.dtype,
    device,
    t_is_leaf: bool = False,
) -> tuple:
    """Flattened design matrix, and (when ``t_is_leaf``) the time node to differentiate.

    Returns ``(x, t_col)`` with ``x`` of shape ``(T * prod(grid), 1 + n_spatial + 1)``
    and ``t_col`` the ``(T, *grid, 1)`` time tensor, or ``None`` when
    ``t_is_leaf`` is false.  ``d/dt`` is taken by autograd of the *first* input
    channel, so the caller must differentiate the same node returned here.
    """
    grid = config.grid
    nt = t.numel()
    # ``(T, 1, ..., 1, 1)``, one singleton per spatial axis, so that the
    # broadcast below reaches ``(T, *grid, 1)`` on a grid of any dimension.  A
    # fixed ``reshape(nt, 1, 1)`` only broadcasts on a one-dimensional grid and
    # raises on Case 2's and Case 4's.
    t_col = t.reshape(nt, *([1] * grid.dim), 1)
    if t_is_leaf:
        # The observation times arrive as plain data, so the leaf has to be
        # created here.  ``+ zeros`` then turns the broadcast view into a real
        # node with one distinct input per (time, grid point), which is what
        # makes the per-element autograd derivative come out at the right shape.
        t_col = t_col.detach().clone().requires_grad_(True)
        t_col = t_col + torch.zeros(nt, *grid.shape, 1, dtype=dtype, device=device)
    else:
        t_col = t_col.expand(nt, *grid.shape, 1)
    coords = coordinates(config, nt, dtype=dtype, device=device)
    zc = torch.full_like(coords[..., :1], float(z_value))
    x = torch.cat([t_col, coords, zc], dim=-1).reshape(-1, 1 + grid.dim + 1)
    return x, (t_col if t_is_leaf else None)


def _phase_net_parameters(module: nn.Module, requires_grad: bool) -> None:
    for prm in module.parameters():
        prm.requires_grad_(requires_grad)


# ---------------------------------------------------------------------------
# Baseline 2
# ---------------------------------------------------------------------------
@dataclass
class NonparametricFit:
    """One fitted regression per trajectory, plus their predictions."""

    models: List[SolutionNet]
    prediction: torch.Tensor  # (n, T, d_y, *grid), ``u_hat`` on the fitted data
    losses: List[float] = field(default_factory=list)

    @property
    def train_mse(self) -> float:
        return float(np.mean(self.losses)) if self.losses else float("nan")

    def theta_error(self) -> float:
        """Baseline 2 estimates no parameter; the paper reports ``--``."""
        return float("nan")


def fit_nonparametric(
    spec: PDESpec,
    config: CaseConfig,
    trajectories: Trajectories,
    *,
    hidden: Sequence[int] = (16, 64, 64, 16),
    activation: str = "tanh",
    epochs: int = 500,
    lr: float = 1e-3,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> NonparametricFit:
    """Fit baseline 2 on ``trajectories``.

    The estimator of the main text is a plain minimiser of the least-squares loss
    over ``S_{L,W}``, so on a new sample it is refit from scratch.  Train and test
    trajectories are therefore passed through this function separately.
    """
    grid = config.grid
    device = trajectories.states.device
    models: List[SolutionNet] = []
    preds: List[torch.Tensor] = []
    losses: List[float] = []

    for i in range(len(trajectories)):
        torch.manual_seed(seed + i)
        model = SolutionNet(
            1 + grid.dim + 1,
            spec.d_y,
            hidden=hidden,
            activation=activation,
            dtype=dtype,
        ).to(device)
        x, _ = _solution_inputs(
            config, trajectories.t, float(trajectories.z[i]), dtype, device
        )
        y = trajectories.states[i].reshape(-1, spec.d_y)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        step = min(batch_size or y.shape[0], y.shape[0])
        for _ in range(epochs):
            for start in range(0, y.shape[0], step):
                sl = slice(start, start + step)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x[sl]) - y[sl]).pow(2).mean()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            fitted = model(x).reshape(trajectories.states[i].shape)
        models.append(model)
        preds.append(fitted.detach())
        losses.append(float((fitted - trajectories.states[i]).pow(2).mean()))
        if verbose and (i + 1) % 8 == 0:
            print(f"    baseline 2: {i + 1}/{len(trajectories)} trajectories")

    return NonparametricFit(
        models=models, prediction=torch.stack(preds, dim=0), losses=losses
    )


# ---------------------------------------------------------------------------
# Baseline 3: joint PINN
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """``theta``, ``F`` and ``u`` estimated jointly.

    ``u`` is a Siren network of ``(t, x, z)``, one per trajectory (see the module
    docstring); ``f`` is the same network the SemiPDE model uses and ``theta`` is
    a shared trainable parameter block.  The residual is formed with autograd in
    time and with the *same* convolution stencils in space, so baseline 3 differs
    from the proposed method in the estimator, not in the discretisation.
    """

    def __init__(
        self,
        spec: PDESpec,
        config: CaseConfig,
        n_trajectories: int = 1,
        hidden: Sequence[int] = (32, 128, 128, 32),
        lambda_data: float = 1.0,
        lambda_ic: float = 1.0,
        theta: Optional[Dict[str, torch.Tensor]] = None,
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
        device=None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.config = config
        self.dtype = dtype
        self.lambda_data = float(lambda_data)
        self.lambda_ic = float(lambda_ic)
        self.in_dim = 1 + config.grid.dim + 1

        self.solutions = nn.ModuleList()
        for _ in range(n_trajectories):
            torch.manual_seed(seed)
            self.solutions.append(
                SolutionNet(
                    self.in_dim,
                    spec.d_y,
                    hidden=hidden,
                    activation="sine",
                    dtype=dtype,
                )
            )

        self.mechanism = build_network(config, spec, dtype=dtype, seed=seed)
        theta = spec.truth() if theta is None else theta
        self.theta = nn.ParameterDict(
            {
                k: nn.Parameter(torch.as_tensor(v, dtype=dtype).clone())
                for k, v in theta.items()
            }
        )
        self._coords: Optional[torch.Tensor] = None
        if device is not None:
            self.to(device)

    # -- pieces -----------------------------------------------------------------
    def _coords_for(self, u: torch.Tensor) -> torch.Tensor:
        if self._coords is None or self._coords.device != u.device:
            self._coords = coordinates(self.config, 1, dtype=u.dtype, device=u.device)
        return self._coords

    def _mechanism(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feats = self.spec.features(u, self._coords_for(u), z)
        return apply_pointwise(self.mechanism, feats).movedim(-1, 1)

    def solution_at(
        self,
        net: SolutionNet,
        t: torch.Tensor,
        z_value: float,
        *,
        t_is_leaf: bool = False,
    ):
        """``u`` on the grid, shape ``(T, d_y, *grid)``.

        Returns ``(u, t_col)``; ``t_col`` is ``None`` unless ``t_is_leaf``.
        """
        grid = self.config.grid
        x, t_col = _solution_inputs(
            self.config, t, z_value, self.dtype, t.device, t_is_leaf=t_is_leaf
        )
        out = net(x)
        u = out.reshape(t.numel(), *grid.shape, self.spec.d_y).movedim(-1, 1)
        return u, t_col

    def residual(
        self, net: SolutionNet, t: torch.Tensor, z_value: float, *, create_graph: bool = True
    ) -> torch.Tensor:
        """``d_t u - drift(u)``, shape ``(T, d_y, *grid)``.

        ``create_graph`` must be true when the result is to be differentiated
        again (training); the value-only path is cheaper and is what a validation
        pass wants.  Either way the caller needs grad mode *enabled* --- the time
        derivative is itself an autograd call.
        """
        u, t_col = self.solution_at(net, t, z_value, t_is_leaf=True)
        # ``u`` is ``(T, d_y, *grid)`` here, so ``u.sum()`` sums over the output
        # channels as well as the space and every field would receive the *sum*
        # of the fields' time derivatives.  That is invisible for ``d_y = 1``
        # and wrong for Case 2 (``d_y = 2``) and Case 4 (``d_y = 3``): the
        # residual of each field has to use its own ``d_t u``.  One vjp per
        # channel is the cheapest way to keep them apart; the forward graph is
        # shared, so the loop costs one extra backward pass per field.
        uts = [
            torch.autograd.grad(
                u[:, c].sum(), t_col, create_graph=create_graph, retain_graph=True
            )[0]
            for c in range(self.spec.d_y)
        ]
        ut = torch.cat(uts, dim=-1).movedim(-1, 1)  # (T, d_y, *grid)
        z = torch.full((1, 1), float(z_value), dtype=self.dtype, device=t.device)
        return ut - self.spec.drift(u, z, self.theta, self._mechanism)

    def loss(
        self,
        batch: Trajectories,
        *,
        create_graph: bool = True,
        index: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Data + initial-condition + residual, averaged over ``index``.

        ``index`` selects a sub-batch of trajectories and the returned means are
        taken over *that sub-batch*, not over the whole one.  This is what lets
        :func:`fit_pinn` accumulate the full-batch gradient in chunks: the
        caller scales each chunk's loss by ``len(chunk) / len(batch)``, and the
        sum of those scaled gradients is exactly the full-batch gradient.  The
        point is memory, not speed -- every ``self.solutions[i]`` kept in one
        graph holds its own activations, and with ``M = 256`` trajectories at
        ``hidden = (32, 128, 128, 32)`` that graph is what grows without bound.
        """
        data_t, ic_t, res_t = [], [], []
        need = len(batch) - len(self.solutions)
        if need > 0:  # a test batch that was never trained on
            raise ValueError(
                f"PINN has {len(self.solutions)} solution networks but was given "
                f"{len(batch)} trajectories; call `refit_solution` for test data"
            )
        idx = range(len(batch)) if index is None else [int(i) for i in index]
        for i in idx:
            net = self.solutions[i]
            z_value = float(batch.z[i])
            pred, _ = self.solution_at(net, batch.t, z_value)
            data_t.append((pred - batch.states[i]).pow(2).mean())
            ic, _ = self.solution_at(net, torch.zeros(1, dtype=self.dtype), z_value)
            ic_t.append((ic[0] - batch.initial[i]).pow(2).mean())
            res_t.append(
                self.residual(
                    net, batch.t, z_value, create_graph=create_graph
                ).pow(2).mean()
            )
        data = torch.stack(data_t).mean()
        init = torch.stack(ic_t).mean()
        residual = torch.stack(res_t).mean()
        return {
            "data": data,
            "ic": init,
            "residual": residual,
            "loss": residual + self.lambda_data * data + self.lambda_ic * init,
        }

    def theta_detached(self) -> Dict[str, object]:
        """Scalars as floats, vectors as lists -- see ``ProfilingModel``."""
        out: Dict[str, object] = {}
        for key, value in self.theta.items():
            flat = value.detach().reshape(-1)
            out[key] = float(flat[0]) if flat.numel() == 1 else [float(v) for v in flat]
        return out

    def mechanism_at(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``F_f(u)`` on arbitrary states, in the layout ``score_common`` uses.

        The private helper sizes its coordinate tensor for one trajectory,
        because that is how :meth:`residual` calls it.  Scoring evaluates the
        mechanism at every observed point of the held-out curves at once, so the
        coordinates have to be broadcast to that batch here.  ``f`` is shared
        across trajectories, so this is the same function the residual was
        trained against.
        """
        coords = coordinates(self.config, u.shape[0], dtype=u.dtype,
                             device=u.device)
        feats = self.spec.features(u, coords, z)
        return apply_pointwise(self.mechanism, feats).movedim(-1, 1)

    # -- prediction and test-set refitting --------------------------------------
    @torch.no_grad()
    def predict(self, batch: Trajectories) -> torch.Tensor:
        """``u_hat`` for batch entries ``0..len(self.solutions)-1``."""
        out = []
        for i in range(min(len(batch), len(self.solutions))):
            u, _ = self.solution_at(self.solutions[i], batch.t, float(batch.z[i]))
            out.append(u.detach())
        return torch.stack(out, dim=0)

    def refit_solution(
        self,
        batch: Trajectories,
        *,
        epochs: int = 300,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> torch.Tensor:
        """Fit a fresh ``u`` to each trajectory with ``theta`` and ``F`` frozen.

        This is what Baseline 3 predicts at a test point: the shared quantities
        are already estimated, and the only thing left is the solution of the
        given trajectory.
        """
        self.eval()
        _phase_net_parameters(self.mechanism, False)
        for prm in self.theta.parameters():
            prm.requires_grad_(False)
        predictions = []
        for i in range(len(batch)):
            torch.manual_seed(seed + i)
            net = SolutionNet(
                self.in_dim,
                self.spec.d_y,
                hidden=self.solutions[0].net.hidden,
                activation="sine",
                dtype=self.dtype,
            ).to(batch.states.device)
            z_value = float(batch.z[i])
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                pred, _ = self.solution_at(net, batch.t, z_value)
                loss = (pred - batch.states[i]).pow(2).mean()
                loss = loss + self.lambda_data * self.residual(
                    net, batch.t, z_value
                ).pow(2).mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final, _ = self.solution_at(net, batch.t, z_value)
                predictions.append(final.detach())
        _phase_net_parameters(self.mechanism, True)
        for prm in self.theta.parameters():
            prm.requires_grad_(True)
        return torch.stack(predictions, dim=0)


def fit_pinn(
    pinn: PINN,
    batch: Trajectories,
    *,
    n_train: Optional[int] = None,
    epochs: int = 100,
    inner_steps: int = 5,
    lr: float = 1e-3,
    batch_size: Optional[int] = None,
    seed: int = 0,
    select: str = "val",
    verbose: bool = False,
) -> Dict[str, object]:
    """Train the joint PINN with checkpoint selection.

    ``batch`` holds *every* curve the PINN owns a solution network for, and
    ``n_train`` says how many of them, counted from the front, are training
    curves; the rest are held out and only scored.  The one-batch signature is
    forced by the architecture: solution networks are indexed by curve position,
    so a validation set of *different* curves has no networks to evaluate and
    the old ``(train_data, val_data)`` call could not work at all.  Callers that
    want a held-out set concatenate it (see ``Trajectories.concat``); with
    ``n_train`` unset every curve trains and every curve validates.

    The held-out solution networks are never updated.  They therefore cannot
    adapt to their data, so their part of ``val_loss`` does *not* become a
    constant: it depends on ``theta`` and ``F`` through the residual, and it is
    dominated by ``||u_net - y||^2`` at an untrained network.  On this model
    the held-out curves are consequently not a usable selection set.  Measured
    on Case 1 (``ref/progress.md`` item 7): ``select="val"`` stopped at epoch
    0--19 and left ``theta`` at its zero initialisation, i.e. the PINN was
    selected before it had fitted anything.

    ``select`` is the checkpoint rule:

    ``"train"``
        The smallest *training* loss -- the legacy ``GFN_PINN_main.py`` rule.
        This is the default and the only rule that is meaningful for a model
        whose architecture has no out-of-sample representation.
    ``"val"``
        The smallest held-out loss, kept for the diagnostic that showed it
        selects the initialisation.
    ``"final"``
        No selection; the last epoch is scored.

    ``batch_size`` is given in **trajectories**, not scalars, and must be
    ``<= n_train``.  ``None`` (the default) keeps the legacy behaviour of one
    full-batch step.  A smaller value accumulates the gradient over
    trajectory-chunks -- each chunk's loss is scaled by
    ``len(chunk) / n_train`` before ``backward``, so the accumulated gradient is
    the full-batch gradient up to floating-point summation order, not a
    minibatch approximation.  The only thing that changes is peak memory: the
    backward graph never holds more than ``batch_size`` solution networks.
    """
    if select not in ("train", "val", "final"):
        raise ValueError(
            f"select must be 'train', 'val' or 'final', got {select!r}"
        )
    params = [p for p in pinn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    best_score = math.inf
    best_val = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(pinn.state_dict())
    history: List[Dict[str, float]] = []

    n_all = len(batch)
    n_traj = n_all if n_train is None else int(n_train)
    if not 0 < n_traj <= n_all:
        raise ValueError(
            f"n_train must lie in 1..{n_all} (the curves the PINN holds "
            f"networks for), got {n_train}"
        )
    train_index = list(range(n_traj))
    val_index = list(range(n_traj, n_all)) or train_index
    chunk = n_traj if batch_size is None else int(batch_size)
    if chunk <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    generator = torch.Generator().manual_seed(int(seed) + 104729)

    for epoch in range(epochs):
        for _ in range(max(1, inner_steps)):
            optimizer.zero_grad(set_to_none=True)
            if chunk >= n_traj:
                pinn.loss(batch, index=train_index)["loss"].backward()
            else:
                order = torch.randperm(n_traj, generator=generator).tolist()
                for start in range(0, n_traj, chunk):
                    part = [train_index[i] for i in order[start : start + chunk]]
                    losses = pinn.loss(batch, index=part)
                    (losses["loss"] * (len(part) / n_traj)).backward()
            optimizer.step()
        with torch.enable_grad():
            val = pinn.loss(batch, index=val_index, create_graph=False)
            val = {k: float(v.detach()) for k, v in val.items()}
            train = None
            if select == "train":
                train = pinn.loss(batch, index=train_index, create_graph=False)
                train = {k: float(v.detach()) for k, v in train.items()}
        entry = {
            "epoch": epoch,
            "val_loss": val["loss"],
            "val_residual": val["residual"],
            "val_data": val["data"],
            "theta": {k: float(v) for k, v in pinn.theta_detached().items()},
        }
        if train is not None:
            entry["train_loss"] = train["loss"]
            entry["train_residual"] = train["residual"]
            entry["train_data"] = train["data"]
        history.append(entry)
        if verbose:
            extra = (f"  train {entry['train_loss']:.4e}" if "train_loss" in entry
                     else "")
            print(
                f"  pinn epoch {epoch:4d}{extra}  val {entry['val_loss']:.4e}  "
                f"resid {entry['val_residual']:.4e}  "
                f"theta {entry['theta']}"
            )
        if entry["val_loss"] < best_val:
            best_val = entry["val_loss"]
        score = {"train": entry.get("train_loss", math.inf),
                 "val": entry["val_loss"],
                 "final": -math.inf}[select]
        if score < best_score or select == "final":
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(pinn.state_dict())

    pinn.load_state_dict(best_state)
    return {"best_val": best_val, "best_epoch": best_epoch, "select": select,
            "history": history, "theta": pinn.theta_detached()}


# ---------------------------------------------------------------------------
# the low-dimensional nuisance baseline
# ---------------------------------------------------------------------------
def fourier_features(x: torch.Tensor, n_modes: int) -> torch.Tensor:
    """``(..., 1) -> (..., 2 * n_modes + 1)`` on the ``[-1, 1]`` period.

    ``1, sin(pi x), cos(pi x), ..., sin(n pi x), cos(n pi x)`` along the last
    axis.  Applying this channel-wise to a ``(..., d_in)`` tensor is what
    :class:`FourierMechanism` does.
    """
    out = [torch.ones_like(x)]
    for k in range(1, n_modes + 1):
        out.append(torch.sin(math.pi * k * x))
        out.append(torch.cos(math.pi * k * x))
    return torch.cat(out, dim=-1)


class FourierMechanism(nn.Module):
    """A truncated Fourier series standing in for the unknown mechanism.

    Drop-in replacement for the centred network inside
    :class:`~semipde.model.ProfilingModel`: ``forward`` maps flattened pointwise
    features to ``d_y`` outputs, ``penalty`` returns the ridge penalty on the
    coefficients.  The estimate, the solver and the penalty are therefore all
    unchanged -- only the function space for ``F`` differs.

    This is the "competitive low-dimensional nuisance baseline" the new
    real-data plan asks for.  On a one-dimensional ``q(x)`` a Fourier basis is a
    strong competitor, so matching it is a meaningful result rather than a
    missing comparison.
    """

    def __init__(
        self,
        feature_dim: int,
        out_dim: int,
        n_modes: int = 8,
        period: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.out_dim = int(out_dim)
        self.n_modes = int(n_modes)
        self.period = float(period)
        width = self.feature_dim * (2 * self.n_modes + 1)
        self.coeff = nn.Parameter(torch.zeros(self.out_dim, width, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = fourier_features(x / self.period, self.n_modes)
        return basis @ self.coeff.t()

    def penalty(self) -> torch.Tensor:
        """Ridge penalty on the coefficients -- the analogue of ``J(F)``."""
        return self.coeff.pow(2).sum()


# ---------------------------------------------------------------------------
# the protocol the shared runner scores with
# ---------------------------------------------------------------------------
@dataclass
class BaselineResult:
    """What ``study``'s record writer reads off a fitted model.

    :class:`~semipde.train.TrainResult` carries the optimiser's own history --
    ``penalty``, ``radius``, the per-epoch trace -- and a baseline has none of
    that: ``R_n`` is a property of the *sieve* the proposal fits, and a data-only
    regression or a PINN has no ``J(F) = ||phi - phi_0||^2`` to report.  The
    fields are kept with ``nan`` rather than dropped so that a record written for
    a baseline has the same shape as one written for the proposal, and the
    comparison table can put them side by side without a special case.
    """

    fitted: bool = True
    penalty: float = float("nan")
    radius: float = float("nan")
    best_epoch: int = 0
    best_val: float = float("nan")


class ParametricModel:
    """Baseline 1 as the scorer sees it: the PDE rollout, with ``F = 0``.

    ``fit_baseline`` could hand back the ``ProfilingModel`` itself -- it is the
    object that predicts ``u`` and carries ``theta`` -- but ``score_common`` also
    asks for ``f_hat``, and with ``use_f=False`` the model's network is untrained
    and never enters the drift.  Reporting that network would describe a function
    Baseline 1 never used.  The baseline *assumes* the unknown mechanism is zero,
    so its implied estimate is exactly zero and
    ``||0 - f_0|| / ||f_0|| = 1`` is the honest nonparametric error; that is what
    :meth:`mechanism` returns, and the table can print it instead of a ``--``
    that would suggest the quantity is undefined.
    """

    def __init__(self, model: "ProfilingModel", spec: PDESpec) -> None:
        self.model = model
        self.spec = spec

    def __call__(self, initial, t, z) -> torch.Tensor:
        return self.model(initial, t, z)

    def mechanism(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self.spec.truth_f(u, z))

    def theta_detached(self) -> Dict[str, object]:
        return self.model.theta_detached()


@dataclass
class BaselineModel:
    """The slice of the ``ProfilingModel`` protocol ``study.score_common`` reads.

    ``score_common`` asks a model three things -- ``model(initial, t, z)``,
    ``model.mechanism(u, z)`` and ``model.theta_detached()`` -- and each baseline
    is missing at least one of them, for a reason that is structural rather than
    an omission: Baseline 2 has no ``theta`` and no mechanism at all, and the
    PINN does not predict from ``(initial, t, z)`` but by refitting a solution
    network.  Rather than give ``score_common`` three code paths, the baseline
    answers with what it has and with ``nan`` where it has nothing, and the
    missing column comes out ``nan``.

    That ``nan`` is not a placeholder for an unknown: it is the paper's own
    convention.  ``tab:simu:compared_baseline`` prints ``--`` for every parameter
    entry of Baseline 2, and a ``nan`` reaches that table as ``--`` without the
    table having to know which method it is printing.
    """

    spec: PDESpec
    prediction: torch.Tensor  # (n, T, d_y, *grid), the curves being scored
    theta: Dict[str, float] = field(default_factory=dict)
    #: ``(u, z) -> f_hat(u)`` when the method fits a mechanism.  ``None`` means
    #: it does not, and the column is ``nan`` -- see :meth:`mechanism`.
    mech: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    def __call__(self, initial, t, z) -> torch.Tensor:
        """The prediction made for the curves this object was built on.

        The arguments are accepted and ignored on purpose.  Every caller in the
        runner scores a baseline on the very curves its prediction was computed
        for -- ``score_common`` passes ``split.test`` -- and a baseline has no
        way to evaluate a *new* initial condition: Baseline 2 is one regression
        per trajectory, and Baseline 3 answers a new curve by refitting, which
        ``fit_baseline`` has already done.  Re-deriving a prediction from these
        arguments would silently fit a second model.
        """
        return self.prediction

    def mechanism(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``f_hat`` when the method estimates one, ``nan`` otherwise.

        Baseline 1 never reaches this class (see :class:`ParametricModel`).
        Baseline 2 puts all of its capacity into ``u`` and fits no mechanism, so
        its column is ``nan`` -- the same absence the parameter column records,
        not a zero.  The PINN does fit a shared ``f`` against its residual, and
        its ``mech`` callable is handed over by :func:`fit_baseline`, so the
        nonparametric error is measured for it rather than assumed away.  The
        shape of the ``nan`` branch comes from ``spec.truth_f`` so the column is
        missing rather than raising.
        """
        if self.mech is None:
            return torch.full_like(self.spec.truth_f(u, z), float("nan"))
        with torch.no_grad():
            return self.mech(u, z)

    def theta_detached(self) -> Dict[str, float]:
        """The estimate, keyed as ``spec.truth()`` so ``theta_error`` is defined.

        A baseline with no parameter reports ``nan`` per key rather than an empty
        dict, which :func:`~semipde.metrics.theta_error` would reject for missing
        keys -- a raise is not the way to say "this method estimates no theta".

        Values are passed through as the estimating object gave them, so Case 4's
        3-vectors stay lists; ``float`` on them is only defined for the scalar
        cases and coercing here would be a second, silent definition of the
        convention.
        """
        return {k: self.theta.get(k, float("nan")) for k in self.spec.truth()}


def fit_baseline(
    method: str,
    spec: PDESpec,
    config: CaseConfig,
    split,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: Optional[Sequence[int]] = None,
    pinn_hidden: Sequence[int] = (32, 128, 128, 32),
    pinn_batch: Optional[int] = None,
    refit_epochs: int = 300,
    theta_init: str = "zero",
    pinn_select: str = "train",
    batch_size: Optional[int] = None,
    cascading_lam: float = 100.0,
    cascading_mu: float = 1e-4,
    cascading_modes: int = 8,
    cascading_basis: int = 10,
    cascading_sieve: int = 6,
    cascading_iters: int = 150,
    verbose: bool = False,
):
    """Fit one of the paper's baselines and hand back something scoreable.

    Returns ``(model, result)``, where ``model`` satisfies the protocol
    :func:`~semipde.study.score_common` reads and ``result`` carries the few
    fields the record writer wants (see :class:`BaselineResult`).

    ``method`` is one of

    ``para``
        Baseline 1: :class:`~semipde.model.ProfilingModel` with ``use_f=False``,
        trained by the *same* :func:`~semipde.train.train` loop the proposal
        uses.  It is not a reimplementation, so the comparison isolates the
        unknown mechanism and nothing else.
    ``nonpara``
        Baseline 2: one :class:`SolutionNet` per trajectory, fitted to that
        trajectory's *noisy* observations with no PDE.  It is scored on the
        curves it was fitted on, because it has no shared unknown through which
        a train curve could inform a test one -- the ``fit_nonparametric``
        docstring makes the same point.  That is the protocol the legacy
        ``*_nonpara.py`` drivers used, and it is *in-sample* against the
        proposal's held-out curves, i.e. it is the version of this baseline
        hardest to beat.  ``theta`` is ``nan`` by construction.
    ``pinn``
        Baseline 3: the joint PINN, trained on the train curves with
        checkpoint selection (``pinn_select``, see :func:`fit_pinn`) and then
        asked to predict the held-out ones.  Those have no solution network of
        their own, so ``refit_solution`` fits one per test curve with ``theta``
        and ``F`` frozen -- which is what the paper's estimator does with a new
        trajectory, and what makes the two comparable.  ``pinn_select`` defaults
        to ``"train"`` (the legacy rule): the held-out curves have no trained
        representation, so a held-out loss selects the initialisation rather
        than the fit (``ref/progress.md`` item 7).

    ``cascading``
        Baseline 4: parameter cascading (Ramsay et al. 2007; Xun et al. 2013),
        see :mod:`semipde.cascading`.  It fits the *same* semiparametric model
        as the proposal -- the unknown mechanism is a tensor B-spline sieve, the
        state a Fourier/B-spline basis -- and differs only in how the three
        levels are estimated: an alternation over the state coefficients and
        the mechanism coefficients inside a search over ``theta``, instead of
        joint optimisation.  ``cascading_modes`` and ``cascading_basis`` size
        the state basis, ``cascading_sieve`` sizes the mechanism sieve per
        axis; none of them is selected per case, because the effective
        smoothness is carried by ``cascading_lam`` (PDE residual against the
        data fit) and ``cascading_mu`` (second-difference penalty on the
        mechanism), which are the method's smoothing parameters.

    ``theta_init`` is the PINN's starting ``theta``.  ``"zero"`` (the default)
    reproduces the legacy ``PINN_model.py``, which initialises the parameter at
    ``0``; ``"truth"`` starts it at ``theta_0`` and is there so that a reader who
    suspects the PINN merely failed to find the basin can check the other
    convention.  Starting at the truth is not a fair baseline and is not the
    default.
    """
    if method == "cascading":
        # Baseline 4 -- the parameter-cascading estimator of the paper's
        # introduction (Ramsay et al. 2007; Xun et al. 2013), applied to the
        # semiparametric model: a sieve for ``f``, a basis for ``u``, and the
        # parameter searched outside the two.  It does not solve the PDE
        # forward, so it is the classical alternative to joint profiling rather
        # than to the model itself.
        from .cascading import CascadingModel, fit_cascading

        fit = fit_cascading(
            spec, config, split,
            lam=cascading_lam, mu=cascading_mu, n_modes=cascading_modes,
            n_basis_t=cascading_basis, n_basis_f=cascading_sieve,
            inner_iters=cascading_iters, seed=seed, verbose=verbose,
        )
        return CascadingModel(spec, fit), BaselineResult()

    if method == "para":
        model = ProfilingModel(spec, config, use_f=False, seed=seed)
        train(
            model,
            split.train,
            split.val,
            config=config,
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=batch_size,
            verbose=verbose,
        )
        return ParametricModel(model, spec), BaselineResult()

    if method == "nonpara":
        fit = fit_nonparametric(
            spec,
            config,
            split.test,
            hidden=hidden or (16, 64, 64, 16),
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=batch_size,
            verbose=verbose,
        )
        # ``theta`` stays empty, so every parameter column reports ``nan`` --
        # the paper's ``--``.
        return BaselineModel(spec, fit.prediction), BaselineResult()

    if method == "pinn":
        theta = None
        if theta_init == "zero":
            theta = {k: torch.zeros_like(v) for k, v in spec.truth().items()}
        elif theta_init != "truth":
            raise ValueError(f"theta_init must be 'zero' or 'truth', got {theta_init!r}")
        # The PINN holds one solution network per curve, so the train and
        # validation curves are handed over as one batch with the first
        # ``len(split.train)`` marked as training.  The held-out networks are
        # never trained, which is why the checkpoint rule defaults to the
        # training loss (``pinn_select``) -- see :func:`fit_pinn`.
        curves = split.train.concat(split.val)
        pinn = PINN(
            spec,
            config,
            n_trajectories=len(curves),
            hidden=pinn_hidden,
            theta=theta,
            seed=seed,
        )
        out = fit_pinn(
            pinn,
            curves,
            n_train=len(split.train),
            epochs=epochs,
            lr=lr,
            batch_size=pinn_batch,
            seed=seed,
            select=pinn_select,
            verbose=verbose,
        )
        prediction = pinn.refit_solution(split.test, epochs=refit_epochs, lr=lr, seed=seed)
        history = out["history"]
        result = BaselineResult(
            best_val=float(history[out["best_epoch"]]["val_loss"]),
            best_epoch=int(out["best_epoch"]),
        )
        return BaselineModel(
            spec, prediction, theta=pinn.theta_detached(),
            mech=pinn.mechanism_at
        ), result

    raise ValueError(
        f"unknown baseline {method!r}; expected one of 'para', 'nonpara', "
        f"'pinn', 'cascading'"
    )


# Imported at the bottom so the module can be read top-down: the three baselines
# above need ``nn``, ``torch`` and the model/network builders, while this is the
# *runner's* optimiser and would be a circular import at the top
# (``train`` imports ``model``, ``model`` imports ``config`` and ``pde``).
from .model import ProfilingModel  # noqa: E402
from .train import train  # noqa: E402
