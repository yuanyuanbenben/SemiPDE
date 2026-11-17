"""The SemiPDE profiling model: known mechanism plus centred neural network.

This is the object the estimator of Eq. (3) minimises over,

.. math::

    \\hat\\theta, \\hat F = \\arg\\min_{\\theta, F\\in S_n}
        L_n(\\theta, F) + \\lambda_n J(F),

assembled from a :class:`~semipde.pde.PDESpec` (which knows ``G_z`` and the
initial condition) and a centred network (which supplies ``F_f``).

The ODE is integrated on the same substep schedule as the reference solver, so
the observation times are hit exactly rather than approximately.  That matters
here: with ``torchdiffeq``'s fixed-step Euler the last step is truncated unless
the step size divides the interval, which silently shifts every time point.

Sign convention
---------------
``drift = G_z(u; theta) + F_f(u)``.  The paper writes the model as
``L_z(u_z; theta) + F_f(u_z) = 0``; the two agree after moving the time
derivative to one side, and the sign of ``F`` is absorbed into the
parameterisation of ``f``.  Everything downstream only ever sees the drift.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as _checkpoint

from .config import CaseConfig
from .networks import CenteredFNN, DoubleFNN, FNN
from .pde import PDESpec, Theta

__all__ = [
    "apply_pointwise",
    "AugmentedMechanism",
    "ProfilingModel",
    "build_network",
    "coordinates",
    "starting_theta",
]

#: Above this many Euler steps per trajectory, :meth:`ProfilingModel.rollout`
#: checkpoints each observation interval instead of keeping every intermediate.
CHECKPOINT_MIN_STEPS = 64


def starting_theta(spec: PDESpec, mode: str = "zero") -> Theta:
    """The ``theta`` the optimisation starts from, shaped like ``spec.truth()``.

    ``"zero"`` reproduces the legacy convention and therefore the published
    tables (see :attr:`~semipde.config.CaseConfig.theta_init`); ``"truth"``
    starts at the answer, which is useful for looking at the local regime the
    theory describes and wrong for anything measuring error.
    """
    truth = spec.truth()
    if mode == "truth":
        return {k: torch.as_tensor(v).clone() for k, v in truth.items()}
    if mode == "zero":
        # Zeros, not a scalar zero: a vector-valued theta (Case 4's
        # ``theta_1, theta_2`` are 3-vectors) has to keep its shape.
        return {k: torch.zeros_like(torch.as_tensor(v)) for k, v in truth.items()}
    raise ValueError(f"unknown theta_init {mode!r}; expected 'zero' or 'truth'")



def coordinates(
    config: CaseConfig,
    batch: int,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Physical coordinates broadcast to ``(batch, *grid, n_spatial)``."""
    mesh = config.grid.mesh()  # one array per axis, each of shape *grid
    stacked = np.stack(mesh, axis=-1)  # (*grid, n_spatial)
    out = torch.as_tensor(stacked, dtype=dtype)
    out = out.unsqueeze(0).expand(batch, *out.shape)
    return out if device is None else out.to(device)


def apply_pointwise(net: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Evaluate ``net`` at every grid point.

    ``features`` is ``(B, *grid, d_in)`` and the result is ``(B, *grid, d_out)``.
    The network is applied along the last axis, so it never mixes grid points --
    the map is pointwise by construction.
    """
    lead = features.shape[:-1]
    flat = features.reshape(-1, features.shape[-1])
    return net(flat).reshape(*lead, -1)


class AugmentedMechanism(nn.Module):
    """A centred network plus a pointwise linear term.

    Algorithm 2 asks for the ``h_j`` that makes
    ``u(theta_hat + delta e_j, f_hat)`` and ``u(theta_hat, f_hat + delta h_j)``
    agree.  Both perturbations enter the drift in the same slot, so ``h_j`` has
    to reproduce an ``O(1)`` drift correction -- but a centred network starts at
    output zero with weights near their initialisation, and for a deep net those
    weights have to grow by two or three orders of magnitude before the output
    reaches ``O(1)``.  Adam spends most of its budget on that growth, which is
    why the inner fit needs thousands of rollout steps and why it is the part of
    Algorithm 2 that does not scale to the 50 repetitions the confidence
    intervals need.

    A linear functional of the features reaches the same ``O(1)`` output in a
    handful of steps -- the map is linear in its parameters, so there is no
    growth phase.  It is admissible: the paper's ``h_j`` ranges over a network
    class, and a linear functional is the lowest-order element of it.  Adding it
    therefore only enlarges the inner search space, and ``M_hat_ij`` moves closer
    to the ``M_0`` it estimates.

    Weights and bias are initialised to **zero**, not to the usual random draw,
    so the untrained surrogate is exactly the finite-difference baseline and the
    ``mismatch_rms_init`` diagnostic keeps its meaning.
    """

    def __init__(
        self,
        network: nn.Module,
        in_dim: int,
        out_dim: int,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.network = network
        self.linear = nn.Linear(in_dim, out_dim, dtype=dtype, device=device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features) + self.linear(features)

    #: ``J(h_j)`` penalises the network only.  The linear term is the part that
    #: is *not* a nuisance -- it is what any smooth ``h`` looks like locally --
    #: so regularising it would bias the correction towards zero exactly where
    #: it is most identifiable.
    def penalty(self) -> torch.Tensor:
        return self.network.penalty()


def build_network(
    config: CaseConfig,
    spec: PDESpec,
    in_dim: Optional[int] = None,
    hidden: Optional[Sequence[int]] = None,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    device=None,
) -> nn.Module:
    """A centred network for the unknown mechanism of ``spec``.

    Normally one block per output component of the mechanism, each reading the
    whole feature vector: ``(f_1(u), ..., f_{d_y}(u))``, with Case 3's drift
    vector as the variant that is *not* one component per field (see
    :attr:`~semipde.pde.PDESpec.mechanism_out_dim`).  With
    ``config.mechanism_shared`` the components share one trunk and differ only
    in the output row, which is the architecture the supplement specifies and
    the legacy code built; see that field for why the difference is not
    cosmetic on the two-field cases.

    ``hidden`` overrides ``config.hidden`` for this network alone.  The two
    places that build a mechanism want different widths: the estimator is sized
    by the study's ``n``, while Algorithm 2's inner ``h_j`` is a *diagnostic*
    fit whose width is an axis of its own -- ``estimate_covariance`` exposes it
    as ``inner_hidden``.  Defaulting to ``config.hidden`` keeps every existing
    caller bit-for-bit unchanged.
    """
    in_dim = spec.feature_dim if in_dim is None else int(in_dim)
    hidden = config.hidden if hidden is None else tuple(int(h) for h in hidden)
    out_dim = int(spec.mechanism_out_dim)

    def _fnn(seed_j: Optional[int]) -> FNN:
        return FNN(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            power=config.repu_power,
            activation=config.activation,
            normalize=config.normalize,
            init=config.init,
            seed=seed_j,
            dtype=dtype,
            device=device,
        )

    if config.mechanism_shared:
        # ``seed`` unchanged rather than ``seed + 0``: identical for every
        # single-output case, so nothing that shipped changes.
        return CenteredFNN(_fnn(seed))

    blocks = []
    for j in range(out_dim):
        blocks.append(
            FNN(
                in_dim=in_dim,
                hidden=hidden,
                out_dim=1,
                power=config.repu_power,
                activation=config.activation,
                normalize=config.normalize,
                init=config.init,
                seed=None if seed is None else seed + j,
                dtype=dtype,
                device=device,
            )
        )
    if len(blocks) == 1:
        return CenteredFNN(blocks[0])
    return DoubleFNN(blocks)


class ProfilingModel(nn.Module):
    """``(theta, f)`` together, integrated to the observation times.

    Parameters
    ----------
    spec
        The PDE case.
    config
        Case configuration; supplies ``hidden``, ``ode_step`` and the optimiser
        settings used by :mod:`semipde.train`.
    theta
        Initial parameter values.  Defaults to the spec's truth, which is the
        same starting point the paper uses to compare against the oracle.
    network
        A centred network for ``F_f``.  Built from ``config`` when omitted.
    use_f
        Set ``False`` for Baseline 1: the model is then purely parametric and
        ``f`` is dropped from the drift and from the penalty.
    base_mechanism
        A frozen pointwise term added to the drift before ``network`` and
        excluded from the penalty.  Algorithm 2 needs it: the perturbation
        ``h_j`` is trained against the drift ``G + F_{f_hat} + delta * h_j``,
        where ``f_hat`` is the already-fitted mechanism and only ``h_j`` is
        optimised with its own ``J``.
    """

    def __init__(
        self,
        spec: PDESpec,
        config: CaseConfig,
        theta: Optional[Theta] = None,
        network: Optional[nn.Module] = None,
        use_f: bool = True,
        base_mechanism: Optional[nn.Module] = None,
        mechanism_scale: float = 1.0,
        dtype: torch.dtype = torch.float64,
        seed: Optional[int] = None,
        device=None,
        feature_map: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.config = config
        self.use_f = bool(use_f)
        self.mechanism_scale = float(mechanism_scale)
        self.dtype = dtype
        self.feature_map = feature_map
        # Cache of the broadcast coordinate tensor; rebuilt whenever the batch
        # size or device changes.  Deliberately not a buffer: it must follow the
        # batch, not the module.
        self._coords: Optional[torch.Tensor] = None

        # ``theta=None`` means "start where the config says", not "start at the
        # truth": see ``CaseConfig.theta_init`` for why the default is zero and
        # why starting at the truth is the inconsistency, not the rule.
        theta = (starting_theta(spec, config.theta_init)
                 if theta is None else theta)
        self.theta = nn.ParameterDict(
            {
                k: nn.Parameter(torch.as_tensor(v, dtype=dtype).clone())
                for k, v in theta.items()
            }
        )
        self.network = (
            network
            if network is not None
            else build_network(config, spec, dtype=dtype, seed=seed, device=device)
        )
        self.base_mechanism = base_mechanism
        if base_mechanism is not None:
            for p in base_mechanism.parameters():
                p.requires_grad_(False)
        if device is not None:
            self.to(device)

    # -- drift ------------------------------------------------------------------
    def _drift(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        f = self._mechanism if self.use_f else None
        return self.spec.drift(u, z, self.theta, f)

    def _mechanism(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``F_f(u)``, the unknown part of the drift.

        ``spec.drift`` passes the *state* here; we convert it to pointwise
        features and evaluate the network, keeping the ``(B, d_y, *grid)``
        shape the drift expects.

        ``mechanism_scale`` multiplies **only** the trainable network, not the
        frozen ``base_mechanism``.  Algorithm 2 depends on that separation: the
        surrogate is the model with drift ``G_z(theta_hat) + F_hat + delta * h_j``,
        so the network produces ``h_j`` at its own natural scale while its
        *effect* on the drift carries the finite-difference factor ``delta``.
        Without the factor the network is asked to emit a drift correction of
        size ``delta * d_theta u`` -- around ``1e-6`` -- which no step size can
        reach without overshooting by orders of magnitude.
        """
        coords = self._coordinates_for(u)
        feats = self.spec.features(u, coords, z)
        # ``feature_map`` lets Algorithm 2's inner ``h_j`` read a *different*
        # input set from the fitted ``f_hat`` (the base mechanism keeps the
        # spec's).  This is a deliberate deviation from the paper, and it is the
        # only lever that has moved ``se/sd`` on Case 1: ``h_0`` is
        # ``d_theta f_hat(theta_0)``, so an ``h_j`` reading inputs the fitted
        # nuisance cannot read makes ``M_hat`` smaller than any ``M_0`` the
        # estimator has, and ``se`` correspondingly too large.  See the note on
        # ``estimate_covariance(inner_features=...)``.
        net_feats = feats if self.feature_map is None else self.feature_map(
            u, coords, z
        )
        out = apply_pointwise(self.network, net_feats)  # (B, *grid, d_y)
        if self.mechanism_scale != 1.0:
            out = out * self.mechanism_scale
        if self.base_mechanism is not None:
            out = out + apply_pointwise(self.base_mechanism, feats)
        return out.movedim(-1, 1)

    def _coordinates_for(self, u: torch.Tensor) -> torch.Tensor:
        coords = self._coords
        if coords is None or coords.shape[0] != u.shape[0] or coords.device != u.device:
            self._coords = coordinates(
                self.config, u.shape[0], dtype=u.dtype, device=u.device
            )
        return self._coords

    # -- integration ------------------------------------------------------------
    def rollout(
        self,
        initial: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        checkpoint: Optional[bool] = None,
    ) -> torch.Tensor:
        """Integrate from the observed initial state to every time in ``t``.

        ``initial`` is ``(B, d_y, *grid)`` and the result is
        ``(T, B, d_y, *grid)`` -- the ``torchdiffeq`` layout, so that callers can
        compare directly against the simulated trajectories.

        The internal step is ``config.ode_step`` with as many substeps as needed
        for each observation interval, which reproduces the paper's fixed-step
        Euler while landing exactly on the observation times.

        Memory
        ------
        With the paper's step of ``1e-3`` over ``[0, 2.5]`` a trajectory needs
        2500 Euler steps.  Backpropagating through all of them retains every
        intermediate activation, which for Case 1 already runs to tens of
        gigabytes.  ``checkpoint=True`` stores only the states at the
        observation times and recomputes each interval during the backward pass,
        trading roughly a factor of two in compute for O(T) memory.  It is on by
        default whenever the trajectory is longer than :data:`CHECKPOINT_MIN_STEPS`.
        """
        t = torch.as_tensor(t, dtype=initial.dtype, device=initial.device)
        if z.ndim == 1:
            z = z.view(-1, 1)
        z = z.to(initial.dtype)
        # A *path* input -- ``(B, n_obs, 1)`` -- carries one known input per
        # observation time, as measured boundary data do.  It is held constant
        # over each observation interval at the average of its two endpoints,
        # which is the midpoint rule for the flux the interval actually
        # delivers; the alternative (either endpoint) is first-order in the
        # interval and biases ``theta`` when the input ramps, as the heater in
        # the heat-conduction application does.
        path = z.ndim == 3
        if path and z.shape[1] != t.shape[0]:
            raise ValueError(
                f"a path input must have one entry per observation time, got "
                f"{z.shape[1]} inputs for {t.shape[0]} times"
            )
        if z.ndim > 3:
            raise ValueError(f"z must be (B, 1) or (B, n_obs, 1), got {tuple(z.shape)}")

        spans = [float(t[i] - t[i - 1]) for i in range(1, t.shape[0])]
        substep_counts = [
            max(1, int(round(s / self.config.ode_step))) for s in spans
        ]
        if checkpoint is None:
            # The wrapper is per *segment*, so it only pays for itself when the
            # segment is long.  At the paper's data step of ``1e-3`` a segment is
            # a hundred Euler steps and checkpointing is what keeps the graph
            # from growing without bound; at the training step of ``2e-2`` it is
            # five, and wrapping 24 such segments measured 18% slower on the CPU
            # and 54% slower on the GPU while saving nothing worth saving.
            # An explicit ``checkpoint=True`` still means every segment.
            checkpoint = (
                sum(substep_counts) > CHECKPOINT_MIN_STEPS
                and max(substep_counts, default=0) >= CHECKPOINT_MIN_STEPS
            )

        out = [initial]
        u = initial
        for j, (span, substeps) in enumerate(zip(spans, substep_counts), start=1):
            h = span / substeps
            zj = torch.add(z[:, j - 1], z[:, j]).mul_(0.5) if path else z
            if checkpoint:
                u = _checkpoint.checkpoint(
                    self._segment,
                    u,
                    zj,
                    h,
                    substeps,
                    use_reentrant=False,
                    # No stochastic layers here, so there is no RNG state to
                    # preserve; skipping it removes a per-segment device sync.
                    preserve_rng_state=False,
                )
            else:
                u = self._segment(u, zj, h, substeps)
            out.append(u)
        return torch.stack(out, dim=0)

    def _segment(
        self,
        u: torch.Tensor,
        z: torch.Tensor,
        h: float,
        substeps: int,
    ) -> torch.Tensor:
        """Advance ``u`` by ``substeps`` steps of size ``h``.

        The scheme is :attr:`~semipde.config.CaseConfig.ode_method`, and it is
        not a cosmetic choice: this rollout is what the estimator fits the data
        *with*, so its discretisation error is a bias in ``theta_hat``, not
        merely an accuracy loss.  Euler is what the paper's implementation
        details specify (step ``0.001``, supp. eq. implementation paragraphs);
        RK4 is here because at the *same* cost it is far more accurate --
        matching the paper's ``1e-3`` Euler needs 1008 drift evaluations per
        forward pass on Example 2, while RK4 at the legacy ``2e-2`` needs 192
        and lands at ``1e-9`` rather than ``1e-5``.  See ``runs/ode_trade``
        for the measured bias against cost.
        """
        step = self._rk4_step if self.config.ode_method == "rk4" else self._euler_step
        for _ in range(substeps):
            u = step(u, z, h)
        return u

    def _euler_step(self, u: torch.Tensor, z: torch.Tensor, h: float) -> torch.Tensor:
        return u + h * self._drift(u, z)

    def _rk4_step(self, u: torch.Tensor, z: torch.Tensor, h: float) -> torch.Tensor:
        """Classical RK4.  Four drift evaluations, ``O(h^4)`` local error."""
        k1 = self._drift(u, z)
        k2 = self._drift(u + 0.5 * h * k1, z)
        k3 = self._drift(u + 0.5 * h * k2, z)
        k4 = self._drift(u + h * k3, z)
        return u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forward(
        self,
        initial: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predictions in the legacy ``(B, T, d_y, *grid)`` layout.

        Projected through :meth:`PDESpec.observe`, so a case whose data live at
        a handful of grid points hands back exactly those: the loss, the
        residual variance and Algorithm 2's matching problem are all defined on
        the observed points, never on the interpolated ones.
        """
        return self.spec.observe(self.rollout(initial, t, z).movedim(0, 1))

    # -- the penalised objective ------------------------------------------------
    def penalty(self) -> torch.Tensor:
        """``J(F) = ||phi - phi_0||^2``; zero when ``f`` is switched off."""
        if not self.use_f:
            return torch.zeros((), dtype=self.dtype)
        return self.network.penalty()

    def mechanism(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``F_f(u)`` at states ``u`` -- the fitted mechanism, for scoring.

        Public because ``||f_hat - f_0||`` cannot be computed from outside the
        model: the features, the ``mechanism_scale`` and the frozen base
        mechanism all have to be applied in the same way the drift applies them,
        and getting one of those wrong makes the reported mechanism error a
        comparison against a different function than the one being fitted.
        """
        return self._mechanism(u, z)

    def theta_values(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.theta.items()}

    def theta_detached(self) -> Dict[str, object]:
        """The estimate as plain Python, keyed as ``spec.truth()``.

        A scalar parameter gives a float and a vector one -- Case 4's
        ``theta_1``, ``theta_2`` -- gives a list, because ``float`` is only
        defined on the former.  The dict is JSON-serialisable either way, which
        is what the run records and the history entries need.
        """
        out: Dict[str, object] = {}
        for key, value in self.theta.items():
            flat = value.detach().reshape(-1)
            out[key] = float(flat[0]) if flat.numel() == 1 else [float(v) for v in flat]
        return out

    def freeze_network(self) -> None:
        """Stop updating ``phi``; used by the inference step of Algorithm 2."""
        for p in self.network.parameters():
            p.requires_grad_(False)

    def freeze_theta(self) -> None:
        for p in self.theta.parameters():
            p.requires_grad_(False)

    def parameter_blocks(self) -> Sequence[Tuple[str, nn.ParameterList]]:
        """The two blocks Algorithm 1 alternates between."""
        return (("theta", self.theta.parameters()), ("phi", self.network.parameters()))

    def trainable_count(self) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"case={self.spec.name}, use_f={self.use_f}, d_y={self.spec.d_y}"
