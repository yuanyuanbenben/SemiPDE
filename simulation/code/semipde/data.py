"""Deterministic trajectory generation.

Two things are deliberately different from the legacy generator:

* **Reproducibility.**  Every trajectory draws its initial condition and its
  noise from ``numpy.random.default_rng(f(seed, index))``.  The legacy code
  called the *global* ``np.random.seed`` from ``Dataset.__init__`` and then
  sampled lazily inside ``__getitem__``, so which initial condition landed on
  which index depended on the DataLoader's shuffle order.

* **Inputs.**  ``Z`` is sampled per trajectory and stored alongside the
  trajectory, so that downstream code can index the known mechanism by input
  instead of assuming a single condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from .config import CaseConfig, InputSpec
from .grids import TimeGrid
from .pde import PDESpec, Theta

__all__ = ["Trajectories", "solve", "generate", "curve_seed", "TRUTH"]

#: Sentinel meaning "use the spec's own ``truth_f``".  Distinct from ``None``,
#: which switches the unknown mechanism *off* (Baseline 1 / the ``F = 0``
#: ablation).  Passing the wrong one silently generates data from the wrong
#: equation, so the distinction is worth a name.
TRUTH = "truth"


def curve_seed(seed: int, index: int) -> int:
    """A stable, order-independent seed for trajectory ``index`` of a run."""
    return (int(seed) * 1_000_003 + int(index)) % (2 ** 63 - 1)


# ---------------------------------------------------------------------------
# reference solver
# ---------------------------------------------------------------------------
def solve(
    spec: PDESpec,
    u0: torch.Tensor,
    time: TimeGrid,
    theta: Theta,
    z: Optional[torch.Tensor] = None,
    f=TRUTH,
    *,
    substeps: Optional[int] = None,
    ode_step: float = 1e-3,
    method: str = "rk4",
) -> torch.Tensor:
    """Integrate the reference solution at the observation times.

    ``u0`` has shape ``(n, d_y, *grid)``.  ``substeps`` internal steps are taken
    between consecutive observation times so that the observation times are hit
    exactly; it defaults to ``round(time.dt / ode_step)``.  ``f`` defaults to
    the spec's true mechanism; pass ``f=None`` to drop it.
    """
    if f is TRUTH:
        f = spec.truth_f
    if substeps is None:
        substeps = max(1, int(round(time.dt / ode_step)))
    h = time.dt / substeps

    if z is None:
        z = torch.zeros(u0.shape[0], 1, dtype=u0.dtype, device=u0.device)
    z = torch.as_tensor(z, dtype=u0.dtype, device=u0.device)
    if z.ndim == 1:
        z = z.view(-1, 1)

    out = [u0]
    u = u0
    for _ in range(time.n_obs - 1):
        for _ in range(substeps):
            u = _step(spec, u, z, theta, f, h, method)
        out.append(u)
    traj = torch.stack(out, dim=1)  # (n, T, d_y, *grid)
    return traj


def _step(spec, u, z, theta, f, h, method):
    if method == "euler":
        return u + h * spec.drift(u, z, theta, f)
    if method == "rk4":
        k1 = spec.drift(u, z, theta, f)
        k2 = spec.drift(u + 0.5 * h * k1, z, theta, f)
        k3 = spec.drift(u + 0.5 * h * k2, z, theta, f)
        k4 = spec.drift(u + h * k3, z, theta, f)
        return u + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"unknown integration method {method!r}")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@dataclass
class Trajectories:
    """A batch of noisy trajectories plus their conditioning.

    Attributes
    ----------
    states
        ``(n, n_obs, d_y, *grid)`` observed values (solution plus noise).
    clean
        The noiseless solution, same shape; used only for scoring.
    z
        ``(n,)`` input values.
    t
        ``(n_obs,)`` observation times.
    noise_mask
        Boolean tensor shaped like ``states``, ``True`` exactly on the points the
        observation model actually corrupted, i.e. the points that enter
        ``Y_i = u_{Z_i}(X_i) + eps_i``.  With ``noise_at_initial=False`` the
        ``t = 0`` slice is kept exact, so it is ``False`` there and ``True``
        everywhere else.  ``None`` means "no mask was recorded" (no noise at
        all, or a legacy file); callers must then fall back to the whole array.
        The residual-variance plug-in *must* use this mask: averaging over the
        exact slice divides ``sigma_hat^2`` down by the fraction of live points,
        which on a 25-point time grid is a 4% deflation and a 2% short standard
        error.
    """

    states: torch.Tensor
    clean: torch.Tensor
    z: torch.Tensor
    t: torch.Tensor
    initial: torch.Tensor
    sigma: float
    noise_mask: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.states.shape[0]

    @property
    def noisy(self) -> Optional[torch.Tensor]:
        """The mask flattened to ``states`` shape, or ``None`` if unrecorded."""
        if self.noise_mask is None:
            return None
        return self.noise_mask.reshape(self.states.shape).to(self.states.device)

    @property
    def noise_fraction(self) -> float:
        """Share of observation points that actually carry noise."""
        mask = self.noisy
        return 1.0 if mask is None else float(mask.to(torch.float64).mean())

    @property
    def grid_shape(self) -> Sequence[int]:
        return tuple(self.states.shape[3:])

    @property
    def d_y(self) -> int:
        return self.states.shape[2]

    def subset(self, index) -> "Trajectories":
        idx = torch.as_tensor(index, dtype=torch.long)
        return Trajectories(
            states=self.states[idx],
            clean=self.clean[idx],
            z=self.z[idx],
            t=self.t,
            initial=self.initial[idx],
            sigma=self.sigma,
            noise_mask=None if self.noise_mask is None else self.noise_mask[idx],
        )

    def concat(self, other: "Trajectories") -> "Trajectories":
        """``self`` followed by ``other``, for batches on the same time grid.

        The two must share ``t``: one batch carries a single time vector, so
        concatenating curves observed at different times is not a bigger batch
        but a malformed one.  The noise masks are concatenated only when both
        sides record one -- a legacy batch with no mask makes the result
        maskless rather than silently inventing ``True`` everywhere.
        """
        if self.t.shape != other.t.shape or not bool(torch.equal(self.t, other.t)):
            raise ValueError(
                "concat needs one time grid: "
                f"{tuple(self.t.shape)} vs {tuple(other.t.shape)}"
            )
        if self.states.shape[2:] != other.states.shape[2:]:
            raise ValueError(
                "concat needs matching observation shapes: "
                f"{tuple(self.states.shape[2:])} vs "
                f"{tuple(other.states.shape[2:])}"
            )
        mask = None
        if self.noise_mask is not None and other.noise_mask is not None:
            mask = torch.cat([self.noise_mask, other.noise_mask], dim=0)
        return Trajectories(
            states=torch.cat([self.states, other.states], dim=0),
            clean=torch.cat([self.clean, other.clean], dim=0),
            z=torch.cat([self.z, other.z], dim=0),
            t=self.t,
            initial=torch.cat([self.initial, other.initial], dim=0),
            sigma=self.sigma,
            noise_mask=mask,
        )

    def to(self, device) -> "Trajectories":
        return Trajectories(
            states=self.states.to(device),
            clean=self.clean.to(device),
            z=self.z.to(device),
            t=self.t.to(device),
            initial=self.initial.to(device),
            sigma=self.sigma,
            noise_mask=(
                None if self.noise_mask is None else self.noise_mask.to(device)
            ),
        )


def generate(
    spec: PDESpec,
    config: CaseConfig,
    n_curves: int,
    seed: int,
    *,
    theta: Optional[Theta] = None,
    f=TRUTH,
    sigma: Optional[float] = None,
    inputs: Optional[InputSpec] = None,
    noise_at_initial: bool = False,
    with_noise: bool = True,
    noise_seed: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
    solve_device=None,
    method: str = "rk4",
) -> Trajectories:
    """Generate ``n_curves`` trajectories with deterministic per-curve seeds.

    ``f`` defaults to :data:`TRUTH`, i.e. the spec's own unknown mechanism, so
    that the reference data really solve the stated PDE.  Pass ``f=None`` for
    the ``F = 0`` ablation, or a callable to solve a different equation.

    ``seed`` drives everything: the initial conditions and inputs (the
    *design*) and, unless ``noise_seed`` says otherwise, the observation noise.
    Supplying ``noise_seed`` redraws only the noise on the same design, which is
    the conditional distribution a standard error describes and the design's
    complement (drawing designs with the noise held fixed) completes the
    decomposition.  Without this the two cannot be varied independently, and
    callers had to re-implement the noise loop to get at either -- see the
    variance decomposition in ``docs/size.md`` Sec. 2.

    ``device`` is where the *returned* trajectories live.  The reference solve
    itself runs on ``solve_device``, which defaults to the **CPU** -- see the
    note below.  Pass ``solve_device=device`` to force it onto the accelerator.

    ``method`` is the reference solver's scheme.  ``"rk4"`` is the default and
    is what the published numbers use; it makes the reference trajectory
    accurate to about ``1e-9`` at ``ode_step = 2e-2`` on the forced-Burgers case.
    ``"euler"`` is there for one specific experiment -- reproducing the step the
    *estimator's* rollout uses, so that the discretisation mismatch between data
    and model is removed and its contribution to the estimate can be measured
    instead of argued about.
    """
    theta = spec.truth() if theta is None else theta
    theta = {
        k: torch.as_tensor(v, dtype=dtype, device=device) for k, v in theta.items()
    }
    if f is TRUTH:
        f = spec.truth_f
    sigma = config.sigma if sigma is None else float(sigma)
    inputs = config.inputs if inputs is None else inputs
    grid = config.grid

    # Draw every initial condition and input *first* (a cheap numpy loop, one
    # seeded RNG each), then integrate the whole batch in a single call.  The
    # solutions are numerically identical to solving curve by curve -- the drift
    # is pointwise in the leading axis and ``Z`` is carried per curve -- but the
    # reference solve is by far the most expensive part of a run, and doing it
    # per curve multiplies the number of Python-level steps and kernel launches
    # by ``n_curves``.
    #
    # The solve then runs on the **CPU**, whatever ``device`` says.  This is a
    # measurement, not a preference: the integration is a Python loop over
    # substeps, each one a handful of operations on a small tensor, so the
    # accelerator sees only kernel launches.  On the shared host this was
    # measured on, 48 curves at ``n_x = 32``, ``n_t = 25``, ``data_step = 1e-3``
    # take **2.7 s on the CPU and 177 s on the GPU** -- a factor of 64, entirely
    # launch latency.  Move the batch to ``device`` afterwards; that transfer is
    # one contiguous copy of the whole trajectory tensor.
    inits, zvals = [], []
    for i in range(n_curves):
        rng = np.random.default_rng(curve_seed(seed, i + 1))
        # ``spec.initial`` returns ``(1, d_y, *grid)``; drop the leading axis so
        # that stacking gives the ``(n, d_y, *grid)`` batch ``solve`` expects.
        inits.append(torch.as_tensor(spec.initial(rng, 1), dtype=dtype)[0])
        zvals.append(_draw_z(rng, inputs))

    if n_curves == 0:
        return Trajectories(
            states=torch.empty(0, config.time.n_obs, spec.d_y, *grid.shape,
                               dtype=dtype, device=device),
            clean=torch.empty(0, config.time.n_obs, spec.d_y, *grid.shape,
                              dtype=dtype, device=device),
            z=torch.empty(0, dtype=dtype, device=device),
            t=torch.as_tensor(config.time.points(), dtype=dtype, device=device),
            initial=torch.empty(0, spec.d_y, *grid.shape, dtype=dtype, device=device),
            sigma=float(sigma),
            noise_mask=None,
        )

    # ``theta`` was lifted onto ``device`` above, but the solve runs on
    # ``solve_device``; keep the two consistent so the drift does not mix them.
    solve_device = torch.device("cpu") if solve_device is None else solve_device
    theta = {k: v.to(solve_device) for k, v in theta.items()}
    init_t = torch.stack(inits, dim=0).to(solve_device)  # (n, d_y, *grid)
    z_t = torch.tensor(zvals, dtype=dtype, device=solve_device)
    clean_t = solve(
        spec,
        init_t,
        config.time,
        theta,
        z=z_t,
        f=f,
        ode_step=config.data_step,
        method=method,
    )

    states = clean_t.clone()
    # The exact slice is not an observation of the noisy process, so it must be
    # excluded when the residual variance is estimated.  Record where the noise
    # actually landed rather than leaving sigma_hat to average over it.
    noise_mask = None
    if with_noise and sigma > 0:
        noise_mask = torch.ones_like(clean_t, dtype=torch.bool)
        if not noise_at_initial:
            noise_mask[:, 0] = False
        noise_root = seed if noise_seed is None else noise_seed
        for i in range(n_curves):
            rng = np.random.default_rng(curve_seed(noise_root, i + 1) ^ 0x5EED)
            eps = rng.normal(0.0, sigma, size=tuple(clean_t[i].shape))
            if not noise_at_initial:
                eps[0] = 0.0
            states[i] += torch.as_tensor(eps, dtype=dtype, device=clean_t.device)

    t = torch.as_tensor(config.time.points(), dtype=dtype, device=solve_device)
    traj = Trajectories(
        states=states, clean=clean_t, z=z_t, t=t, initial=init_t,
        sigma=float(sigma), noise_mask=noise_mask,
    )
    # The solve is over; hand the caller the whole batch on ``device`` in one
    # contiguous copy rather than one transfer per curve.
    return traj.to(device) if device is not None else traj


def _draw_z(rng: np.random.Generator, inputs: Optional[InputSpec]) -> float:
    if inputs is None:
        return 0.0
    values = np.asarray(inputs.values, dtype=np.float64)
    probs = np.asarray(inputs.probs, dtype=np.float64)
    return float(rng.choice(values, p=probs))


# ---------------------------------------------------------------------------
# splitting and caching
# ---------------------------------------------------------------------------
@dataclass
class Split:
    train: Trajectories
    val: Trajectories
    test: Trajectories


def split_trajectories(
    traj: Trajectories,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> Split:
    """Split by trajectory, in a fixed order (not shuffled by a DataLoader)."""
    total = n_train + n_val + n_test
    if total > len(traj):
        raise ValueError(f"asked for {total} curves but only {len(traj)} exist")
    order = np.random.default_rng(curve_seed(seed, 99)).permutation(len(traj))
    idx = order[:total]
    train = traj.subset(idx[:n_train])
    val = traj.subset(idx[n_train : n_train + n_val])
    test = traj.subset(idx[n_train + n_val :])
    return Split(train=train, val=val, test=test)


def save_npz(traj: Trajectories, path: str) -> None:
    np.savez_compressed(
        path,
        states=traj.states.numpy(),
        clean=traj.clean.numpy(),
        z=traj.z.numpy(),
        t=traj.t.numpy(),
        initial=traj.initial.numpy(),
        sigma=np.asarray(traj.sigma),
        noise_mask=np.asarray(traj.noise_mask.numpy() if traj.noise_mask is not None
                              else np.zeros(0, dtype=bool)),
    )


def load_npz(path: str, device=None) -> Trajectories:
    with np.load(path) as data:
        mask = data["noise_mask"] if "noise_mask" in data.files else None
        traj = Trajectories(
            states=torch.as_tensor(data["states"]),
            clean=torch.as_tensor(data["clean"]),
            z=torch.as_tensor(data["z"]),
            t=torch.as_tensor(data["t"]),
            initial=torch.as_tensor(data["initial"]),
            sigma=float(data["sigma"]),
            noise_mask=(torch.as_tensor(mask) if mask is not None and mask.size
                        else None),
        )
    return traj.to(device) if device is not None else traj
