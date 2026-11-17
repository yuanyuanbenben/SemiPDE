"""Algorithm 2: efficient covariance and Wald intervals for ``gamma^T theta_0``.

The estimator ``(theta_hat, f_hat)`` is efficient when the nuisance direction is
orthogonalised out, and the efficient covariance is

.. math::

    \\Sigma_{\\mathrm{eff}} = \\sigma^2 \\{E[M_0^\\top M_0]\\}^{-1},
    \\qquad
    M_0(X) = \\partial_\\theta u - \\partial_f u[h_0],

with ``h_0`` the least-squares representer of the parameter direction in the
nuisance tangent space.  Algorithm 2 estimates ``M_0`` by finite differences and
a *matched* network perturbation:

1. ``h_j`` is trained so that ``u(theta_hat, f_hat + delta h_j)`` reproduces
   ``u(theta_hat + delta e_j, f_hat)`` in ``L_2(P)``;
2. ``M_hat_ij = delta^{-1} {u(theta_hat + delta e_j, f_hat)
   - u(theta_hat, f_hat + delta h_j_hat)}(X_i)`` is the *leftover* mismatch --
   zero when the two perturbations act identically on the data, which is exactly
   the orthogonality the construction is after.

Then

.. math::

    \\hat\\Sigma_{\\mathrm{eff}} = \\hat\\sigma^2
      \\Big\\{\\frac1n\\sum_i \\widehat M_i^\\top \\widehat M_i\\Big\\}^{-1},
    \\qquad
    \\hat\\sigma^2 = \\frac{1}{n d_y}\\sum_i \\|Y_i - \\tilde u(X_i)\\Vert^2,

and the interval is
``gamma^T theta_hat +/- n_est^{-1/2} z_{1-alpha/2} sqrt(gamma^T Sigma gamma)``.

Note the two sample sizes: the average ``n^{-1} sum M_i^T M_i`` estimates the
*per-observation* matrix ``E[M_0^T M_0]`` from the inference data (``n_inf``),
while the width is driven by ``n_est``, the size of the sample the estimator was
fit on.  They coincide in the paper's experiments; keeping them separate is what
makes the ``n_est`` versus ``n_inf`` ablation in the main text meaningful.

Numerical scaling
-----------------
Step 1 asks for a network whose *effect on the drift* is ``delta`` times its
output.  That factor has to be carried explicitly
(:attr:`~semipde.model.ProfilingModel.mechanism_scale`), and it is what makes
the objective well posed: without it the network is being asked to emit a
correction of size ``delta * d_theta u``, which for ``delta = 1e-5`` is around
``1e-6``.  Every optimiser then overshoots permanently -- measured on Case 1 at
``n_x = 16``, no learning rate from ``1e-6`` to ``1e-2`` reduced the mismatch at
all (``0%`` of the ``theta``-effect reproduced), because the drift perturbation
the optimiser could see was ``1e5`` times smaller than the one it needed.

``delta`` therefore has two roles that must not be confused: it is the step of
the finite difference *and* the scale of the drift perturbation.  The ``delta``
only re-enters the estimate once, at ``M_hat``.

How much of the perturbation the inner fit can actually recover
--------------------------------------------------------------
With the scaling fixed the inner problem has a **representability floor**, and
knowing where it is matters for reading the results, because the floor comes
from the paper's own choice of class for ``h_j``, not from the optimiser.

``h_j`` is a function of the *state*, while the target
``dG_z/dtheta = lap(u)`` is a function of the *coordinate*.  Computed directly
-- binning ``(u_i, lap(u)_i)`` over every observed point of every curve, on
Case 1 at ``n_x = 32`` -- a *flexible* function of ``u`` (a 200-bin conditional
mean) explains only about ``15%`` of the source variance; the best cubic
explains ``14.8%``.  To first order the source is not a function of the state at
all: it is dominated by the high spatial frequencies the Laplacian amplifies and
that no ``h(u)`` can produce.

The floor is now available exactly rather than by extrapolation.
``M_hat(h) = d - L[h]`` is *affine* in ``h`` -- ``L[h] = delta^{-1}{u(theta_hat,
f_hat + delta h) - u(theta_hat, f_hat)}`` is linear, since ``h`` enters the
drift additively -- so with ``inner_solve="exact"`` the inner problem is solved
by :func:`numpy.linalg.lstsq` and the reported mismatch *is* the class's floor,
to machine precision and with no seed dependence.  Each basis column costs one
perturbed rollout; there is no learning rate, no growth phase and no epoch
budget, and ``epochs``/``lr``/``inner_hidden``/``linear_term`` are all ignored.
The basis is :class:`ChebyshevBasis`, whose ``columns`` select what ``h_j``
reads (``spec.inner_features``, defaulting to ``spec.features``), and whose
``degree`` is the total degree ``basis_degree``.

The ``network`` path is what the published numbers were produced with, and it
approaches the same floor slowly.  The measured curve on Case 1 at ``n_x = 16``,
one ``h_j``, full batch, ``lr = 1e-2`` (mismatch rms, and the shrinkage against
the untrained baseline of ``1.310``)::

    ep   5    10    25    50    100   200   400
         1.507 1.410 1.316 1.237 1.184 1.147 1.123
         0.87x 0.93x 1.00x 1.06x 1.11x 1.14x 1.17x

Two things to read off it.  First, the first few epochs make the mismatch
*worse* than doing nothing (``0.87x`` at five epochs): the network starts at zero
and has to grow into the target, and until it is roughly the right size the
correction it applies costs more than it saves.  A budget below about 25 epochs
therefore returns something worse than the plain finite difference, which is why
``epochs`` defaults to ``40`` and not to something smaller.  Second, the decay is
slow and roughly linear in ``log`` epochs -- there is no knee, so on this path
the reported rms is a *budget-truncated* upper bound on the floor and the number
moves with ``epochs``.

The learning rate between ``3e-3`` and ``1e-2`` changes none of these numbers,
and adding a pointwise linear term to the inner network changes none of them
either -- which is the evidence that the network path is limited by its *class*
and not by its optimiser, and hence that its long tail is the approach to the
floor rather than further fitting gain.  ``inner_solve="exact"`` removes the
question: it attains the floor directly, so a comparison between two inner
classes there is a statement about the classes and cannot be confounded with
either the budget or the initialisation.  In the trajectory metric the floor is
higher than the ``15%`` above, because the PDE smooths exactly the frequencies
that the drift metric weights at full strength; see the next section.

The practical consequence depends on which path is used.  Under ``network`` the
inner fit is bounded by epochs rather than by convergence, and for a study that
needs many repetitions the lever is the size of the **inference** sample, not
the number of epochs.  Under ``exact`` the cost is one rollout per basis column
per parameter, independent of the inference sample.  Either way ``n_est`` and
the sample the inner fit sees are separate arguments -- pass a reduced
:class:`~semipde.data.Trajectories` and keep ``n_est`` at the full size.

Why the objective is on trajectories and not on the drift
---------------------------------------------------------
It is tempting to replace step 1 with a plain regression of ``h_j`` onto the
drift source ``dG_z/dtheta_j``, which needs no ODE at all.  The two are **not**
equivalent, and the regression is much worse.  On Case 1 at ``n_x = 16`` the
drift source has rms ``14.3`` while the perturbation it induces in the state has
rms ``1.3e-5 / delta`` -- the equation smooths the drift, and it smooths the
high spatial frequencies (which is where a Laplacian-like source lives) far more
than the low ones.  A network reading ``u`` pointwise explains about ``15%`` of
the *source* variance but about ``23%`` of the *trajectory* perturbation, because
minimising in the drift metric weights the unlearnable high frequencies at full
strength.  Measured directly: fitting the source and then scoring in the
trajectory metric made the mismatch *worse* (``5.22`` against an initial
``1.31``).  The rollout is not an inconvenience to be optimised away; it is the
correct inner product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import CaseConfig
from .data import Trajectories
from .model import AugmentedMechanism, ProfilingModel, build_network

__all__ = [
    "InferenceResult",
    "parameter_index",
    "estimate_covariance",
    "normal_quantile",
]


def normal_quantile(p: float) -> float:
    """``z_p`` without a scipy dependency (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


# ---------------------------------------------------------------------------
# parameter bookkeeping
# ---------------------------------------------------------------------------
def parameter_index(model: ProfilingModel) -> List[Tuple[str, int, Tuple[int, ...]]]:
    """Flatten ``theta`` into ``p`` coordinate directions.

    Returns one ``(name, flat_offset, shape)`` entry per parameter tensor.
    """
    out, offset = [], 0
    for name, p in model.theta.items():
        out.append((name, offset, tuple(p.shape)))
        offset += p.numel()
    return out


def _flatten_theta(model: ProfilingModel) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in model.theta.items()])


def parameter_names(model: ProfilingModel) -> List[str]:
    """Column labels of ``theta``, one per scalar, in ``theta_detached`` order.

    Multi-parameter cases index their entries (``theta[0]``, ``theta[1]``) so a
    contrast vector can address them; a scalar keeps its bare name, which is
    what the paper's single-parameter tables print.
    """
    return [
        f"{name}" if p_t.numel() == 1 else f"{name}[{k}]"
        for name, p_t in model.theta.items()
        for k in range(p_t.numel())
    ]


def _num_parameters(model: ProfilingModel) -> int:
    return sum(p.numel() for p in model.theta.parameters())


@torch.no_grad()
def _perturb_theta(model: ProfilingModel, flat: torch.Tensor) -> None:
    offset = 0
    for _, p in model.theta.items():
        n = p.numel()
        p.copy_(flat[offset : offset + n].reshape(p.shape))
        offset += n


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
class WaldIntervals:
    """``(theta, cov)`` turned into Wald intervals for ``gamma^T theta_0``.

    Shared by Algorithm 2 and by the refit-based estimators in
    :mod:`semipde.resample`.  The two reach ``cov`` by completely different
    routes -- a plug-in score versus the sampling spread of the estimator
    itself -- but once they do, the arithmetic that turns a covariance into an
    interval must not be allowed to differ between them, or comparing the two
    standard errors would compare the comparisons as much as the estimators.

    Subclasses provide ``names``, ``theta``, ``cov`` and ``n_est``.
    """

    names: List[str]
    theta: np.ndarray
    cov: np.ndarray
    n_est: int

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:  # pragma: no cover - caller error
            raise KeyError(f"no parameter named {name!r} in {self.names}") from exc

    def contrast(self, gamma: Optional[Sequence[float]] = None) -> np.ndarray:
        """``gamma`` as a coefficient vector; defaults to each coordinate."""
        if gamma is None:
            return np.eye(len(self.names))
        g = np.asarray(gamma, dtype=np.float64)
        if g.shape != (len(self.names),):
            raise ValueError(
                f"gamma must have length {len(self.names)}, got {g.shape}"
            )
        return g[None, :] if g.ndim == 1 else g

    def interval(
        self,
        alpha: float = 0.05,
        gamma: Optional[Sequence[float]] = None,
        n: Optional[int] = None,
    ) -> Dict[str, object]:
        """Wald intervals for ``gamma^T theta_0``.

        ``n`` overrides ``n_est``; passing ``n_inf`` gives the "wrong" interval
        that the paper's ablation contrasts against.
        """
        g = self.contrast(gamma)
        n = self.n_est if n is None else int(n)
        z = normal_quantile(1.0 - alpha / 2.0)
        point = g @ self.theta
        var = np.einsum("ij,jk,ik->i", g, self.cov, g)
        half = z * np.sqrt(var) / math.sqrt(n)
        return {
            "alpha": float(alpha),
            "n": int(n),
            "z": float(z),
            "gamma": g,
            "estimate": point,
            "se": np.sqrt(var) / math.sqrt(n),
            "half_width": half,
            "lower": point - half,
            "upper": point + half,
        }

    def covers(
        self,
        truth: Sequence[float],
        alpha: float = 0.05,
        gamma: Optional[Sequence[float]] = None,
        n: Optional[int] = None,
    ) -> np.ndarray:
        """Boolean array: does each interval contain ``gamma^T theta_0``?"""
        g = self.contrast(gamma)
        target = g @ np.asarray(truth, dtype=np.float64)
        iv = self.interval(alpha=alpha, gamma=gamma, n=n)
        return (iv["lower"] <= target) & (target <= iv["upper"])


@dataclass
class InferenceResult(WaldIntervals):
    """``Sigma_eff`` plus everything needed to form and audit an interval."""

    names: List[str]
    theta: np.ndarray                 # (p,) point estimate
    cov: np.ndarray                   # (p, p) Sigma_eff
    sigma: float                      # sigma_hat
    m_hat: np.ndarray                 # (n_inf, p) influence proxy
    n_inf: int
    n_est: int
    delta: float
    # Not all floats: the ridge ladder is stored as a list per
    # coordinate, so the record can be read back and re-summarised
    # without re-running the fit that produced it.
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------
def _mechanism_only(model: ProfilingModel) -> nn.Module:
    """The fitted mechanism as a bare pointwise module (no penalty attached)."""
    return model.network


def _multi_indices(in_dim: int, degree: int) -> List[Tuple[int, ...]]:
    """Multi-indices with ``1 <= |alpha| <= degree``, in graded order.

    The constant is excluded, matching the ``h_j`` classes the probes measure:
    ``f_hat`` is only identified up to what ``theta`` can absorb, so a constant
    direction is not a well-posed part of the nuisance tangent.
    """
    out: List[Tuple[int, ...]] = []
    for total in range(1, degree + 1):
        stack: List[Tuple[Tuple[int, ...], int, int]] = [((), total, in_dim)]
        while stack:
            prefix, left, slots = stack.pop()
            if slots == 1:
                out.append(prefix + (left,))
                continue
            for value in range(left + 1):
                stack.append((prefix + (value,), left - value, slots - 1))
    return out


def _chebyshev_table(t: torch.Tensor, degree: int) -> torch.Tensor:
    """``T_0..T_degree`` at ``t``: ``(..., in_dim) -> (..., in_dim, degree+1)``."""
    terms = [torch.ones_like(t)]
    if degree >= 1:
        terms.append(t)
    for _ in range(2, degree + 1):
        terms.append(2.0 * t * terms[-1] - terms[-2])
    return torch.stack(terms, dim=-1)


class ChebyshevBasis(nn.Module):
    """A pointwise ``h`` that is **linear** in its coefficients.

    ``h(features) = sum_k coef_k * T_{alpha_k}(features)`` over the tensor
    Chebyshev multi-indices with ``1 <= |alpha| <= degree``, evaluated on the
    features scaled to ``[-1, 1]`` by the window observed on the first rollout.

    That linearity is the whole point.  Algorithm 2's inner problem is

        min_h (1/n) sum_i ||d_i - L_i[h]||^2 + lambda_n J(h),   L_i[h] = d_f u_i[h],

    which is *linear* least squares in any basis, so the exact minimiser is one
    ``lstsq`` away with one perturbed PDE solve per basis column.  No optimiser,
    no learning rate, no growth phase, and -- unlike the network -- no gap
    between the fit and the class's own representability floor: the recorded
    ``mismatch_rms_j`` *is* that floor.

    ``J`` is the empirical ``L^2`` norm of ``h``, accumulated over the (x, t)
    points the drift is evaluated at during the column rollouts.  A coefficient
    ``L^2`` penalty would not be parametrisation invariant -- rescaling one
    column would change the answer -- so the ridge uses the accumulated Gram
    instead (``penalty()`` is kept as the coefficient norm only so that a
    ``ProfilingModel`` wrapping this module stays usable by
    :func:`_train_perturbation`).

    ``columns`` restricts the tensor product to a subset of the feature tensor's
    columns, with the case's own convention that the state leads and the
    coordinate follows.  It is how the class ladder is expressed in one object
    -- ``[0]`` is "a function of the state", ``[1]`` "of the coordinate", and
    ``[0, 1]`` the tensor product of the two -- which is what makes the comparison
    between them a comparison of *classes* and nothing else: same solve, same
    window rule, same code path.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        degree: int = 8,
        margin: float = 0.1,
        columns: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.degree = int(degree)
        # The window is measured on the baseline trajectory, and the column
        # rollouts then move the state slightly outside it -- a little is enough
        # to matter, because ``T_k`` grows like ``(2|x|)^k``.  The margin buys
        # room and the clamp in ``_scaled`` makes the blow-up impossible rather
        # than merely unlikely.  Neither touches ``coef``, so ``h`` is still
        # exactly linear in it, which is the only property the solve needs.
        self.margin = float(margin)
        self.columns = (tuple(range(self.in_dim)) if columns is None
                        else tuple(int(c) % self.in_dim for c in columns))
        dim = len(self.columns)
        self.alphas = _multi_indices(dim, self.degree)
        self.register_buffer(
            "lo", torch.full((dim,), float("inf"), dtype=dtype, device=device)
        )
        self.register_buffer(
            "hi", torch.full((dim,), float("-inf"), dtype=dtype, device=device)
        )
        self.coef = nn.Parameter(
            torch.zeros(len(self.alphas), self.out_dim, dtype=dtype, device=device)
        )
        # The window is fixed after the first rollout: it is not a parameter, and
        # letting each column's rollout move it would make ``L`` depend on the
        # order the columns happen to be built in.
        self.window_fixed = False
        self.accumulate = False
        self._gram = torch.zeros(
            len(self.alphas), len(self.alphas), dtype=dtype, device=device
        )
        self._gram_n = 0

    @property
    def n_terms(self) -> int:
        return len(self.alphas)

    def _select(self, features: torch.Tensor) -> torch.Tensor:
        """The feature columns the basis is a function of."""
        return features[..., list(self.columns)]

    def observe(self, features: torch.Tensor) -> None:
        """Widen the normalisation window.  A no-op once it is fixed."""
        if self.window_fixed:
            return
        with torch.no_grad():
            selected = self._select(features.detach())
            self.lo.copy_(torch.minimum(self.lo, selected.amin(dim=0)))
            self.hi.copy_(torch.maximum(self.hi, selected.amax(dim=0)))

    def fix_window(self) -> None:
        """Freeze the window, widening a degenerate column to a unit span.

        Called after the baseline rollout, which is what observes the window.
        A column that never varied -- the coordinate columns of a design with no
        observation points, or a feature the drift ignores -- would otherwise
        divide by zero, so its span is set to one.
        """
        with torch.no_grad():
            if not bool(torch.isfinite(self.lo).all() and torch.isfinite(self.hi).all()):
                raise RuntimeError(
                    "the basis window was never observed: no rollout was taken "
                    "before fix_window(), so the Chebyshev normalisation is "
                    "undefined"
                )
            span = self.hi - self.lo
            flat = span.abs() < 1e-12
            # A column that never varied gets a unit span; both then get the
            # margin.
            width = torch.where(flat, torch.ones_like(span), span) * (1.0 + self.margin)
            self.lo.copy_(self.lo - (width - span) / 2.0)
            self.hi.copy_(self.hi + (width - span) / 2.0)
        self.window_fixed = True

    def _scaled(self, features: torch.Tensor) -> torch.Tensor:
        scaled = (features - (self.lo + self.hi) / 2.0) / ((self.hi - self.lo) / 2.0)
        return scaled.clamp(-1.0, 1.0)

    def basis(self, features: torch.Tensor) -> torch.Tensor:
        """The basis matrix ``(n_points, n_terms)`` at ``features``."""
        table = _chebyshev_table(self._scaled(self._select(features)), self.degree)
        columns = []
        for alpha in self.alphas:
            column = table[..., 0, alpha[0]]
            for dim in range(1, len(self.columns)):
                column = column * table[..., dim, alpha[dim]]
            columns.append(column)
        return torch.stack(columns, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.window_fixed:
            # The baseline rollout is also the pass that measures the window,
            # so the two cannot be separated -- but they must not interfere.
            # The output here is returned as literal zero instead of
            # ``basis(features) @ coef``: ``coef`` starts at zero, so the two
            # agree once the window exists, and literal zero keeps the first
            # evaluations -- made before the window has any finite extent --
            # from pushing a nan into the integrated trajectory.
            self.observe(features)
            return torch.zeros(
                features.shape[:-1] + (self.out_dim,),
                dtype=features.dtype,
                device=features.device,
            )
        values = self.basis(features)
        if self.accumulate:
            with torch.no_grad():
                self._gram += values.detach().T @ values.detach()
                self._gram_n += int(values.shape[0])
        return values @ self.coef

    def penalty(self) -> torch.Tensor:
        return self.coef.pow(2).sum()

    def empirical_gram(self) -> Optional[np.ndarray]:
        """``E[T_alpha T_beta]`` over every point the drift was evaluated at."""
        if self._gram_n == 0:
            return None
        return (self._gram / self._gram_n).detach().cpu().numpy().astype(np.float64)

    def reset_accumulator(self) -> None:
        self._gram.zero_()
        self._gram_n = 0


def _inner_design(
    surrogate: ProfilingModel,
    data: Trajectories,
    target: torch.Tensor,
    *,
    delta: float,
    index: int = 0,
    diag: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(L, d, initial_rms)`` for one Algorithm-2 column, over a linear basis.

    ``surrogate`` must be a :class:`~semipde.model.ProfilingModel` whose network
    is a :class:`ChebyshevBasis`; ``L`` is then one perturbed PDE solve per basis
    column -- ``L[:, k * out_dim + r] = delta^{-1}`` times the change in the
    trajectory when coefficient ``(k, r)`` alone is one -- and ``d`` is the plain
    finite difference in ``theta``, divided by ``delta``.

    The exact solve is ``lstsq(L, d)``; the probes want ``L`` itself (its
    conditioning, and for the class ladder the frontier of what the class can
    reach).  One builder serves both, so a probe cannot measure a different
    object from the one the estimator minimises over.

    The baseline rollout is what observes the normalisation window, and it is
    also the "do nothing" reference; the returned ``initial_rms`` is the same
    quantity the network path records as ``mismatch_rms_init``.
    """
    basis: ChebyshevBasis = surrogate.network
    with torch.no_grad():
        # The untrained surrogate is exactly the finite-difference baseline: the
        # basis carries no constant term and every coefficient starts at zero, so
        # the first rollout already is ``u(theta_hat, f_hat)``.
        base = surrogate(data.initial, data.t, data.z).reshape(-1)
        basis.fix_window()
        basis.reset_accumulator()
        basis.accumulate = True
        columns = []
        for k in range(basis.n_terms):
            for r in range(basis.out_dim):
                basis.coef.zero_()
                basis.coef[k, r] = 1.0
                got = surrogate(data.initial, data.t, data.z).reshape(-1)
                columns.append(
                    ((got - base) / delta).cpu().numpy().astype(np.float64)
                )
        basis.accumulate = False
        basis.coef.zero_()
    L = np.stack(columns, axis=1)
    d = ((target.reshape(-1) - base) / delta).cpu().numpy().astype(np.float64)
    if diag is not None:
        singular = np.linalg.svd(L, compute_uv=False)
        diag[f"inner_columns_{index}"] = float(basis.n_terms * basis.out_dim)
        diag[f"inner_terms_{index}"] = float(basis.n_terms)
        diag[f"inner_degree_{index}"] = float(basis.degree)
        diag[f"cond_L_{index}"] = float(singular[0] / max(singular[-1], 1e-300))
    return L, d, float(np.sqrt(np.mean(d ** 2)))


def _ridge_root(basis: ChebyshevBasis) -> np.ndarray:
    """The symmetric square root of the basis's empirical Gram.

    The ridge is written in the *empirical* ``L^2`` inner product rather than in
    coefficient coordinates, so that a change of basis leaves the fit unchanged;
    the square root is what turns that into a penalty on the same scale as ``L``.
    """
    gram = basis.empirical_gram()
    if gram is None:
        raise RuntimeError(
            "lambda_n > 0 needs the inner Gram, but no drift evaluation was "
            "recorded; the basis was not attached to the model"
        )
    gram = 0.5 * (gram + gram.T)
    weights, vectors = np.linalg.eigh(gram)
    return np.diag(np.sqrt(np.clip(weights, 0.0, None))) @ vectors.T


def _ridge_coef(L: np.ndarray, d: np.ndarray, root: Optional[np.ndarray],
                out_dim: int, lambda_n: float) -> np.ndarray:
    """The exact solve's coefficients at one ridge strength ``lambda_n``.

    ``root`` is :func:`_ridge_root`; ``None`` marks the unregularised solve, and
    is the same computation with the penalty block left off.
    """
    if lambda_n > 0.0 and root is not None:
        ridge = np.sqrt(d.size * float(lambda_n)) * np.kron(root, np.eye(out_dim))
        coef, *_ = np.linalg.lstsq(
            np.vstack([L, ridge]), np.concatenate([d, np.zeros(ridge.shape[0])]),
            rcond=None,
        )
        return coef
    coef, *_ = np.linalg.lstsq(L, d, rcond=None)
    return coef


def _exact_minimiser_from(
    L: np.ndarray,
    d: np.ndarray,
    basis: ChebyshevBasis,
    lambda_n: float,
    *,
    index: int,
    diag: Dict[str, float],
) -> np.ndarray:
    """The exact least-squares ``M_hat`` from an already-built design.

    Split out from :func:`_exact_minimiser` so that the ridge -- which is a
    post-processing of ``(L, d)`` and never a property of them -- can be swept at
    the *same* design.  Building ``L`` costs one perturbed PDE solve per basis
    column; solving for a coefficient vector costs a small dense solve.  Keeping
    the two apart is what makes a ridge ladder affordable: without it, every
    ``lambda`` in a ladder re-runs every rollout and reproduces ``L`` and ``d``
    bit for bit.
    """
    root = _ridge_root(basis) if lambda_n > 0.0 else None
    coef = _ridge_coef(L, d, root, basis.out_dim, lambda_n)
    diag[f"coef_rms_{index}"] = float(np.sqrt(np.mean(coef ** 2)))
    diag[f"inner_exact_{index}"] = 1.0
    return d - L @ coef


def _exact_minimiser(
    surrogate: ProfilingModel,
    data: Trajectories,
    target: torch.Tensor,
    *,
    delta: float,
    lambda_n: float,
    index: int,
    diag: Dict[str, float],
) -> Tuple[np.ndarray, float]:
    """The exact least-squares ``M_hat`` for one column, over a linear basis.

    ``L`` is built one column at a time -- one perturbed PDE solve each -- and
    the minimiser is read off with ``lstsq``.  The ridge, when ``lambda_n > 0``,
    is the empirical ``L^2`` Gram of the basis rather than a coefficient norm,
    so a change of basis leaves the fit unchanged.

    Returns the mismatch and the untrained baseline's rms, which is the same
    "do nothing" reference the network path records as ``mismatch_rms_init``.
    """
    basis: ChebyshevBasis = surrogate.network
    L, d, initial_rms = _inner_design(
        surrogate, data, target, delta=delta, index=index, diag=diag
    )
    # The unregularised residual is the class's representability floor, and under
    # ``lambda_n = 0`` it is also what ``mismatch_rms_j`` reports.  Record it
    # separately so that a ridge cannot be mistaken for extra representability.
    diag[f"inner_floor_rms_{index}"] = float(
        np.sqrt(np.mean((d - L @ _ridge_coef(L, d, None, basis.out_dim, 0.0)) ** 2))
    )
    m_hat = _exact_minimiser_from(L, d, basis, lambda_n, index=index, diag=diag)
    return m_hat, initial_rms


def estimate_covariance(
    model: ProfilingModel,
    data: Trajectories,
    *,
    delta: float = 1e-5,
    lambda_n: float = 0.0,
    epochs: int = 40,
    inner_steps: int = 5,
    lr: float = 1e-2,
    batch_size: Optional[int] = None,
    n_est: Optional[int] = None,
    linear_term: bool = True,
    inner_hidden: Optional[Sequence[int]] = None,
    inner_solve: str = "network",
    basis_degree: int = 8,
    inner_features: Optional[Callable] = None,
    inner_feature_dim: Optional[int] = None,
    cov_estimator: str = "efficient",
    lambda_grid: Optional[Sequence[float]] = None,
    seed: int = 0,
    verbose: bool = False,
) -> InferenceResult:
    """Run Algorithm 2 on the fitted ``model`` and the inference data.

    ``model`` is left at the fitted iterate; ``theta`` is not modified (the
    perturbations are applied to a copy and restored).  ``lambda_n`` is the
    ``tilde lambda_n`` of the algorithm, penalising ``J(h_j)`` only.

    ``n_est`` defaults to the number of scalars in ``data`` (``n_inf``); pass the
    estimation sample size explicitly to separate the two, as the paper's
    ablation does.

    ``inner_hidden`` sizes the inner ``h_j`` network, independently of the
    estimator's own ``config.hidden`` (see
    :func:`~semipde.model.build_network`).  It is the one knob that changes the
    *class* the inner problem minimises over, which is what the module
    docstring's representability floor is about -- the budget (``epochs``),
    learning rate and penalty have all been swept and none of them moves ``se``
    materially.  Left ``None`` the inner net matches the estimator, which is
    what the paper's ``S_n`` prescribes and what every published number used.

    ``inner_features`` (with ``inner_feature_dim``) gives ``h_j`` an input set
    *other* than the one the fitted ``f_hat`` reads, e.g. ``lambda u, coords, z:
    cat([coords, u.movedim(1, -1)], -1)`` for a Case-1 model whose mechanism is
    state-only.  Left ``None`` it is taken from the spec, so a case can set it
    once (``ReactionDiffusion(..., inner_inputs="location")``) and every caller
    picks it up; the arguments here override that and are what the probes use.

    ``inner_solve`` picks how step 1 is solved.  ``"network"`` is the paper's
    gradient descent (:func:`_train_perturbation`); ``"exact"`` replaces the
    inner net with a :class:`ChebyshevBasis` and solves the least-squares
    problem in closed form.  The two differ only in *how* the minimum over the
    inner class is found, never in which class -- with an adequate basis the
    exact solve reaches the class's own floor (``inner_floor_rms_j``), which the
    optimiser approaches only from above and slowly (see the module docstring's
    epoch table, where 400 epochs still leave ``1.123`` against a floor near
    ``1.0``).  It is deterministic, keeps ``se`` free of inner-optimiser noise,
    and costs one rollout per basis column instead of ``epochs`` passes, which
    is what makes a many-repetition size study affordable.

    ``basis_degree`` is the total degree of that basis, in every column of the
    inner feature tensor jointly: at ``inner_feature_dim = 1`` it is the number
    of terms, and at ``2`` the cross terms of the state and the coordinate are
    what a ``location``-widened inner map buys (see
    :class:`ChebyshevBasis`).  ``inner_hidden`` and ``linear_term`` are ignored
    under ``"exact"``: the basis already contains the degree-1 terms, so the
    linear augmentation is a special case of it.

    This is a **deliberate deviation from the paper**, and the deviation is what
    makes it work.  ``h_0`` is not a free test direction: it is
    ``d_theta f_hat(theta_0)``, the sensitivity of the fitted nuisance to
    ``theta``, which is why ``psi = d_theta l - d_f l[h_0]`` is the *profile*
    score.  ``S_n`` is the model's own class and fixes ``h_0`` through the
    projection, so as the paper states it the inner class is pinned to the
    estimator's.  Widening only ``h_j`` computes ``M_hat = d_theta u -
    d_f u[h_hat]`` against directions the estimator can never take: ``M_hat``
    falls, ``se`` rises, and the interval *over*-covers.  By how much is set by
    the width of the class and not by the data -- it is a calibration knob with
    no operational meaning, not an estimate.  The justification is strictly
    empirical (it is the only lever that moves ``se/sd`` onto 1; budget,
    learning rate, penalty and inner width do not), and the calibration --
    including its cost in bias -- is recorded in ``docs/size.md``.

    ``linear_term`` augments the inner network with a pointwise linear map on
    the features (see :class:`~semipde.model.AugmentedMechanism`).  On Case 1 it
    is measurably neutral -- the inner problem is at a *representability* floor,
    not an optimisation one (see the module docstring), so enlarging the class
    by its lowest-order element changes nothing.  It is kept on because on the
    coordinate-reading cases (the forced-Burgers mechanism is ``q(x)``) a linear
    term in ``x`` is the leading Fourier mode of the thing being estimated, and
    there the enlargement is not degenerate.

    ``lambda_grid`` asks for the whole ridge *curve* rather than one point on it.
    The ridge enters only through ``M_hat``, and ``M_hat = d - L @ coef(lambda)``
    is a post-processing of the pair ``(L, d)`` that the inner design already
    built -- so sweeping ``lambda`` costs one small dense solve per grid point,
    while re-calling this function per ``lambda`` would re-run every perturbed
    PDE solve and reconstruct ``L`` and ``d`` bit for bit.  That is the whole
    reason a ladder is affordable.  Each grid point carries the *same* sandwich
    as a pointwise call at that ``lambda`` (the efficient middle is shared, the
    robust middle is rebuilt from that grid point's ``M_hat``), so
    ``ridge_se_j[k]`` is what a separate call with ``lambda_n =
    lambda_grid[k]`` returns; ``ridge_rms_j`` and ``ridge_shrink_j`` are the
    matching ``mismatch_rms_j`` / ``mismatch_shrinkage_j``.  Only ``"exact"``
    fills the ladder -- the network path has no closed form to interpolate, and
    the grid would cost another ``epochs`` per point, which the calibration in
    ``docs/size.md`` shows buys nothing.

    ``cov_estimator`` picks the middle matrix of the sandwich.  ``"efficient"``
    is the paper's variance estimator -- ``sigma_hat^2 * gram^{-1}``,
    which replaces ``mean(r_i^2 M_i M_i^T)`` by ``mean(r_i^2) *
    mean(M_i M_i^T)`` and is therefore exact only when the residual variance is
    the *same* at every point.  ``"robust"`` is the Eicker--Huber--White form
    ``gram^{-1} [mean(r_i^2 M_i M_i^T)] gram^{-1}``, which keeps the pointwise
    weights.  The two agree iff the fitted model is correctly specified, so the
    gap between them is a direct read-out of how much of the residual is the
    systematic ``u - (theta_hat, f_hat)`` misfit rather than the noise the
    paper's ``sigma_hat^2`` measures.  See ``docs/size.md``: on Case 1 the gap
    is the leading candidate for the coverage shortfall.
    """
    config: CaseConfig = model.config
    spec = model.spec
    # Checked here rather than where it is used: the rollouts below cost minutes,
    # and a typo in the option name must not be paid for with them.
    if cov_estimator not in ("efficient", "robust"):
        raise ValueError(
            f"cov_estimator must be 'efficient' or 'robust', got {cov_estimator!r}"
        )
    if inner_solve not in ("network", "exact"):
        raise ValueError(
            f"inner_solve must be 'network' or 'exact', got {inner_solve!r}"
        )
    # A ladder that silently records nothing is worse than no ladder: the caller
    # reads ``ridge_se_j`` and finds it missing several minutes later, after the
    # rollouts.  The network solve has no design to reuse, so a grid there would
    # cost ``len(grid)`` inner optimisations and is refused rather than ignored.
    if lambda_grid is not None and inner_solve != "exact":
        raise ValueError(
            "lambda_grid needs inner_solve='exact' (the network solve has no "
            "cached design to sweep); pass a single lambda_n instead"
        )
    theta_hat = _flatten_theta(model)
    p = _num_parameters(model)
    names = parameter_names(model)
    n_inf = int(data.states[0].numel()) * len(data)
    n_est = n_inf if n_est is None else int(n_est)

    # -- target rollouts: u(theta_hat + delta e_j, f_hat) ----------------------
    fitted_mechanism = _mechanism_only(model)
    fitted_mechanism.eval()
    # The fitted model is frozen for the duration and restored on the way out,
    # so that an inference run leaves the estimator exactly as it found it.
    frozen_net = [prm for prm in fitted_mechanism.parameters()]
    frozen_theta = list(model.theta.parameters())
    was_net = [prm.requires_grad for prm in frozen_net]
    was_theta = [prm.requires_grad for prm in frozen_theta]
    for prm in frozen_net + frozen_theta:
        prm.requires_grad_(False)

    targets = []
    with torch.no_grad():
        for j in range(p):
            perturbed = theta_hat.clone()
            perturbed[j] += delta
            _perturb_theta(model, perturbed)
            targets.append(model(data.initial, data.t, data.z))
        _perturb_theta(model, theta_hat)

    # -- the inner class --------------------------------------------------------
    # Nothing given means "whatever the spec says".  ``spec.inner_features`` is
    # the estimator's own map unless the case widened it, in which case ``h_j``
    # reads a strictly larger input set than ``f_hat`` (see
    # ``PDESpec.inner_features``); the explicit argument overrides the spec and
    # is what the probes use.
    if inner_features is None and spec.inner_feature_dim is not None:
        inner_features, inner_feature_dim = (
            spec.inner_features, spec.inner_feature_dim,
        )
    if inner_features is not None and inner_feature_dim is None:
        raise ValueError(
            "inner_features needs inner_feature_dim: the width cannot be "
            "inferred without a sample of the features"
        )

    # -- fit h_j against the target, then read off the mismatch ----------------
    m_hat_blocks = []
    # One column of ``M_hat`` per ridge strength, filled only when a ladder is
    # asked for; the columns are assembled into the *same* sandwich below, so a
    # ridge cannot change how the variance is formed -- only ``M_hat``.
    grid_blocks: List[List[np.ndarray]] = []
    diag: Dict[str, Any] = {}
    for j in range(p):
        h_dim = spec.feature_dim if inner_features is None else inner_feature_dim
        if inner_solve == "exact":
            h_net = ChebyshevBasis(
                h_dim, spec.mechanism_out_dim, degree=basis_degree,
                dtype=model.dtype, device=theta_hat.device,
            )
        else:
            h_net = build_network(
                config, spec, in_dim=h_dim, hidden=inner_hidden, dtype=model.dtype,
                seed=seed + 1000 * j,
            ).to(theta_hat.device)
            if linear_term:
                h_net = AugmentedMechanism(
                    h_net,
                    h_dim,
                    spec.mechanism_out_dim,
                    dtype=model.dtype,
                    device=theta_hat.device,
                )
        # The inner class is the one thing the sweeps in the module docstring
        # could not move, so record its size in the row: a reader comparing two
        # runs has to be able to see whether they used the same sieve.
        diag.setdefault(
            "inner_network_params", float(sum(q.numel() for q in h_net.parameters()))
        )
        # theta frozen at the estimate; drift = G(theta_hat) + f_hat + delta*h_j.
        # ``mechanism_scale`` carries the ``delta``: ``h_j`` is optimised at the
        # network's own scale and only its *effect* is multiplied by the
        # finite-difference step, which is what Algorithm 2 writes.
        surrogate = ProfilingModel(
            spec,
            config,
            theta={k: v.detach().clone() for k, v in model.theta.items()},
            network=h_net,
            base_mechanism=fitted_mechanism,
            mechanism_scale=delta,
            dtype=model.dtype,
            feature_map=inner_features,
        ).to(theta_hat.device)
        for prm in surrogate.theta.parameters():
            prm.requires_grad_(False)

        # The mismatch before any training is what "do nothing" would give --
        # i.e. the plain finite-difference derivative.  Recording it separates
        # "the inner optimisation failed" from "the feature map cannot represent
        # the perturbation at all", which are very different diagnoses.  The
        # exact path gets it from its own untouched first rollout, so that the
        # baseline and the columns share one normalisation window.
        if inner_solve == "exact":
            # Built once and shared by the recorded fit and the ridge ladder:
            # ``L`` and ``d`` do not depend on ``lambda``, so a sweep of the
            # ridge that rebuilt them would re-run every perturbed PDE solve and
            # reproduce the same two matrices bit for bit.
            basis = surrogate.network
            L, d, initial_rms = _inner_design(
                surrogate, data, targets[j], delta=delta, index=j, diag=diag
            )
            diag[f"inner_floor_rms_{j}"] = float(np.sqrt(np.mean(
                (d - L @ _ridge_coef(L, d, None, basis.out_dim, 0.0)) ** 2
            )))
            m_column = _exact_minimiser_from(
                L, d, basis, lambda_n, index=j, diag=diag
            )
            m_hat_blocks.append(m_column)
            diag[f"mismatch_rms_{j}"] = float(np.sqrt(np.mean(m_column ** 2)))
            diag[f"match_mse_{j}"] = diag[f"mismatch_rms_{j}"] ** 2
            if lambda_grid is not None:
                root = (_ridge_root(basis) if any(lam > 0.0 for lam in lambda_grid)
                        else None)
                grid_blocks.append([
                    d - L @ _ridge_coef(L, d, root, basis.out_dim, float(lam))
                    for lam in lambda_grid
                ])
        else:
            with torch.no_grad():
                initial_rms = float(
                    ((targets[j] - surrogate(data.initial, data.t, data.z)) / delta)
                    .pow(2)
                    .mean()
                    .sqrt()
                )
            _train_perturbation(
                surrogate,
                data,
                targets[j],
                delta=delta,
                lambda_n=lambda_n,
                epochs=epochs,
                inner_steps=inner_steps,
                lr=lr,
                batch_size=batch_size,
            )
            with torch.no_grad():
                matched = surrogate(data.initial, data.t, data.z)
                mismatch = (targets[j] - matched) / delta
            m_hat_blocks.append(mismatch.reshape(-1).cpu().numpy())
            diag[f"match_mse_{j}"] = float(
                (targets[j] - matched).pow(2).mean()
            )
            diag[f"mismatch_rms_{j}"] = float(mismatch.pow(2).mean().sqrt())
        diag[f"mismatch_rms_init_{j}"] = initial_rms
        diag[f"mismatch_shrinkage_{j}"] = (
            initial_rms / diag[f"mismatch_rms_{j}"]
            if diag[f"mismatch_rms_{j}"] > 0
            else float("inf")
        )
        if verbose:
            print(
                f"  h_{j}: match mse {diag[f'match_mse_{j}']:.3e}  "
                f"mismatch rms {diag[f'mismatch_rms_{j}']:.3e}"
            )

    m_hat = np.stack(m_hat_blocks, axis=1)  # (n_inf, p)

    # -- residual variance and the efficient covariance ------------------------
    # ``sigma_hat^2`` has to be the variance of the noise actually entering
    # ``Y_i = u(X_i) + eps_i``, so it is averaged over the points that carry
    # noise and not over the exact slice.  The two differ by exactly the share of
    # exact points (1/n_obs with ``noise_at_initial=False``); averaging over the
    # whole array therefore deflates sigma_hat by ``sqrt(1 - 1/n_obs)``, and
    # nothing compensates for it, because the surrogate and the perturbed model
    # integrate from the same initial state and so have identical values -- and
    # hence ``M_hat = 0`` -- on that slice.
    with torch.no_grad():
        fitted = model(data.initial, data.t, data.z)
        resid = (data.states - fitted).reshape(-1)
        mask = data.noisy
        mask = None if mask is None else mask.reshape(-1)
        n_all = int(resid.numel())
        if mask is None or not bool(mask.any()):
            sigma2 = float(resid.pow(2).mean())
            n_sigma = n_all
        else:
            sigma2 = float(resid[mask].pow(2).mean())
            n_sigma = int(mask.sum())
        sigma2_all = float(resid.pow(2).mean())
    gram = m_hat.T @ m_hat / m_hat.shape[0]
    middle = None
    if cov_estimator == "robust":
        # Eicker--Huber--White: the middle matrix keeps the pointwise weights
        # ``r_i^2`` instead of collapsing them to the single ``sigma_hat^2``.
        # The efficient form is the special case ``r_i^2 -> mean(r_i^2)``, so the
        # difference between the two is exactly the part of the residual that is
        # correlated with ``||M_i||`` -- i.e. the systematic misfit.
        weights = resid.detach().cpu().numpy().astype(np.float64) ** 2
        middle = (m_hat * weights[:, None]).T @ m_hat / m_hat.shape[0]
    try:
        gram_inv = np.linalg.inv(gram)
        cov = gram_inv @ middle @ gram_inv if middle is not None else sigma2 * gram_inv
    except np.linalg.LinAlgError:
        # ``gram`` is exactly singular, so this design does not identify every
        # coordinate.  The pseudo-inverse is the least-squares covariance --
        # right for the identified directions, and *zero* for the unidentified
        # ones -- so taking it unqualified would report a zero-width interval,
        # i.e. certainty, for a parameter the data cannot pin down.  That is the
        # one error a coverage claim cannot absorb, so the null directions are
        # given an infinite variance instead: the interval is then empty, and
        # ``inf`` propagates through every contrast that touches them.
        #
        # This is not hypothetical.  On the unidentifiable forced-Burgers limit
        # (``P(Z = 1) = 1``, where ``M_0 = {Z - (2 pi - 1)} K g`` vanishes
        # identically) the code only reports a finite ``se`` because the inner
        # fit *fails* to reach ``h_0``; a successful fit drives ``M_hat`` to zero
        # and lands here.  See ``docs/size.md``.
        pinv = np.linalg.pinv(gram)
        cov = pinv @ middle @ pinv if middle is not None else sigma2 * pinv
        null = np.diag(pinv) <= 0.0
        variances = np.array(np.diag(cov), dtype=np.float64, copy=True)
        variances[null] = np.inf
        np.fill_diagonal(cov, variances)
        diag["pinv_used"] = 1.0
        diag["n_unidentified"] = float(null.sum())
    diag["sigma"] = math.sqrt(sigma2)
    diag["sigma_all_points"] = math.sqrt(sigma2_all)
    diag["n_points_used_for_sigma"] = float(n_sigma)
    diag["noise_fraction"] = float(n_sigma / n_all) if n_all else 1.0
    diag["cond_gram"] = float(np.linalg.cond(gram))
    # A Gram matrix that inverts without raising can still be numerically
    # singular, in which case the reported variances are meaningless in a way
    # that no other diagnostic shows: record the smallest one so that a
    # *negative* or vanishing variance is visible in the record rather than
    # surfacing as a ``nan`` half-width downstream.
    diag["min_var"] = float(np.min(np.diag(cov)))
    # One-number summary of the two middle matrices: the factor by which
    # pointwise residual weighting inflates the reported variance.  It is exact
    # for ``p = 1`` (Case 1, and either coordinate of Case 2) and equals 1 under
    # correct specification, so a value well above 1 is a direct read-out of
    # systematic misfit and of how much ``se`` would move under ``"robust"``.
    diag["cov_estimator_robust"] = 0.0 if middle is None else 1.0
    row_norm = (m_hat ** 2).sum(axis=1)
    weights_all = resid.detach().cpu().numpy().astype(np.float64) ** 2
    denom = sigma2 * float(row_norm.mean())
    diag["middle_inflation"] = float(
        (weights_all @ row_norm) / m_hat.shape[0] / denom
    ) if denom > 0.0 else float("inf")

    # -- the ridge ladder, on the same sandwich -------------------------------
    # ``se(lambda)`` is the *whole* of the calibration: the inner class fixes
    # where the curve starts and where it saturates, and ``lambda`` places it
    # between the two.  Reporting the curve rather than a point is what makes a
    # size study convex-cost -- one set of rollouts, one solve per grid point.
    if lambda_grid is not None and grid_blocks:
        lambdas = [float(lam) for lam in lambda_grid]
        rms_by_lambda: List[List[float]] = []
        se_by_lambda: List[List[float]] = []
        for k in range(len(lambdas)):
            m_k = np.stack([block[k] for block in grid_blocks], axis=1)
            gram_k = m_k.T @ m_k / m_k.shape[0]
            try:
                inv_k = np.linalg.inv(gram_k)
            except np.linalg.LinAlgError:
                inv_k = np.linalg.pinv(gram_k)
            # The robust middle matrix is built from ``M_hat`` itself, so it
            # moves with ``lambda`` and has to be rebuilt per grid point; under
            # the efficient estimator it is ``sigma_hat^2`` and the whole ladder
            # shares one.  Reusing the *point* middle here would have made the
            # ladder disagree with the pointwise call it is supposed to
            # reproduce, which is the one thing a ladder must not do.
            middle_k = (
                (m_k * weights_all[:, None]).T @ m_k / m_k.shape[0]
                if middle is not None else None
            )
            cov_k = (inv_k @ middle_k @ inv_k if middle_k is not None
                     else sigma2 * inv_k)
            # ``cov`` is the *design-unit* covariance: ``cov_estimator`` above
            # is unscaled, and the ``1 / sqrt(n_est)`` of the asymptotic statement
            # is applied where the interval is formed (:meth:`interval`).  The
            # ladder is read against ``sd`` and against ``interval()["se"]``, so
            # it has to be reported in the same units, or a caller comparing the
            # two would be off by ``sqrt(n_est)`` -- a factor of 14 on the rd
            # ladder -- while both numbers looked plausible.
            se_k = np.sqrt(np.clip(np.diag(cov_k), 0.0, None)) / math.sqrt(n_est)
            rms_by_lambda.append(
                np.sqrt((m_k ** 2).mean(axis=0)).astype(float).tolist()
            )
            se_by_lambda.append(se_k.astype(float).tolist())
        diag["ridge_lambdas"] = lambdas
        for j in range(p):
            diag[f"ridge_rms_{j}"] = [row[j] for row in rms_by_lambda]
            diag[f"ridge_se_{j}"] = [row[j] for row in se_by_lambda]
            # Same ratio the pointwise path records as ``mismatch_shrinkage_j``
            # (``rms(d) / rms(M_hat)``), so a ladder row and a fitted row carry
            # the same diagnostics and a reader can compare them column by
            # column without knowing which route produced which.
            init_j = diag.get(f"mismatch_rms_init_{j}")
            diag[f"ridge_shrink_{j}"] = [
                float(init_j) / row[j] if row[j] > 0.0 else float("inf")
                for row in rms_by_lambda
            ] if init_j is not None else [float("nan")] * len(lambdas)

    for prm, flag in zip(frozen_net, was_net):
        prm.requires_grad_(flag)
    for prm, flag in zip(frozen_theta, was_theta):
        prm.requires_grad_(flag)

    return InferenceResult(
        names=names,
        theta=theta_hat.cpu().numpy(),
        cov=cov,
        sigma=math.sqrt(sigma2),
        m_hat=m_hat,
        n_inf=n_inf,
        n_est=n_est,
        delta=float(delta),
        diagnostics=diag,
    )


def _train_perturbation(
    surrogate: ProfilingModel,
    data: Trajectories,
    target: torch.Tensor,
    *,
    delta: float,
    lambda_n: float,
    epochs: int,
    inner_steps: int,
    lr: float,
    batch_size: Optional[int] = None,
) -> None:
    """Gradient descent on the Algorithm-2 objective for one ``h_j``.

    The target trajectory is fixed, so nothing here needs to differentiate
    through ``u(theta_hat + delta e_j, f_hat)``.

    The objective is the paper's,

    .. math::

        \\frac1n \\sum_i \\|\\delta^{-1}\\{\\tilde u(X_i; \\theta\\!+\\!\\delta e_j, \\hat f)
            - \\tilde u(X_i; \\theta, \\hat f + \\delta h_j)\\}\\\\|^2
            + \\tilde\\lambda_n J(h_j),

    and the ``delta^{-1}`` is **load-bearing**, not cosmetic.  ``surrogate``
    carries ``mechanism_scale = delta``, so the network already emits ``delta h_j``;
    dividing the residual by ``delta`` restores the ``O(1)`` scale of the
    quantity being matched.  Drop it and the whole objective -- gradients
    included -- is scaled by ``delta^2 = 1e-10``, far below Adam's ``eps``, so the
    optimiser stalls instead of fitting.  Keep it and Adam's default step is
    correctly sized.
    """
    params = [prm for prm in surrogate.network.parameters() if prm.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    n = len(data)
    step = min(batch_size or n, n)
    order = torch.arange(n)
    for _ in range(epochs):
        for start in range(0, n, step):
            idx = order[start : start + step]
            sub = data.subset(idx)
            # The paper updates each batch ``inner_steps`` times before moving
            # on, in the inference phase as well as during estimation.
            for _ in range(max(1, inner_steps)):
                optimizer.zero_grad(set_to_none=True)
                pred = surrogate(sub.initial, sub.t, sub.z)
                resid = (target[idx] - pred) / delta
                loss = resid.pow(2).mean()
                if lambda_n:
                    loss = loss + lambda_n * surrogate.network.penalty()
                loss.backward()
                optimizer.step()
