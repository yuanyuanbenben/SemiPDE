"""Baseline 4: parameter cascading for semiparametric PDEs.

The method is the one of \citet{ramsay2007parameter}, applied to a PDE as in
\citet{xun2013parameter}, *with the unknown mechanism expanded in a basis of its
own*.  The model is the same one the proposal fits,

.. math:: \\partial_t u_z = G_z(u_z;\\theta) + F_f(u_z),

and both unknowns are sieves:

``u``
    a *linear* basis expansion ``u(x, t) = \\sum_{k,j} c_{kj} \\phi_k(x) \\psi_j(t)``,
    periodic Fourier modes in space and cubic B-splines in time, with one
    coefficient block ``c`` per trajectory;
``f``
    a tensor cubic B-spline sieve over the mechanism's inputs,
    ``f(v) = \\sum_k \\beta_k B_k(v)``, with **one** coefficient vector ``\\beta``
    shared by every trajectory -- the same shared nuisance the proposal fits.

Neither unknown is obtained by solving the PDE forward: the PDE enters the
criterion, not the representation.  The three levels are

outer (minimise over ``theta``)
    ``||u_c - y||^2`` at the inner optimum
middle (minimise over ``beta``; ``u`` held fixed)
    ``||d_t u - G(u; theta) - F_f(u)||^2 + mu * ||D^2 beta||^2``
inner (minimise over ``c``; ``theta``, ``beta`` fixed)
    ``||u_c - y||^2 + lam * (||d_t u_c - G(u_c; theta) - F_f(u_c)||^2 + mu * ||D^2 beta||^2)``

``lam`` is the method's smoothing parameter: it trades the data fit against the
requirement that the fitted trajectory satisfy the PDE at the current
``(theta, beta)``.  ``mu`` is the roughness penalty on ``f`` (the second
difference of the coefficient vector, the P-spline convention).  Both are
described under :func:`fit_cascading`.

Why the sieve matters
---------------------
With no ``f`` (the parametric version of this baseline) the outer criterion is
what identifies ``theta``.  With ``f`` free but *unpenalised*, the same
criterion can be made flat: the fitted state plus a sufficiently flexible
mechanism reproduce the observed trajectory for a whole range of ``theta``,
and the search then reports whichever value the bracket ends at.  The
roughness penalty on ``\\beta`` is the classical device against exactly that,
and ``lam`` is what keeps the state close enough to the PDE for ``theta`` to
carry information.  ``F = 0`` is the hard case for this baseline in the other
direction: it fits a mechanism it does not need, which is the point of
including it in that experiment.

Scope
-----
The state basis is Fourier on a periodic grid and cosine on a non-periodic one.
Both are the eigenbasis of the Laplacian under the case's own boundary
condition -- the cosine basis ``cos(pi k s)`` has zero derivative at both ends,
which is the same Neumann condition ``operators.laplacian`` imposes with
replicate padding on those grids -- so the basis spans the solutions the spec
can produce without the code knowing which variant it was handed.  (A Dirichlet
or mixed condition would need a different basis; the code raises there rather
than silently wrapping the domain.)  The ``f`` sieve reads whatever
:meth:`PDESpec.features` reads, so it is a function of the state on the
state-driven cases and of the coordinate on the coordinate-driven ones -- the
class the case itself allows, not a narrower one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from .pde import PDESpec


# ---------------------------------------------------------------------------
# state basis: Fourier in space, cubic B-splines in time
# ---------------------------------------------------------------------------
def fourier_design(grid, n_modes: int) -> np.ndarray:
    """Design matrix of the periodic Fourier basis on ``grid``.

    One column per basis function, one row per grid point in the grid's own
    (C order) flattening, so that ``U = Phi @ c`` has the layout of
    ``u.reshape(-1)``.  ``n_modes`` is capped per axis at the largest wave
    number the axis can carry (``(n - 1) // 2``), because a mode at or above
    Nyquist is aliased on the grid and would make the residual fit noise.
    """
    if not grid.periodic:
        raise ValueError(
            "cascading's basis is periodic Fourier; the grid is not periodic"
        )
    per_axis = []
    for axis, (a, b), n in zip(grid.axes(), grid.ranges, grid.shape):
        s = (axis - float(a)) / (float(b) - float(a))
        k_max = max(1, min(int(n_modes), (int(n) - 1) // 2))
        cols = [np.ones_like(s)]
        for k in range(1, k_max + 1):
            cols.append(np.cos(2.0 * np.pi * k * s))
            cols.append(np.sin(2.0 * np.pi * k * s))
        per_axis.append(np.stack(cols, axis=1))
    design = per_axis[0]
    for extra in per_axis[1:]:
        design = np.kron(design, extra)
    return design


def cosine_design(grid, n_modes: int) -> np.ndarray:
    """Design matrix of the Neumann cosine basis on a non-periodic ``grid``.

    ``cos(pi k s)`` on ``s in [0, 1]``: ``k = 0`` is the mean and every mode has
    zero derivative at both endpoints.  That is the eigenbasis
    :func:`~semipde.operators.laplacian` effectively uses there (replicate
    padding), so a state expanded in it is one the spec's drift can produce.
    ``k`` is capped at ``n - 1`` per axis, where a mode becomes unrepresentable
    on the grid.
    """
    per_axis = []
    for axis, (a, b), n in zip(grid.axes(), grid.ranges, grid.shape):
        s = (axis - float(a)) / (float(b) - float(a))
        k_max = max(1, min(int(n_modes), int(n) - 1))
        cols = [np.cos(np.pi * k * s) for k in range(k_max + 1)]
        per_axis.append(np.stack(cols, axis=1))
    design = per_axis[0]
    for extra in per_axis[1:]:
        design = np.kron(design, extra)
    return design


def state_design(grid, n_modes: int) -> np.ndarray:
    """The state basis for ``grid``'s boundary condition: Fourier or cosine."""
    return fourier_design(grid, n_modes) if grid.periodic else cosine_design(grid, n_modes)


def bspline_design(t: np.ndarray, t0: float, t1: float, n_basis: int,
                   degree: int = 3) -> tuple:
    """Cubic B-spline design matrix and its first derivative at times ``t``.

    Clamped knot vector: ``degree + 1`` copies of each endpoint and evenly
    spaced interior knots, so ``n_basis`` functions span ``[t0, t1]`` and the
    fit is not driven by an extrapolation outside the observation window.
    """
    n_basis = max(int(n_basis), degree + 2)
    n_interior = n_basis - degree - 1
    interior = np.linspace(t0, t1, n_interior + 2)[1:-1]
    knots = np.concatenate(
        [[t0] * (degree + 1), interior, [t1] * (degree + 1)]
    )
    base = BSpline(knots, np.eye(n_basis), degree, extrapolate=True)
    return base(np.asarray(t, float)), base.derivative()(np.asarray(t, float))


# ---------------------------------------------------------------------------
# fountain basis for f: a differentiable tensor B-spline sieve
# ---------------------------------------------------------------------------
def _clamped_knots(lo: float, hi: float, n_basis: int, degree: int,
                   dtype: torch.dtype) -> torch.Tensor:
    """Clamped, evenly spaced knots for ``n_basis`` B-splines on ``[lo, hi]``."""
    n_interior = max(int(n_basis) - degree - 1, 1)
    interior = torch.linspace(lo, hi, n_interior + 2, dtype=dtype)[1:-1]
    return torch.cat([
        torch.full((degree + 1,), float(lo), dtype=dtype),
        interior,
        torch.full((degree + 1,), float(hi), dtype=dtype),
    ])


def _bspline_basis_torch(x: torch.Tensor, knots: torch.Tensor,
                         degree: int) -> torch.Tensor:
    """Cox--de Boor recursion in torch; ``(...) -> (..., n_basis)``.

    Torch rather than :mod:`scipy.interpolate` because ``f`` is evaluated at the
    *fitted* state on the state-driven cases, so the design matrix has to be
    differentiable in the coefficients of ``u``.  The right endpoint is folded
    into the last non-empty interval, as in the clamped construction.
    """
    x = x.unsqueeze(-1)
    n_basis = int(knots.numel()) - degree - 1
    table = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
    def ratio(offset: torch.Tensor, span: torch.Tensor,
              block: torch.Tensor) -> torch.Tensor:
        # ``(x - a) / (b - a) * block``, with the degenerate spans (the repeated
        # knots of the clamped ends) contributing *exactly* zero -- and zero
        # gradient.  Guarding the denominator alone is not enough: the recursion
        # leaves ~1e-47 instead of an exact 0 in some of those blocks, and
        # dividing that by a guarded 1e-300 is an inf that then cancels to nan.
        empty = span == 0
        safe = torch.where(empty, torch.ones_like(span), span)
        return (~empty).to(x.dtype) * (offset / safe) * block

    for d in range(1, degree + 1):
        n = table.shape[-1] - 1
        left = knots[:n]
        left_d = knots[d:d + n]
        right_d = knots[d + 1:d + 1 + n]
        right = knots[1:1 + n]
        table = ratio(x - left, left_d - left, table[..., :n]) + \
                ratio(right_d - x, right_d - right, table[..., 1:n + 1])
    # The half-open convention leaves the right endpoint to no interval; with
    # clamped knots the last basis function is the one that reaches it.
    return table + (x == knots[-1]).to(x.dtype) * torch.eye(
        n_basis, dtype=x.dtype, device=x.device
    )[-1]


def _second_difference_penalty(n: int, dtype: torch.dtype) -> torch.Tensor:
    """``D^T D`` of the second-difference operator on ``n`` coefficients."""
    m = max(n - 2, 1)
    D = torch.zeros(m, n, dtype=dtype)
    idx = torch.arange(min(m, n - 2))
    D[idx, idx] = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    return D.T @ D


class SieveBasis:
    """Tensor cubic-spline sieve for the unknown mechanism ``f``.

    ``design(features)`` is ``(..., d) -> (..., n_terms)``; the coefficients
    ``beta`` are ``(n_terms, out_dim)``.  The knot window is measured on the
    *observed* training states, widened by ``margin``, and clamped afterwards so
    that a state the fit moves slightly outside the window is evaluated at the
    boundary rather than extrapolated.
    """

    def __init__(self, lo: torch.Tensor, hi: torch.Tensor, n_basis: int = 8,
                 degree: int = 3, margin: float = 0.1,
                 dtype: torch.dtype = torch.float64) -> None:
        self.dim = int(lo.numel())
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        # A clamped knot vector with ``n_basis`` functions needs at least one
        # interior knot, i.e. ``n_basis > degree + 1``; below that
        # ``_clamped_knots`` silently returns a basis of a different size and
        # the design matrix no longer matches ``beta``.
        if self.n_basis <= self.degree + 1:
            raise ValueError(
                f"n_basis={self.n_basis} is too small for degree={self.degree}: "
                f"a clamped B-spline basis of this degree needs at least "
                f"{self.degree + 2} functions"
            )
        span = (hi - lo).clamp_min(1e-6)
        self.lo = lo - margin * span
        self.hi = hi + margin * span
        self.knots = [
            _clamped_knots(float(self.lo[a]), float(self.hi[a]), n_basis,
                           degree, dtype)
            for a in range(self.dim)
        ]
        self.n_terms = int(n_basis) ** self.dim
        pen = _second_difference_penalty(n_basis, dtype)
        P = pen
        for _ in range(self.dim - 1):
            P = torch.kron(P, pen)
        # The P-spline scaling: ``sum (D^2 beta)^2 / h^3`` approximates
        # ``integral (f'')^2``, so ``mu`` then has the meaning of a smoothing
        # parameter and is comparable across cases instead of depending on how
        # many knots happen to sit in the window.
        spacing = (self.hi - self.lo) / max(n_basis - degree, 1)
        scale = float(spacing.prod().clamp_min(1e-12))
        self.penalty_matrix = P / scale

    def design(self, features: torch.Tensor) -> torch.Tensor:
        cols = []
        for a in range(self.dim):
            x = features[..., a].clamp(float(self.lo[a]), float(self.hi[a]))
            cols.append(_bspline_basis_torch(x, self.knots[a], self.degree))
        out = cols[0]
        for extra in cols[1:]:
            out = out.unsqueeze(-1) * extra.unsqueeze(-2)
        return out.reshape(*features.shape[:-1], self.n_terms)

    def penalty(self, beta: torch.Tensor) -> torch.Tensor:
        """``beta^T P beta`` summed over the output components."""
        return (beta * (self.penalty_matrix @ beta)).sum()


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
@dataclass
class CascadingFit:
    """One cascading fit: the parameter, the mechanism, the scored state."""

    theta: Dict[str, object]
    prediction: torch.Tensor  # (n, T, d_y, *grid), the curves being scored
    sieve: SieveBasis
    beta: torch.Tensor
    lam: float
    mu: float
    n_modes: int
    n_basis_t: int
    data_sse: float = float("nan")
    residual: float = float("nan")
    outer_evaluations: int = 0


class CascadingModel:
    """The slice of the ``ProfilingModel`` protocol ``score_common`` reads.

    Unlike the parametric version of this baseline, the fitted mechanism is a
    function that has to be reported: ``||f_hat - f_0||`` is one of the table's
    columns, and on ``F = 0`` the size of this sieve's output is what says how
    much spurious mechanism the method invented.
    """

    def __init__(self, spec: PDESpec, fit: CascadingFit) -> None:
        self.spec = spec
        self.fit = fit

    def __call__(self, initial, t, z) -> torch.Tensor:
        return self.fit.prediction

    def _sieve_at(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        coords = self.spec.coordinates(u.shape[0], u.dtype, u.device)
        feats = self.spec.features(u, coords, z)
        out = self.fit.sieve.design(feats) @ self.fit.beta
        return out.movedim(-1, 1)

    def mechanism(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._sieve_at(u, z)

    def theta_detached(self) -> Dict[str, object]:
        return dict(self.fit.theta)


def _flatten_curves(states: torch.Tensor):
    """``(n, T, d_y, *grid)`` to ``(n, d_y, T, P)`` and the grid shape."""
    n, n_obs, d_y = states.shape[:3]
    grid_shape = tuple(states.shape[3:])
    out = states.reshape(n, n_obs, d_y, -1).permute(0, 2, 1, 3).contiguous()
    return out, grid_shape


def _observed_features(spec: PDESpec, states: torch.Tensor,
                       z: torch.Tensor) -> torch.Tensor:
    """The sieve's inputs at every observed point of every curve: ``(N, d)``."""
    u = states.reshape(states.shape[0] * states.shape[1], *states.shape[2:])
    coords = spec.coordinates(u.shape[0], u.dtype, u.device)
    zz = z.repeat_interleave(int(states.shape[1])) if z.numel() == states.shape[0] else z
    feats = spec.features(u, coords, zz)
    return feats.reshape(-1, feats.shape[-1])


class _Inner:
    """The inner and middle problems for one ``theta``, reusable across calls.

    Both coefficient blocks are kept between calls so that the outer search can
    warm start: consecutive ``theta`` in a grid differ little, and without the
    warm start the inner solve -- not the criterion -- dominates the run time.
    """

    def __init__(self, spec, grid, t_col, phi, psi_obs, psi_col, psi_col_d,
                 states, z, sieve, lam, mu, iters, f_iters, rounds):
        self.spec = spec
        self.grid = grid
        self.phi = phi
        self.psi_obs = psi_obs
        self.psi_col = psi_col
        self.psi_col_d = psi_col_d
        self.states = states          # (M, d_y, T, P)
        self.z = z                    # (M,)
        self.sieve = sieve
        self.lam = float(lam)
        self.mu = float(mu)
        self.iters = int(iters)
        self.f_iters = int(f_iters)
        self.rounds = int(rounds)
        self.M, self.d_y, self.n_obs, self.P = states.shape
        self.t_col = t_col.shape[0]

    # -- mechanism and criterion -------------------------------------------------
    def _mechanism(self, beta: torch.Tensor):
        def f_hat(u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            coords = self.spec.coordinates(u.shape[0], u.dtype, u.device)
            feats = self.spec.features(u, coords, z)
            return (self.sieve.design(feats) @ beta).movedim(-1, 1)
        return f_hat

    def _state(self, coeffs: torch.Tensor) -> tuple:
        U_obs = torch.einsum("mdkj,pk,tj->mdtp", coeffs, self.phi, self.psi_obs)
        U_col = torch.einsum("mdkj,pk,tj->mdtp", coeffs, self.phi, self.psi_col)
        U_dot = torch.einsum("mdkj,pk,tj->mdtp", coeffs, self.phi, self.psi_col_d)
        M = self.M
        u = U_col.permute(0, 2, 1, 3).reshape(M * self.t_col, self.d_y,
                                              *self.grid.shape)
        u_dot = U_dot.permute(0, 2, 1, 3).reshape(M * self.t_col, self.d_y,
                                                  *self.grid.shape)
        z = self.z.repeat_interleave(self.t_col)
        return U_obs, u, u_dot, z

    def loss(self, coeffs, beta, theta, parts: bool = False):
        U_obs, u, u_dot, z = self._state(coeffs)
        data = (U_obs - self.states).pow(2).mean()
        drift = self.spec.drift(u, z, theta, self._mechanism(beta))
        residual = (u_dot - drift).pow(2).mean()
        pen = self.sieve.penalty(beta)
        total = data + self.lam * (residual + self.mu * pen)
        if parts:
            return total, float(data.detach()), float(residual.detach())
        return total

    # -- the two coefficient blocks ---------------------------------------------
    def _data_start(self) -> torch.Tensor:
        """The state coefficients that fit the data alone, in closed form.

        ``U = Psi C Phi^T`` per curve and field, so the least-squares ``C`` is a
        pair of pseudo-inverses.  Starting the cascade there rather than at zero
        is what makes the inner solve converge: from zero the first pass has to
        discover the whole solution, and the penalty makes that landscape stiff
        (measured on rd at the true parameter: first-order steps from this start
        stop at twice the attainable data term, quasi-Newton from it reaches
        the optimum in ~100 iterations).
        """
        psi_pinv = torch.linalg.pinv(self.psi_obs)   # (J, T)
        phi_pinv = torch.linalg.pinv(self.phi)       # (K, P)
        flat = self.states.reshape(self.M * self.d_y, self.n_obs, self.P)
        coeffs = torch.einsum("jt,mtp,kp->mkj", psi_pinv, flat, phi_pinv)
        return coeffs.reshape(self.M, self.d_y, self.phi.shape[1],
                              self.psi_obs.shape[1])

    def _c_step(self, coeffs, beta, theta, iters: int):
        """The inner level: the state, penalised data-plus-residual.

        Quasi-Newton from the first iteration.  Measured on rd at the true
        parameter, LBFGS reaches the optimum (data 8.59e-3, residual 6e-8) in
        ~100 iterations from the closed-form start, while Adam at the same
        budget stops at twice the data term and then *wanders* along the flat
        direction that the residual leaves open -- the criterion is quadratic
        enough that a curvature-aware method is the right one, and the
        outer search needs the bottom of this problem, not a neighbourhood of
        it.
        """
        if coeffs is None:
            coeffs = self._data_start()
        # ``clone`` keeps the strides, and LBFGS flattens its parameters with a
        # ``view``: a non-contiguous block (the one ``_data_start``'s einsum
        # leaves behind) makes that view raise, and the ``except`` below then
        # silently turns the step into a no-op.  Materialise it first.
        coeffs = coeffs.detach().contiguous().clone().requires_grad_(True)
        lbfgs = torch.optim.LBFGS(
            [coeffs], lr=1.0, max_iter=int(iters), tolerance_grad=1e-12,
            tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            loss = self.loss(coeffs, beta, theta)
            loss.backward()
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError:  # pragma: no cover - a line search that diverged
            pass
        with torch.no_grad():
            _, data, residual = self.loss(coeffs, beta, theta, parts=True)
        return coeffs.detach(), data, residual

    def _beta_step(self, coeffs, beta, theta):
        """The middle level: ``f`` given ``u``, penalised residual only."""
        coeffs = coeffs.detach()
        beta = beta.detach().contiguous().clone().requires_grad_(True)
        # ``U_obs`` does not enter this level, so the state is a constant here.
        _, u, u_dot, z = self._state(coeffs)

        def objective(b):
            drift = self.spec.drift(u, z, theta, self._mechanism(b))
            return (u_dot - drift).pow(2).mean() + self.mu * self.sieve.penalty(b)

        lbfgs = torch.optim.LBFGS(
            [beta], lr=1.0, max_iter=int(self.f_iters), tolerance_grad=1e-12,
            tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            loss = objective(beta)
            loss.backward()
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError:  # pragma: no cover
            pass
        return beta.detach()

    def solve(self, theta, coeffs=None, beta=None, iters: Optional[int] = None,
              fit_beta: bool = True):
        """Alternate the two blocks; return ``(c, beta, data, residual)``."""
        if beta is None:
            beta = torch.zeros(
                self.sieve.n_terms, self.spec.mechanism_out_dim,
                dtype=self.states.dtype, device=self.states.device,
            )
        n = int(iters if iters is not None else self.iters)
        rounds = self.rounds if fit_beta else 1
        for r in range(rounds):
            coeffs, data, residual = self._c_step(coeffs, beta, theta, n)
            if fit_beta and r + 1 < rounds:
                beta = self._beta_step(coeffs, beta, theta)
        # One more state pass after the last mechanism update, so that the
        # returned criterion is evaluated at the pair the cascade stopped at
        # rather than at a state fitted against the previous ``beta``.
        if fit_beta and rounds > 1:
            coeffs, data, residual = self._c_step(coeffs, beta, theta, n)
        return coeffs.detach(), beta.detach(), data, residual


def _theta_from(x: Sequence[float], keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {k: torch.tensor(float(v), dtype=torch.float64) for k, v in zip(keys, x)}


def _outer_search(inner: _Inner, keys: Sequence[str], *, log_lo: float,
                  log_hi: float, coarse: int, polish_maxfev: int,
                  polish_maxiter: int) -> tuple:
    """Two-stage search over ``theta`` on a log grid, then Nelder--Mead.

    The search is on ``log10(theta)`` over ``[log_lo, log_hi]`` rather than
    around a point estimate: the method has no natural starting value, and a
    grid that brackets the parameter is cheaper than making a bad warm start
    the outer optimum.  The grid is coarse-then-fine, and the refinement ends
    with Nelder--Mead so that the reported value is not a grid point.

    Returns ``(x, coeffs, beta, data_sse, n_evaluations)``.
    """
    p = len(keys)
    evaluations = 0
    best = None
    if p == 1:
        grid = np.linspace(log_lo, log_hi, coarse)
        combinations = [[g] for g in grid]
    else:
        # A tensor grid of ``coarse ** p`` points is what the one-dimensional
        # branch would cost *per coordinate*; cap the total instead, so that a
        # two-parameter case does not multiply the inner solves by ``coarse``.
        n_per = max(3, int(round(coarse ** (1.0 / p))))
        grid = np.linspace(log_lo, log_hi, n_per)
        combinations = [list(v) for v in np.stack(
            np.meshgrid(*([grid] * p), indexing="ij"), axis=-1
        ).reshape(-1, p)]
    coeffs = None
    beta = None
    for x in combinations:
        th = _theta_from(10.0 ** np.asarray(x), keys)
        coeffs, beta, data, _ = inner.solve(th, coeffs, beta)
        evaluations += 1
        if best is None or data < best[1]:
            best = (list(x), data, beta)

    x0, data_best, beta_best = best

    def objective(v):
        nonlocal evaluations, coeffs, beta
        # Keep the search inside the bracket.  Nelder--Mead on a log scale can
        # otherwise wander into ten orders of magnitude, where the inner fit is
        # numerically flat and the returned point is meaningless.
        if np.any(v < log_lo) or np.any(v > log_hi):
            return 1e12
        th = _theta_from(10.0 ** np.asarray(v), keys)
        coeffs, beta, data, _ = inner.solve(th, coeffs, beta,
                                            iters=max(inner.iters // 2, 40))
        evaluations += 1
        return data

    # Nelder--Mead's default initial simplex is a 5% step of each coordinate,
    # which for a log10 coordinate of -4 is 2e-4 -- so a search that starts on
    # the *boundary* of the bracket (the common outcome when the criterion is
    # flat in a coordinate) cannot leave it, and converges on the boundary
    # after two evaluations.  The step has to span decades, not percentages.
    start = np.asarray(x0, float)
    simplex = np.tile(start, (p + 1, 1))
    for i in range(p):
        simplex[i + 1, i] = min(start[i] + 1.0, log_hi)
        if simplex[i + 1, i] == start[i]:  # already at the top of the bracket
            simplex[i + 1, i] = max(start[i] - 1.0, log_lo)
    # Restart from the grid optimum: the criterion is evaluated at the pair the
    # objective last returned, and the caller keeps whatever ended the search.
    beta = beta_best
    res = minimize(
        objective, start, method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-14,
                 "maxiter": int(polish_maxiter), "maxfev": int(polish_maxfev),
                 "initial_simplex": simplex},
    )
    x = res.x if res.fun < data_best else np.asarray(x0, float)
    th = _theta_from(10.0 ** np.asarray(x), keys)
    coeffs, beta, data, _ = inner.solve(
        th, coeffs, beta, iters=max(inner.iters, 300))
    evaluations += 1
    return x, coeffs, beta, data, evaluations


def fit_cascading(
    spec: PDESpec,
    config,
    split,
    *,
    lam: float = 100.0,
    mu: float = 1e-4,
    n_modes: int = 8,
    n_basis_t: int = 10,
    n_basis_f: int = 6,
    f_degree: int = 3,
    inner_iters: int = 150,
    f_iters: int = 40,
    rounds: int = 2,
    collocation_factor: int = 5,
    log_lo: float = -4.0,
    log_hi: float = 1.0,
    coarse: int = 21,
    polish_maxfev: int = 80,
    polish_maxiter: int = 50,
    seed: int = 0,
    verbose: bool = False,
) -> CascadingFit:
    """Cascade the state, the mechanism and the parameter.

    ``n_modes`` and ``n_basis_t`` size the *state* sieve, ``n_basis_f`` the
    *mechanism* sieve per axis.  All three are fixed: no case, noise level or
    parameter value selects them, and none of them is small enough to stand in
    for the mechanism (the point of the expansion is that a basis which can
    only represent ``f`` out of the box is a parametric model in disguise).
    ``lam`` weights the PDE residual against the data fit in the inner
    criterion; ``mu`` weights the second-difference penalty on ``f`` in the
    middle and inner criteria.  ``lam`` is large enough that the residual term
    is a non-negligible part of the inner criterion (measured: 2-7% of the data
    term at the fitted state on the cases here) -- below that the state is an
    unconstrained smoother and the outer criterion carries no information
    about ``theta`` at all.

    What the outer criterion can and cannot identify is a property of the
    case, and it is visible in the returned ``data_sse`` profile: on
    ``burgers``, ``advz-theta`` and ``np`` the data fit of the constrained
    state has a clear interior minimum at the generating parameter, while on
    the Fisher-KPP case it decreases monotonically towards small ``theta``
    (a front makes ``Delta u`` a function of ``u``, so a free ``f`` absorbs the
    diffusion term for *any* ``theta`` and the residual falls to 1e-7
    everywhere).  The estimator then returns the lower end of the bracket.  No
    setting of ``lam``, ``mu`` or the sieve size changes that -- it is the
    non-identifiability of the semiparametric model, not a tuning failure.

    ``log_lo``/``log_hi`` bracket the parameter search, in ``log10`` units.
    They are not a prior on the value -- they are the range of magnitudes the
    cases in this codebase live in, and the search refines inside them.

    The test curves get the same cascade with ``theta`` and ``beta`` fixed: the
    inner criterion is exactly what a parametric method applies to a new
    trajectory, and holding ``beta`` is what a shared mechanism means.
    """
    torch.manual_seed(int(seed))
    grid = config.grid
    t = split.train.t.detach().cpu().numpy()
    t0, t1 = float(t.min()), float(t.max())
    phi_np = state_design(grid, n_modes)
    psi_obs_np, _ = bspline_design(t, t0, t1, n_basis_t)
    n_col = int(collocation_factor) * (len(t) - 1) + 1
    t_col = np.linspace(t0, t1, n_col)
    psi_col_np, psi_col_d_np = bspline_design(t_col, t0, t1, n_basis_t)

    phi = torch.as_tensor(phi_np, dtype=torch.float64)
    psi_obs = torch.as_tensor(psi_obs_np, dtype=torch.float64)
    psi_col = torch.as_tensor(psi_col_np, dtype=torch.float64)
    psi_col_d = torch.as_tensor(psi_col_d_np, dtype=torch.float64)

    # The mechanism sieve's window is measured on the observed training states.
    feats = _observed_features(spec, split.train.states, split.train.z)
    sieve = SieveBasis(
        lo=feats.amin(dim=0).detach(), hi=feats.amax(dim=0).detach(),
        n_basis=n_basis_f, degree=f_degree, dtype=torch.float64,
    )
    def make_inner(curves):
        states, _ = _flatten_curves(curves.states)
        return _Inner(
            spec, grid, t_col, phi, psi_obs, psi_col, psi_col_d, states,
            curves.z, sieve, lam, mu, inner_iters, f_iters, rounds,
        )

    train_inner = make_inner(split.train)
    keys = list(spec.truth().keys())
    x, coeffs, beta, data, evaluations = _outer_search(
        train_inner, keys, log_lo=log_lo, log_hi=log_hi, coarse=coarse,
        polish_maxfev=polish_maxfev, polish_maxiter=polish_maxiter,
    )
    theta = {k: float(v) for k, v in zip(keys, 10.0 ** np.asarray(x))}

    # The held-out curves get the same cascade with ``(theta, beta)`` fixed:
    # the state basis is fitted to each new trajectory, the mechanism is not.
    test_inner = make_inner(split.test)
    theta_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in theta.items()}
    c_test, _, _, residual = test_inner.solve(
        theta_t, None, beta, iters=max(inner_iters, 300), fit_beta=False)
    U = torch.einsum("mdkj,pk,tj->mdtp", c_test, phi, psi_obs)
    n, d_y, n_obs, P = U.shape
    prediction = U.permute(0, 2, 1, 3).reshape(
        n, n_obs, d_y, *grid.shape
    ).contiguous()

    if verbose:
        print(f"  cascading: theta={theta} data_sse={data:.4e} "
              f"beta_norm={float(beta.norm()):.3e} evaluations={evaluations}",
              flush=True)
    return CascadingFit(
        theta=theta, prediction=prediction, sieve=sieve, beta=beta,
        lam=float(lam), mu=float(mu), n_modes=int(n_modes),
        n_basis_t=int(n_basis_t), data_sse=float(data),
        residual=float(residual), outer_evaluations=int(evaluations),
    )
