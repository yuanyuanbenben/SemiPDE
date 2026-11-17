"""Size control: the sieve conditions that tie the network to ``n``.

The estimator of Eq. (3) minimises over ``F in S_n``, and ``S_n`` is indexed by
the network, so the network's size is a *statistical* tuning parameter rather
than an implementation detail.  The supplement states the constraint in the
Corollary ``repu_width_tolerance``: fix the depth and the activation power
``l >= 2``, let ``R_n <= C n^q`` be a deterministic radius dominating all
estimation, oracle and inference weight radii, and suppose
``lambda_n^{-1} = O(n^l)``.  Then any width

.. math::

    m \\ge n^{C_m},\\qquad
    C_m > \\max\\{8(A+q),\\ 4A+2(\\kappa+3)q,\\ 4(\\kappa+1)q,\\ 2l+2\\kappa q\\}

satisfies, with probability tending to one, the three conditions the
statistical arguments consume::

    R_n (omega_m(R_n) + sqrt(eps_m0))   = o(n^-A)
    R_n sqrt(eps_m,tan(R_n))            = o(n^-A)
    H_m(R_n)                            = o(lambda_n)

with ``eps_m,tan = eps_m0 + C omega_m + omega_m^2`` and
``H_m(R) <= C m^{-1/2}(1+R)^kappa (log m)^kappa``.  The radius restriction
``R(1+R)^kappa (log m)^kappa <= m^{1/4}`` is part of the statement, not a
consequence of it.

Three things about this are worth saying plainly, because together they decide
what code can and cannot do here.

``kappa`` and the constants are not numeric
    The paper carries ``C`` and ``kappa`` through as fixed-but-unspecified.
    The *exponents* are therefore implementable and the *widths* are not
    absolute: everything below is reported at the convention ``C = 1``, so
    :meth:`SieveSpec.max_radius` is the admissible radius up to that constant.
    For ``l >= 2`` the width exponent itself needs no constant and is exact.

The width condition is astronomically conservative
    At the default ``A = 1``, ``q = 1/2`` the corollary asks for ``C_m > 12``
    (RePU) or ``C_m > 48`` (ReLU), i.e. ``m >= n^{12}``.  No finite experiment
    satisfies that.  It is a *sufficient* condition with a deliberately huge
    slack, so "the corollary is violated" is the normal state of affairs and
    says nothing on its own.  What is informative is the next point.

The radius condition is measurable
    ``R_n`` is not free: it is ``sqrt(J(F))``, and ``J(F) = ||phi-phi_0||^2``
    is exactly what :meth:`~semipde.model.ProfilingModel.penalty` returns and
    what ``lambda_n`` penalises.  The estimator therefore reports its own
    radius at every validation check, and :meth:`SieveSpec.radius_ratio`
    compares it against the ball the deployed width admits.  This is the one
    part of the corollary that can be checked against a fitted model instead of
    assumed, and it is what :class:`SieveReport` leads with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

__all__ = ["SieveSpec", "SieveReport", "scaled_hidden"]


def _bisect_max_radius(kappa: float, log_m: float, limit: float, rhs: float) -> float:
    """Largest ``R >= 0`` with ``R (1+R)^kappa (log m)^kappa <= rhs``.

    The left-hand side is continuous and strictly increasing on ``R >= 0``, so
    bisection is exact to the bracket.  Returns ``0.0`` when even ``R = 0``
    fails, which happens when the width is too small for any nonzero radius.
    """
    f = lambda r: r * (1.0 + r) ** kappa * log_m ** kappa  # noqa: E731
    if f(0.0) > rhs:
        return 0.0
    lo, hi = 0.0, 1.0
    while f(hi) <= rhs and hi < limit:
        hi *= 2.0
    if f(hi) > rhs:
        hi = min(hi, limit)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) <= rhs:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class SieveSpec:
    """The free quantities of the corollary, plus the deployed architecture.

    Parameters
    ----------
    A
        The target rate ``o(n^{-A})`` of the three conditions.  The paper takes
        it "larger than the finite polynomial factors multiplying these errors
        in the score and covariance bounds", so it is a modelling choice; the
        default ``1`` is the smallest value for which the conditions say
        anything at all.
    q
        The exponent of the radius bound ``R_n <= C n^q``.
    kappa
        The unspecified tightness exponent.  Every result here is monotone in
        it, so it is worth reporting at more than one value.
    ell
        The penalty-rate exponent, ``lambda_n^{-1} = O(n^ell)``.
    power
        The activation power ``l``.  ``1`` is ReLU, for which the corollary
        does not apply verbatim -- see :meth:`width_exponent`.
    """

    A: float = 1.0
    q: float = 0.5
    kappa: float = 1.0
    ell: float = 0.5
    power: int = 1

    # -- the width condition -----------------------------------------------------
    @property
    def is_repu(self) -> bool:
        return int(self.power) >= 2

    def width_exponent(self) -> float:
        """The exponent ``C_m`` of the corollary.

        For ``l >= 2`` this is Eq. (``repu_common_width21``) verbatim and is
        exact.  For ReLU the corollary is not stated, so the value returned is
        the max of the exponents required by the two bounds the paper *does*
        state for ReLU,

        .. math::

            \\omega_m(R) \\le C m^{-1/6} R^{1/3}\\sqrt{\\log m},\\qquad
            \\epsilon_{m,0} \\le C m^{-1/16}(\\log m)^\\kappa,

        namely ``max{6A+8q, 32(A+q), 12A+14q}``.  The ``H_m`` condition is
        omitted because no ReLU bound for ``H_m`` appears in the paper; use
        :attr:`SieveReport.hessian_covered` to see which case you are in.
        """
        A, q, k = self.A, self.q, self.kappa
        if self.is_repu:
            return max(
                8.0 * (A + q),
                4.0 * A + 2.0 * (k + 3.0) * q,
                4.0 * (k + 1.0) * q,
                2.0 * self.ell + 2.0 * k * q,
            )
        return max(6.0 * A + 8.0 * q, 32.0 * (A + q), 12.0 * A + 14.0 * q)

    def required_width(self, n: int) -> float:
        """The sufficient width ``n^{C_m}``; astronomically large by design."""
        return float(n) ** self.width_exponent()

    # -- the radius condition ----------------------------------------------------
    def omega(self, radius: float, m: int) -> float:
        """The feature-change modulus ``omega_m(R)`` at the stated bound.

        ``C = 1``; see the module docstring on why no other convention is
        available.
        """
        lm = math.log(max(int(m), 2))
        if self.is_repu:
            return lm ** self.kappa * radius * (1.0 + radius) ** self.kappa / math.sqrt(m)
        return radius ** (1.0 / 3.0) * math.sqrt(lm) / m ** (1.0 / 6.0)

    def eps0(self, m: int) -> float:
        """``epsilon_{m,0}``: ``m^{-1/4}`` for RePU, ``m^{-1/16}`` for ReLU."""
        lm = math.log(max(int(m), 2))
        power = 0.25 if self.is_repu else 1.0 / 16.0
        return lm ** self.kappa / m ** power

    def eps_tan(self, radius: float, m: int) -> float:
        """``epsilon_{m,tan}(R) = eps_m0 + omega_m + omega_m^2`` (its own bound)."""
        w = self.omega(radius, m)
        return self.eps0(m) + w + w * w

    def hessian(self, radius: float, m: int) -> Optional[float]:
        """``H_m(R) <= C m^{-1/2}(1+R)^kappa (log m)^kappa``; ``None`` for ReLU."""
        if not self.is_repu:
            return None
        lm = math.log(max(int(m), 2))
        return lm ** self.kappa * (1.0 + radius) ** self.kappa / math.sqrt(m)

    def max_radius(self, m: int) -> float:
        """Largest ``R`` the radius restriction admits at width ``m``."""
        lm = math.log(max(int(m), 2))
        if self.is_repu:
            return _bisect_max_radius(self.kappa, lm, m ** 0.25, m ** 0.25)
        # R <= c sqrt(m) / (log m)^3, with the same C = c = 1 convention.
        return math.sqrt(m) / lm ** 3

    def radius_ratio(self, radius: float, m: int) -> float:
        """How far outside the admissible ball ``radius`` sits.  ``<= 1`` passes."""
        limit = self.max_radius(m)
        if limit <= 0.0:
            return math.inf
        return float(radius) / limit

    # -- the whole check ---------------------------------------------------------
    def check(
        self,
        n: int,
        m: int,
        radius: float,
        lambda_n: float = 0.0,
    ) -> "SieveReport":
        """Evaluate the three conditions at a *fitted* configuration."""
        w = self.omega(radius, m)
        e0 = self.eps0(m)
        target = float(n) ** (-self.A)
        c1 = radius * (w + math.sqrt(e0))
        c2 = radius * math.sqrt(self.eps_tan(radius, m))
        h = self.hessian(radius, m)
        # H_m = o(lambda_n): a ratio is the only scale-free way to say it, and
        # lambda_n = 0 makes the condition unsatisfiable rather than trivially
        # true -- zero is the one penalty weight for which nothing controls R_n.
        c3 = math.inf if h is None else (math.inf if lambda_n <= 0.0 else h / lambda_n)
        return SieveReport(
            spec=self,
            n=int(n),
            m=int(m),
            radius=float(radius),
            lambda_n=float(lambda_n),
            target=target,
            cond_radius=self.radius_ratio(radius, m),
            cond_omega_eps=c1,
            cond_tangent=c2,
            cond_hessian=c3,
            max_radius=self.max_radius(m),
            width_exponent=self.width_exponent(),
            required_width=self.required_width(n),
        )


@dataclass(frozen=True)
class SieveReport:
    """Outcome of :meth:`SieveSpec.check`, with the three conditions separated.

    ``cond_omega_eps`` and ``cond_tangent`` are absolute error sizes, so they
    are only comparable to ``target`` up to the unspecified constant in the
    bounds; ``cond_radius`` and ``cond_hessian`` are ratios and are exact.
    """

    spec: SieveSpec
    n: int
    m: int
    radius: float
    lambda_n: float
    target: float
    cond_radius: float
    cond_omega_eps: float
    cond_tangent: float
    cond_hessian: float
    max_radius: float
    width_exponent: float
    required_width: float

    @property
    def hessian_covered(self) -> bool:
        return self.spec.is_repu

    @property
    def radius_ok(self) -> bool:
        return self.cond_radius <= 1.0

    @property
    def ok(self) -> bool:
        """``True`` when every condition that is scale-free here passes."""
        return self.radius_ok and self.cond_hessian < 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "m": self.m,
            "R_n": self.radius,
            "max_R_n": self.max_radius,
            "radius_ratio": self.cond_radius,
            "omega_eps_term": self.cond_omega_eps,
            "tangent_term": self.cond_tangent,
            "hessian_over_lambda": self.cond_hessian,
            "target_n^-A": self.target,
            "width_exponent": self.width_exponent,
            "required_width": self.required_width,
        }

    def __str__(self) -> str:
        lines = [
            f"sieve: n={self.n}, m={self.m}, R_n={self.radius:.4g}",
            f"  radius      R_n/R_max = {self.cond_radius:.4g}"
            f"   ({'ok' if self.radius_ok else 'VIOLATED'}), R_max={self.max_radius:.4g}",
            f"  omega+eps   {self.cond_omega_eps:.3e}   vs n^-A = {self.target:.3e}",
            f"  tangent     {self.cond_tangent:.3e}   vs n^-A = {self.target:.3e}",
        ]
        if self.hessian_covered:
            lines.append(
                f"  hessian     H_m/lambda_n = {self.cond_hessian:.3e}"
                f"   ({'ok' if self.cond_hessian < 1.0 else 'VIOLATED'})"
            )
        else:
            lines.append("  hessian     not covered: the paper states no ReLU bound for H_m")
        # The sufficient width is a much stronger statement than the three
        # conditions, so say so where the number appears rather than leaving the
        # reader to supply the context.
        note = " (far beyond any attainable network)" if self.required_width > 1e6 else ""
        lines.append(
            f"  sufficient width: C_m = {self.width_exponent:.3g}, "
            f"m >= n^C_m = {self.required_width:.3e}{note}"
        )
        return "\n".join(lines)


def scaled_hidden(hidden: Sequence[int], m: int) -> Tuple[int, ...]:
    """Rescale the layer shape ``hidden`` so its widest layer has ``m`` units.

    Depth and the layer-to-layer ratios are kept, because the corollary fixes
    the depth and constrains only the width.  This is the knob that lets a run
    be placed on a width schedule at all: ``hidden`` held fixed for every ``n``
    is ``C_m = 0``, which fails the corollary's ``C_m > 0`` in every case, and
    that is precisely the state the published runs are in.
    """
    hidden = tuple(int(h) for h in hidden)
    if not hidden:
        raise ValueError("hidden must be non-empty")
    widest = max(hidden)
    if m < 1:
        raise ValueError(f"width must be positive, got {m}")
    scaled = tuple(max(1, int(round(h * m / widest))) for h in hidden)
    return scaled
