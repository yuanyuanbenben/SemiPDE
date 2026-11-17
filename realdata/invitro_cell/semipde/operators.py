"""Known differential operators as circular convolution kernels.

All operators act on tensors of shape ``(batch, d_y, *grid)`` and run
*depthwise* over the field axis, so a call with ``d_y > 1`` differentiates each
field independently.  This mirrors the legacy ``ParametricPart`` classes, which
split the channels by hand, but generalises to any spatial dimension and any
number of fields.

The stencils are the ones the paper records in the supplementary material:
``(1, -2, 1)`` for the 1D Laplacian, the 5-point star for 2D and the 7-point
star for 3D, and ``(-1/2, 0, 1/2)`` for the first derivative.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "d1",
    "d2",
    "laplacian",
    "gradient",
    "divergence",
    "flux_divergence",
    "advection",
    "Laplacian",
]


def _stencil_conv(
    x: torch.Tensor,
    axis: int,
    weights: torch.Tensor,
    denom: float,
    periodic: bool,
) -> torch.Tensor:
    """Convolve ``x`` along spatial ``axis`` with a 1D stencil.

    ``x`` has shape ``(batch, d_y, *grid)``; ``axis`` is indexed among the
    *spatial* axes, i.e. tensor dimension ``axis + 2``.  ``weights`` is given in
    the order ``(offset -1, 0, +1)`` and the result is divided by ``denom``
    (``h`` for a first derivative, ``h**2`` for a second one).
    """
    a = axis + 2
    if x.ndim <= a:
        raise ValueError(f"axis {axis} out of range for shape {tuple(x.shape)}")
    length = x.shape[a]
    y = x.movedim(a, -1)
    lead = y.shape[:-1]
    y = y.reshape(-1, 1, length)
    w = (weights.to(dtype=x.dtype, device=x.device) / denom).reshape(1, 1, 3)
    if periodic:
        y = F.pad(y, (1, 1), mode="circular")
    else:
        # Replicate the boundary value (Neumann on the stencil's outer samples).
        y = F.pad(y, (1, 1), mode="replicate")
    y = F.conv1d(y, w)
    y = y.reshape(*lead, length)
    return y.movedim(-1, a)


def d1(
    x: torch.Tensor,
    axis: int,
    spacing: float,
    periodic: bool = True,
) -> torch.Tensor:
    """First derivative along spatial ``axis``, central difference."""
    w = torch.tensor([-0.5, 0.0, 0.5])
    return _stencil_conv(x, axis, w, spacing, periodic)


def d2(
    x: torch.Tensor,
    axis: int,
    spacing: float,
    periodic: bool = True,
) -> torch.Tensor:
    """Second derivative along spatial ``axis``, three-point stencil."""
    w = torch.tensor([1.0, -2.0, 1.0])
    return _stencil_conv(x, axis, w, spacing * spacing, periodic)


def laplacian(
    x: torch.Tensor,
    spacing: Sequence[float],
    periodic: bool = True,
) -> torch.Tensor:
    """Sum of second derivatives over all spatial axes.

    ``spacing`` has one entry per spatial axis.  ``x`` is left unchanged; the
    sum is returned in a new tensor.
    """
    out = torch.zeros_like(x)
    for a, h in enumerate(spacing):
        out = out + d2(x, a, h, periodic)
    return out


class Laplacian(torch.nn.Module):
    """Callable Laplacian with a fixed grid spacing.

    Kept as a module so that PDE definitions can register it and so that a
    ``torch.compile``-friendly call site is available.
    """

    def __init__(self, spacing: Sequence[float], periodic: bool = True) -> None:
        super().__init__()
        self.spacing: Tuple[float, ...] = tuple(float(h) for h in spacing)
        self.periodic = bool(periodic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return laplacian(x, self.spacing, self.periodic)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"spacing={self.spacing}, periodic={self.periodic}"


def gradient(
    x: torch.Tensor,
    spacing: Sequence[float],
    periodic: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Central-difference gradient, one tensor per spatial axis.

    ``x`` is a scalar field ``(batch, 1, *grid)``; every returned tensor has the
    same shape.
    """
    return tuple(d1(x, a, h, periodic) for a, h in enumerate(spacing))


def divergence(
    flux: Sequence[torch.Tensor],
    spacing: Sequence[float],
    periodic: bool = True,
) -> torch.Tensor:
    """Divergence of a vector field given as one scalar field per axis."""
    if len(flux) != len(spacing):
        raise ValueError(
            f"flux has {len(flux)} components but spacing has {len(spacing)}"
        )
    out = torch.zeros_like(flux[0])
    for a, (f, h) in enumerate(zip(flux, spacing)):
        out = out + d1(f, a, h, periodic)
    return out


def flux_divergence(
    coeff: torch.Tensor,
    x: torch.Tensor,
    spacing: Sequence[float],
    periodic: bool = True,
) -> torch.Tensor:
    """``div(coeff * grad x)`` in conservative (face-centred) form.

    The naive composition ``divergence(coeff * gradient(x))`` is *not*
    conservative: central differences of the flux at the nodes, with the
    replicate padding used for the Neumann boundary, leave a spurious mass
    source of ``O(1)`` at the two end nodes.  For the cell-culture data that
    source is not small -- the fit pays for it by inflating ``D`` by a factor of
    three -- because a diffusion term that creates cells has to be compensated
    by a larger coefficient elsewhere.

    Here the flux is evaluated on the faces, ``F_{i+1/2} = c_{i+1/2}
    (x_{i+1} - x_i)/h`` with the coefficient averaged to the face, and the
    divergence is ``(F_{i+1/2} - F_{i-1/2})/h``.  The interior flux cancels
    telescopes exactly, and the two outer faces are set to zero (Neumann), so
    the operator conserves the total mass to machine precision.  At ``c = 1``
    it reduces to the three-point :func:`laplacian` with the same boundary
    stencil.
    """
    out = torch.zeros_like(x)
    for a, h in enumerate(spacing):
        ax = a + 2
        c = coeff.movedim(ax, -1)
        y = x.movedim(ax, -1)
        # Faces between consecutive nodes: ``i = 0, ..., N-2``.
        face = 0.5 * (c[..., 1:] + c[..., :-1]) * (y[..., 1:] - y[..., :-1]) / h
        if periodic:
            wrap = 0.5 * (c[..., :1] + c[..., -1:]) * (y[..., :1] - y[..., -1:]) / h
            flux = torch.cat([face, wrap], dim=-1)          # ``F_{i+1/2}``
            div = (flux - torch.roll(flux, 1, dims=-1)) / h
        else:
            zero = torch.zeros_like(face[..., :1])
            flux = torch.cat([zero, face, zero], dim=-1)    # ``F_{i-1/2}``, N + 1
            div = (flux[..., 1:] - flux[..., :-1]) / h
        out = out + div.movedim(-1, ax)
    return out


def advection(
    u: torch.Tensor,
    spacing: Sequence[float],
    periodic: bool = True,
) -> torch.Tensor:
    """``(u . grad) u`` for a vector field ``u`` of shape ``(batch, d, *grid)``.

    The velocity component ``u_a`` multiplies ``d/dx_a u_b``, so the result has
    the same shape as ``u``.
    """
    d = u.shape[1]
    comps = []
    for b in range(d):
        acc = torch.zeros_like(u[:, b : b + 1])
        for a in range(d):
            acc = acc + u[:, a : a + 1] * d1(u[:, b : b + 1], a, spacing[a], periodic)
        comps.append(acc)
    return torch.cat(comps, dim=1)
