"""Feedforward networks, centred parameterisation and the penalty ``J``.

Implements Eq. (5) of the paper exactly::

    h^(0)(v)          = v
    h^(j)(v)          = sqrt(c_l / m_j) * rho_l{ W_{j-1} h^(j-1)(v) + b_{j-1} },
                        1 <= j <= L,
    f_tilde(v; phi)   = sqrt((l+1) / d_f) * { W_L h^(L)(v) + b_L },
    rho_l(t)          = t_+^l,          c_l = 2 / (2l - 1)!!,

together with the centred space ``f_phi = f_tilde(.; phi) - f_tilde(.; phi_0)``
and ``J(f_phi) = ||phi - phi_0||_2^2``.

Initialisation follows the paper (standard Gaussian weights, standard Gaussian
first-layer biases, zero elsewhere).  ``init="xavier"`` and ``normalize=False``
are provided only so that the legacy numbers can be reproduced bit for bit --
they are *not* what the theory assumes.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["double_factorial", "c_l", "FNN", "CenteredFNN", "DoubleFNN"]


def double_factorial(n: int) -> int:
    """``n!!`` for odd ``n >= 1``."""
    if n <= 0:
        return 1
    return n * double_factorial(n - 2)


def c_l(power: int) -> float:
    """``c_l = 2 / (2l - 1)!!``, so that ``c_l * E[G_+^{2l}] = 1``."""
    if power < 1:
        raise ValueError(f"power must be at least 1, got {power}")
    return 2.0 / double_factorial(2 * power - 1)


def _activation(name: str, power: int):
    name = name.lower()
    if name == "repu":
        return lambda t: torch.relu(t) ** power
    if name == "relu":
        return torch.relu
    if name == "tanh":
        return torch.tanh
    if name == "sine":
        return torch.sin
    raise ValueError(f"unknown activation {name!r}")


class FNN(nn.Module):
    """A feedforward network in the parameterisation of Eq. (5).

    Parameters
    ----------
    in_dim, out_dim
        ``m_0 = d`` and ``d_f``.
    hidden
        The hidden widths ``(m_1, ..., m_L)``; ``L = len(hidden)``.
    power
        The RePU power ``l``; ``l = 1`` is ReLU.
    normalize
        Apply ``sqrt(c_l / m_j)`` after each hidden activation and
        ``sqrt((l + 1) / d_f)`` at the output.  Disable only to reproduce the
        legacy implementation.
    init
        ``"gaussian"`` (the paper's scheme) or ``"xavier"`` (legacy).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
        power: int = 1,
        activation: str = "repu",
        normalize: bool = True,
        init: str = "gaussian",
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        #: Per-network seed.  ``None`` draws from the global RNG, so the driver
        #: controls reproducibility with a single ``torch.manual_seed``.
        self.seed = seed
        hidden = tuple(int(h) for h in hidden)
        if len(hidden) < 1:
            raise ValueError("hidden must contain at least one width")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = hidden
        self.power = int(power)
        self.normalize = bool(normalize)
        self.activation_name = activation
        self.init_scheme = init

        widths = [self.in_dim, *hidden, self.out_dim]
        self.widths = widths
        self.L = len(hidden)

        # F.linear expects (out_features, in_features).
        self.weight = nn.ParameterList(
            nn.Parameter(torch.empty(widths[j + 1], widths[j], dtype=dtype))
            for j in range(self.L + 1)
        )
        self.bias = nn.ParameterList(
            nn.Parameter(torch.empty(widths[j + 1], dtype=dtype))
            for j in range(self.L + 1)
        )

        self.reset_parameters()
        if device is not None:
            self.to(device)

        # Reference (initial) parameters for the centred parameterisation.
        self.register_buffer(
            "_ref_flat", torch.cat([p.detach().reshape(-1) for p in self.parameters()])
        )
        self._ref_slices: List[slice] = self._make_slices()

    # -- construction ----------------------------------------------------------
    def _make_slices(self) -> List[slice]:
        slices, start = [], 0
        for p in self.parameters():
            n = p.numel()
            slices.append(slice(start, start + n))
            start += n
        return slices

    @torch.no_grad()
    def reset_parameters(self) -> None:
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(self.seed))
        scheme = self.init_scheme
        # ``gaussian_zero_out`` is the paper's Gaussian scheme with the *output*
        # layer zeroed, so that ``f_{phi_0} = 0`` and, more importantly,
        # ``d f / d u = 0`` at the initial iterate.  The centred space already
        # makes the initial *value* zero, but its input-Jacobian is a random
        # O(1) field: harmless over the paper's ``[0, 2.5]`` window, fatal over
        # the 48 h of the cell-culture experiment, where the reverse-time
        # adjoint multiplies it by ``e^{48 * ||d f / d u||}`` and a single
        # backward pass reaches 1e46.  With the output layer at zero the first
        # drift is pure diffusion and the mechanism is learned from zero, which
        # is also where the physics starts (no reaction before the cells meet).
        if scheme == "gaussian_zero_out":
            scheme = "gaussian"
        for j, (W, b) in enumerate(zip(self.weight, self.bias)):
            if scheme == "gaussian":
                W.copy_(torch.randn(W.shape, generator=gen).to(W))
                if j == 0:
                    b.copy_(torch.randn(b.shape, generator=gen).to(b))
                else:
                    b.zero_()
            elif scheme == "xavier":
                nn.init.xavier_normal_(W)
                b.zero_()
            else:
                raise ValueError(f"unknown init scheme {self.init_scheme!r}")
        if self.init_scheme == "gaussian_zero_out":
            self.weight[self.L].zero_()
            self.bias[self.L].zero_()

    # -- forward ---------------------------------------------------------------
    def _raw(
        self,
        x: torch.Tensor,
        weight: Sequence[torch.Tensor],
        bias: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        h = x
        act = _activation(self.activation_name, self.power)
        c = c_l(self.power) if self.power >= 1 else 1.0
        for j in range(self.L):
            h = act(F.linear(h, weight[j], bias[j]))
            if self.normalize:
                h = h * math.sqrt(c / self.widths[j + 1])
        out = F.linear(h, weight[self.L], bias[self.L])
        if self.normalize:
            out = out * math.sqrt((self.power + 1) / self.out_dim)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``f_tilde(x; phi)`` --- the *uncentred* network."""
        return self._raw(x, list(self.weight), list(self.bias))

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """``f_tilde(x; phi_0)``, held fixed throughout estimation."""
        weight, bias = self._split_reference()
        return self._raw(x, weight, bias)

    def _split_reference(self):
        flat = self._ref_flat
        weight, bias = [], []
        params = list(self.weight) + list(self.bias)
        # parameters() is ordered as (W_0, ..., W_L, b_0, ..., b_L).
        shapes = [p.shape for p in params]
        pieces = [flat[s].reshape(sh) for s, sh in zip(self._ref_slices, shapes)]
        nW = self.L + 1
        weight = pieces[:nW]
        bias = pieces[nW:]
        return weight, bias

    # -- centred parameterisation ---------------------------------------------
    def reference_is_zero(self) -> bool:
        """Whether ``f_tilde(., phi_0)`` is identically zero.

        True under ``gaussian_zero_out``, whose whole point is a zeroed output
        layer, and after any :meth:`snapshot` that leaves it zero.  The
        reference pass is then exactly ``0``, so skipping it saves a full
        network evaluation per drift call without changing a single float:
        subtracting exact zeros is a no-op.
        """
        flat = self._ref_flat
        nW = self.L + 1
        last_w, last_b = self._ref_slices[nW - 1], self._ref_slices[2 * nW - 1]
        return not (torch.count_nonzero(flat[last_w]) or torch.count_nonzero(flat[last_b]))

    def forward_centered(self, x: torch.Tensor) -> torch.Tensor:
        """``f_phi(x) = f_tilde(x; phi) - f_tilde(x; phi_0)``."""
        out = self.forward(x)
        if self.reference_is_zero():
            return out
        return out - self.reference(x).detach()

    def param_distance(self) -> torch.Tensor:
        """``J(f_phi) = ||phi - phi_0||_2^2``."""
        out = torch.zeros((), dtype=self.weight[0].dtype, device=self.weight[0].device)
        for p, s in zip(self.parameters(), self._ref_slices):
            d = (p.reshape(-1) - self._ref_flat[s]).to(out.dtype)
            out = out + d.pow(2).sum()
        return out

    @torch.no_grad()
    def snapshot(self) -> None:
        """Re-centre: make the current parameters the new reference."""
        self._ref_flat.copy_(
            torch.cat([p.detach().reshape(-1) for p in self.parameters()])
        )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in={self.in_dim}, hidden={self.hidden}, out={self.out_dim}, "
            f"l={self.power}, act={self.activation_name}"
        )


class CenteredFNN(nn.Module):
    """The centred function space ``S_n = {f_phi - f_{phi_0}}``.

    Wraps a single :class:`FNN`.  ``forward`` evaluates ``f_phi`` (centred),
    ``penalty`` evaluates ``J``.
    """

    def __init__(self, fnn: FNN) -> None:
        super().__init__()
        self.fnn = fnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fnn.forward_centered(x)

    def penalty(self) -> torch.Tensor:
        return self.fnn.param_distance()

    @property
    def parameters_flat(self) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in self.fnn.parameters()])


class DoubleFNN(nn.Module):
    """Two independent :class:`FNN` blocks evaluated side by side.

    Used by the two-field cases (FitzHugh--Nagumo, Navier--Stokes), where the
    unknown mechanism is ``(f_1(u, w), f_2(u, w))``: both blocks read the whole
    state vector and one block supplies each output component.
    """

    def __init__(self, blocks: Sequence[FNN]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b.forward_centered(x) for b in self.blocks], dim=1)

    def penalty(self) -> torch.Tensor:
        out = torch.zeros((), dtype=next(self.parameters()).dtype)
        for b in self.blocks:
            out = out + b.param_distance()
        return out
