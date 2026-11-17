"""Spatial and temporal grids.

The temporal convention is the one place where "how big is the sample" is
decided, so it is spelled out explicitly here rather than being buried in a
data generator.

Convention
----------
``n_obs`` is the number of *observation times* of a single trajectory, which is
the quantity the paper calls ``n_t``.  With ``include_endpoint=True`` the times
are ``linspace(t0, t1, n_obs)``; with ``include_endpoint=False`` they are
``t0 + arange(n_obs) * (t1 - t0) / n_obs``.

The legacy code generated ``sample_size + 1`` times at spacing
``(t1 - t0) / sample_size``, i.e. 26 times for ``n_t = 25``.  Reproduce that
with ``TimeGrid(0.0, 2.5, 26, include_endpoint=True)``.  Use
``TimeGrid(0.0, 2.5, 25, include_endpoint=True)`` to match the paper's report
of ``n = n_t * n_x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch

Ranges = Sequence[Tuple[float, float]]


@dataclass(frozen=True)
class Grid:
    """A uniform Cartesian grid on a product of intervals."""

    ranges: Tuple[Tuple[float, float], ...]
    shape: Tuple[int, ...]
    periodic: bool = True

    def __post_init__(self) -> None:
        if len(self.ranges) != len(self.shape):
            raise ValueError(
                f"ranges has {len(self.ranges)} entries but shape has "
                f"{len(self.shape)}"
            )
        if any(n <= 0 for n in self.shape):
            raise ValueError(f"non-positive grid shape {self.shape}")

    @classmethod
    def from_ranges(
        cls, ranges: Ranges, shape: Sequence[int], periodic: bool = True
    ) -> "Grid":
        return cls(tuple(tuple(r) for r in ranges), tuple(int(n) for n in shape), periodic)

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def spacing(self) -> Tuple[float, ...]:
        return tuple(
            (float(r[1]) - float(r[0])) / n for r, n in zip(self.ranges, self.shape)
        )

    @property
    def size(self) -> int:
        """Number of grid points (the product of the shape)."""
        n = 1
        for s in self.shape:
            n *= s
        return n

    def axes(self) -> Tuple[np.ndarray, ...]:
        """Cell-centred coordinates, matching the legacy ``(2i-1)/n - 1`` rule.

        For the interval ``[a, b]`` with ``n`` points this is
        ``a + (i + 0.5) * (b - a) / n``, which for ``[-1, 1]`` gives
        ``(2i + 1) / n - 1`` for ``i = 0, ..., n - 1``.
        """
        out = []
        for (a, b), n in zip(self.ranges, self.shape):
            h = (b - a) / n
            out.append(a + (np.arange(n) + 0.5) * h)
        return tuple(out)

    def mesh(self, *, indexing: str = "ij") -> Tuple[np.ndarray, ...]:
        return np.meshgrid(*self.axes(), indexing=indexing)

    def to(self, device: str | torch.device) -> "Grid":
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        rr = ", ".join(f"[{a}, {b}]" for a, b in self.ranges)
        return f"Grid({rr}; {self.shape}; periodic={self.periodic})"


@dataclass(frozen=True)
class TimeGrid:
    """Uniform observation times on ``[t0, t1]``."""

    t0: float
    t1: float
    n_obs: int
    include_endpoint: bool = True

    def __post_init__(self) -> None:
        if self.n_obs < 2:
            raise ValueError(f"n_obs must be at least 2, got {self.n_obs}")
        if not self.t1 > self.t0:
            raise ValueError(f"need t1 > t0, got t0={self.t0}, t1={self.t1}")

    @property
    def span(self) -> float:
        return self.t1 - self.t0

    @property
    def dt(self) -> float:
        """Spacing between consecutive observation times."""
        if self.include_endpoint:
            return self.span / (self.n_obs - 1)
        return self.span / self.n_obs

    def points(self, *, dtype=np.float64) -> np.ndarray:
        if self.include_endpoint:
            return np.linspace(self.t0, self.t1, self.n_obs, dtype=dtype)
        return self.t0 + np.arange(self.n_obs, dtype=dtype) * self.span / self.n_obs

    def tensor(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        t = torch.as_tensor(self.points(), dtype=dtype)
        return t.to(device) if device is not None else t

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TimeGrid([{self.t0}, {self.t1}]; n_obs={self.n_obs}, "
            f"dt={self.dt:.6g}, endpoint={self.include_endpoint})"
        )
