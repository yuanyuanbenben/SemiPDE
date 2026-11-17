"""Estimation errors, coverage and the summaries the paper's tables report.

There are three different "errors" in play, and the paper's tables use all of
them, so each gets an explicit name here:

``theta_error``
    ``||theta_hat - theta_0||_2`` -- the second block of Table
    ``tab:simu:compared_baseline``.

``l2_error``
    ``||u_hat - u_0||_{L2}`` on the test trajectories, with the grid measure.
    The first block of the same table.

``relative_l2``
    The same divided by ``||u_0||_{L2}``.  Useful when comparing cases whose
    solution scales differ (Case 1 lives in ``[0, 1]``, Case 4 has velocities of
    size ``10^2``), which the table's shared ``x 10^{-2}`` column cannot do.

For inference, ``bias``/``std`` are *empirical* across repeated experiments,
while ``mean_se`` comes from Algorithm 2.  The paper reports the first two and,
in the commented-out version of Table ``tab:simu:para``, the third as ``sigma_hat``;
it is the agreement between ``std`` and ``mean_se`` that validates the
covariance estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import CaseConfig
from .data import Trajectories, Split

__all__ = [
    "cell_measure",
    "l2_error",
    "relative_l2",
    "theta_error",
    "flat_param",
    "BiasStd",
    "bias_std",
    "coverage_rate",
    "interval_report",
    "studentized",
    "qq_points",
    "EstimationSummary",
    "summarise_repetitions",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
def cell_measure(config: CaseConfig) -> float:
    """Volume of one grid cell; the quadrature weight for ``L2`` norms."""
    out = 1.0
    for h in config.grid.spacing:
        out *= float(h)
    return out


def l2_error(
    pred: np.ndarray,
    target: np.ndarray,
    config: Optional[CaseConfig] = None,
    measure: Optional[float] = None,
) -> float:
    """``||pred - target||_{L2}`` over the last axes, averaged over samples.

    ``pred`` and ``target`` are ``(n, *grid)`` or ``(*grid,)``.  The norm is
    taken as ``sqrt(measure * sum (diff)^2 / n)`` where ``measure`` is the cell
    volume, so that time samples and trajectories are averaged rather than
    summed.  Passing ``measure=1`` gives the plain RMS.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if measure is None:
        measure = cell_measure(config) if config is not None else 1.0
    if p.ndim == 1:
        p, t = p[None, :], t[None, :]
    diff2 = (p - t) ** 2
    return float(math.sqrt(measure * diff2.mean(axis=tuple(range(1, diff2.ndim))).mean()))


def relative_l2(pred, target, config=None, measure=None) -> float:
    denom = l2_error(np.zeros_like(np.asarray(target)), target, config, measure)
    if denom == 0.0:
        return float("nan")
    return l2_error(pred, target, config, measure) / denom


def flat_param(value) -> List[float]:
    """One ``theta`` entry as a flat list of floats.

    Case 4's ``theta_1`` and ``theta_2`` are 3-vectors, so an entry is not
    always a scalar and ``float(entry)`` is not always defined.  Flattening here
    rather than at every call site keeps the scalar cases reading exactly as
    they did -- ``flat_param(3e-3) == [3e-3]``.
    """
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, (int, float)) or isinstance(value, np.floating):
        return [float(value)]
    return [float(x) for x in value.reshape(-1)]


def theta_error(theta_hat: Dict[str, object], theta_0: Dict[str, object]) -> float:
    """``||theta_hat - theta_0||_2`` over every coordinate; keys must agree.

    An estimate that is ``nan`` on a *vector* parameter is read as "this method
    estimates no parameter" rather than as a shape error: Baseline 2 reports
    ``nan`` by construction, and for a 3-vector entry it has one ``nan`` where
    the truth has three coordinates.
    """
    missing = set(theta_0) - set(theta_hat)
    if missing:
        raise KeyError(f"theta_hat is missing {sorted(missing)}")
    total = 0.0
    for k in theta_0:
        want = flat_param(theta_0[k])
        got = flat_param(theta_hat[k])
        if len(got) != len(want):
            if len(got) == 1 and not math.isfinite(got[0]):
                return float("nan")
            raise ValueError(
                f"theta_hat[{k!r}] has {len(got)} values against {len(want)}"
            )
        total += sum((x - y) ** 2 for x, y in zip(got, want))
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# repeated-experiment summaries
# ---------------------------------------------------------------------------
@dataclass
class BiasStd:
    """Bias and standard deviation of an estimator over repeated experiments."""

    mean: float
    bias: float
    std: float
    rmse: float
    n: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "bias": self.bias,
            "std": self.std,
            "rmse": self.rmse,
            "n": float(self.n),
        }


def bias_std(values: Sequence[float], truth: float) -> BiasStd:
    """Bias/Std of ``values`` against ``truth``.

    The paper's convention: ``Bias = mean(hat) - truth`` and ``Std`` is the
    sample standard deviation using the ``n - 1`` denominator, computed around
    the *mean*, not around the truth (so ``rmse^2 = bias^2 + std^2 * (n-1)/n``
    only to first order).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = v.size
    if n == 0:
        raise ValueError("no repetitions to summarise")
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    bias = mean - float(truth)
    rmse = float(math.sqrt(np.mean((v - float(truth)) ** 2)))
    return BiasStd(mean=mean, bias=bias, std=std, rmse=rmse, n=n)


def coverage_rate(
    lower: Sequence[float],
    upper: Sequence[float],
    truth: float,
) -> float:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return float(np.mean((lo <= truth) & (truth <= hi)))


def interval_report(
    estimates: Sequence[float],
    lowers: Sequence[float],
    uppers: Sequence[float],
    ses: Sequence[float],
    truth: float,
    alpha: float,
) -> Dict[str, float]:
    """One row of the inference table: bias, std, mean SE, length, coverage."""
    est = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(ses, dtype=np.float64)
    lo = np.asarray(lowers, dtype=np.float64)
    hi = np.asarray(uppers, dtype=np.float64)
    bs = bias_std(est, truth)
    return {
        "alpha": float(alpha),
        "nominal": 1.0 - float(alpha),
        "bias": bs.bias,
        "std": bs.std,
        # The two quantities the paper's commented-out column compares:
        # ``std`` is empirical, ``mean_se`` is what Algorithm 2 predicts.
        "mean_se": float(se.mean()),
        "sd_of_se": float(se.std(ddof=1)) if se.size > 1 else 0.0,
        "mean_length": float((hi - lo).mean()),
        "coverage": coverage_rate(lo, hi, truth),
        "n": float(est.size),
    }


def studentized(estimates: Sequence[float], ses: Sequence[float], truth: float):
    """``(hat - truth) / se``, the pivot whose distribution should be N(0, 1)."""
    est = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(ses, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (est - float(truth)) / se


def qq_points(samples: Sequence[float]):
    """Empirical quantiles versus standard normal quantiles.

    Returns ``(theoretical, empirical)``; the caller plots them (and the paper's
    Figure ``fig:fig_qqplot`` shows the histogram of ``theta_hat - theta_0``).
    """
    s = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    n = s.size
    if n == 0:
        raise ValueError("no samples")
    probs = (np.arange(1, n + 1) - 0.5) / n
    # Inverse standard normal, vectorised, via the same rational approximation
    # used by semipde.inference.normal_quantile.
    from .inference import normal_quantile

    theo = np.array([normal_quantile(float(p)) for p in probs])
    return theo, s


# ---------------------------------------------------------------------------
# whole-experiment summary
# ---------------------------------------------------------------------------
@dataclass
class EstimationSummary:
    """Everything one row of each table needs, for one case and one setting."""

    case: str
    method: str
    theta: Dict[str, float] = field(default_factory=dict)
    u_error: float = float("nan")
    u_error_relative: float = float("nan")
    theta_dist: float = float("nan")
    f_error: float = float("nan")
    n_traj: int = 0
    n_total: int = 0
    repeats: int = 0
    extras: Dict[str, float] = field(default_factory=dict)

    def row(self) -> Dict[str, object]:
        return {
            "case": self.case,
            "method": self.method,
            "u_error": self.u_error,
            "u_error_relative": self.u_error_relative,
            "theta_error": self.theta_dist,
            "f_error": self.f_error,
            "n_traj": self.n_traj,
            "n_total": self.n_total,
            "repeats": self.repeats,
            **self.extras,
        }


def summarise_repetitions(
    rows: Sequence[EstimationSummary],
) -> List[Dict[str, object]]:
    """Average repeated runs and report mean/standard error of the mean."""
    out: List[Dict[str, object]] = []
    if not rows:
        return out
    keys = ("u_error", "u_error_relative", "theta_dist", "f_error")
    for key in keys:
        vals = np.array([getattr(r, key) for r in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        out.append(
            {
                "case": rows[0].case,
                "method": rows[0].method,
                "metric": key,
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                "sem": float(finite.std(ddof=1) / math.sqrt(finite.size))
                if finite.size > 1
                else 0.0,
                "repeats": int(finite.size),
            }
        )
    return out
