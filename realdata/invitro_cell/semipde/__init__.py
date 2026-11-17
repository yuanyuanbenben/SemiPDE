"""SemiPDE: penalized profiling M-estimation for semiparametric PDE models.

Package layout
--------------
``semipde.config``     case configuration; the single source of truth for the
                       notion of "sample size" ``n`` and the repetition count ``M``.
``semipde.grids``      spatial / temporal grids.
``semipde.operators``  known differential operators as circular convolution kernels.
``semipde.networks``   feedforward networks (RePU / ReLU / sine), centred
                       parameterisation and the penalty ``J`` of the paper.
``semipde.data``       deterministic dataset generation (supports an input ``Z``).
``semipde.model``      the profiling model: known operator + nonparametric part,
                       integrated with an explicit ODE solver.
``semipde.train``      Algorithm 1 (gradient flow with the paper's stopping rule).
``semipde.inference``  Algorithm 2 (efficient covariance and Wald intervals).
``semipde.baselines``  Baselines 1--3 of the paper, plus a spline/Fourier baseline.
``semipde.metrics``    estimation errors, coverage, Q-Q statistics.
``semipde.cases``      the published cases assembled into runnable specs.
"""

from .config import CaseConfig, Grid, TimeGrid, InputSpec
from .pde import PDESpec, ReactionDiffusion
from .data import Trajectories, generate, solve, split_trajectories
from .model import ProfilingModel
from . import (
    baselines,
    cases,
    data,
    grids,
    inference,
    metrics,
    model,
    networks,
    operators,
    pde,
    train,
)

__all__ = [
    "CaseConfig",
    "Grid",
    "TimeGrid",
    "InputSpec",
    "PDESpec",
    "ReactionDiffusion",
    "Trajectories",
    "ProfilingModel",
    "baselines",
    "cases",
    "data",
    "generate",
    "grids",
    "inference",
    "metrics",
    "model",
    "networks",
    "operators",
    "pde",
    "solve",
    "split_trajectories",
    "train",
]

__version__ = "0.1.0"
