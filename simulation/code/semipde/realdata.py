"""The cell-culture real-data application (scratch assay, PC-3 cells).

The dataset is the supplementary material of Jin et al., *J. Theor. Biol.*
(2016), ``10.1016/j.jtbi.2015.10.040``: PC-3 prostate-cancer cell densities in
a scratch assay, six environmental settings with three repeated wells each,
five observation times over 48 h, a common set of 38 positions along the
monolayer.

The legacy driver (``realdata/realdata1/main.py``) assembled the 18 curves,
kept 15 for training and 3 for testing, and fitted two PDE models against
them:

==========  ==================================================================
Benchmark 1 ``d_t u = theta lap(u) + lambda u (1 - u/K)``  (Fisher--KPP)
Benchmark 2 ``d_t u = theta d_x((u/K) d_x u) + lambda u (1 - u/K)``
            (Porous--Fisher)
==========  ==================================================================

with Neumann boundaries.  The paper's SemiPDE 1/2 replace the logistic term by
an unknown ``f(u)`` and keep the two diffusion operators.  This module carries
the two diffusion operators (``variant="linear"`` / ``"porous"``) and the
unknown mechanism; the logistic is available through :class:`LogisticCell` for
the paper's parametric benchmark.

Units
-----
The paper reports the diffusivity in ``um^2/h`` -- ``1232`` for its SemiPDE 1 --
and the assay spans ``1.9`` mm with ``dx = 50`` um, so ``D * T / L^2`` is
``1232 * 48 / 2500 = 24`` and a scratch closes over the 48 h window
(``sqrt(1232 * 48) = 243`` um).  The length unit here is therefore ``50`` um and
the time unit the hour: the grid is ``[0, 38]`` with ``dx = 1``, the observation
times are ``0, ..., 48``, and ``D`` comes out at ``1232 / 2500 = 0.493``.

Writing the equation in mm instead would put ``D`` at ``1.2e-3``, and Adam --
which moves every coordinate by roughly its learning rate -- cannot both find
that value and refine it at a step the network's own weights tolerate.  The
state is rescaled by ``1e-3`` for the same reason (cell densities live near
``1e-3``; the unknown mechanism is then ``f(v) = O(1e-2)``), and the two
rescalings leave ``D`` and the time coordinate untouched.

``D`` is reported in ``um^2/h`` by multiplying by :data:`UNIT_UM` squared.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from . import operators as ops
from .config import CaseConfig
from .data import Split, Trajectories
from .grids import Grid, TimeGrid
from .pde import PDESpec, Theta

__all__ = [
    "TRAIN_INDEX",
    "TEST_INDEX",
    "load_cell_culture",
    "load_all",
    "replicate_sigma",
    "cell_case",
    "cell_split",
    "cell_trajectories",
    "theta_to_um2h",
    "CellCulture",
    "LogisticCell",
]

#: The legacy driver's split of the 18 curves (``main.py``): fifteen training
#: curves, three held out.  Kept so the numbers are comparable with the
#: application as it was run before; a 12/3/3 split with a validation set is
#: what :func:`cell_case` defaults to instead, because the estimator needs a
#: held-out selection set that is not the test set.
TRAIN_INDEX = (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17)
TEST_INDEX = (4, 9, 14)

#: Time points (hours) and positions of the assay: 38 positions, 50 um apart,
#: spanning 1.9 mm.
TIMES_H = (0.0, 12.0, 24.0, 36.0, 48.0)
UNIT_UM = 50.0
LENGTH = 38.0  # in units of UNIT_UM
N_X = 38

#: ``u = scale * v``: the state the network sees is ``v``, so that the fitted
#: mechanism lives at ``O(1e-2)`` instead of ``O(1e-5)``.
STATE_SCALE = 1e-3


def load_cell_culture(root: str = "realdata/realdata1") -> Tuple[np.ndarray, np.ndarray]:
    """The cached arrays the legacy driver wrote: ``(15, 5, 38)`` and ``(3, 5, 38)``.

    The ``.npy`` files are produced by ``main.py`` from the seven ``.xlsx``
    sheets; they are read here rather than re-parsing the spreadsheets because
    the workbook layout carries no information the arrays lack.
    """
    train = np.load(os.path.join(root, "data", "traindata.npy"))
    test = np.load(os.path.join(root, "data", "testdata.npy"))
    return train, test


def load_all(root: str = "realdata/realdata1") -> np.ndarray:
    """The 18 curves in the original order ``(condition, replicate)``.

    Rebuilt from the train/test arrays by inverting the legacy split, which is
    what makes the replicate structure available for a noise estimate: the six
    settings each have three wells, and the wells within a setting share their
    nominal initial condition.
    """
    train, test = load_cell_culture(root)
    total = np.empty((18, len(TIMES_H), N_X), dtype=np.float64)
    total[list(TRAIN_INDEX)] = train
    total[list(TEST_INDEX)] = test
    return total


def replicate_sigma(total: Optional[np.ndarray] = None,
                    root: str = "realdata/realdata1") -> float:
    """Within-setting replicate standard deviation, pooled over space and time.

    This is the only noise estimate the data support without a model: the three
    wells of a setting differ in observation noise *and* in their realised
    initial condition, so the number is an upper bound on the measurement
    noise.  It is reported for scale; the Wald interval estimates ``sigma`` from
    the residuals instead and never reads this.
    """
    total = load_all(root) if total is None else total
    groups = total.reshape(6, 3, len(TIMES_H), N_X)
    return float(np.sqrt(np.mean(np.var(groups, axis=1, ddof=1))))


#: Curves taken out of the legacy training set to form the selection set: one
#: replicate per setting, leaving twelve to fit on.  The three legacy test
#: curves are untouched, so the held-out number is the same one the application
#: was previously evaluated on.
VAL_INDEX = (2, 11, 17)


def cell_split(root: str = "realdata/realdata1",
               scale: float = STATE_SCALE) -> Split:
    """Twelve training curves, three for selection, the three legacy test curves.

    ``split_trajectories`` shuffles by seed; this one does not.  The replicates
    within a setting are exchangeable, so a shuffled split would give a number
    that depends on a seed without being more representative, and it would break
    the comparability with the legacy run, whose held-out curves are fixed.
    """
    total = load_all(root)
    train = [i for i in TRAIN_INDEX if i not in VAL_INDEX]
    return Split(
        train=cell_trajectories(total[train], scale=scale),
        val=cell_trajectories(total[list(VAL_INDEX)], scale=scale),
        test=cell_trajectories(total[list(TEST_INDEX)], scale=scale),
    )


def cell_trajectories(
    states: np.ndarray,
    *,
    scale: float = STATE_SCALE,
    sigma: Optional[float] = None,
) -> Trajectories:
    """Wrap ``(n, 5, 38)`` densities as a :class:`~semipde.data.Trajectories`.

    ``clean`` repeats ``states``: there is no noiseless reference for real
    data, and every scorer here reports the fit against the observations.
    ``noise_mask`` is ``False`` on the ``t = 0`` slice because that slice is the
    initial condition -- the residual there is zero by construction and
    averaging over it would deflate ``sigma_hat``.
    """
    v = torch.as_tensor(np.asarray(states, dtype=np.float64) / scale)
    v = v.unsqueeze(2)  # (n, T, 1, n_x)
    t = torch.as_tensor(TIMES_H, dtype=torch.float64)
    mask = torch.ones_like(v, dtype=torch.bool)
    mask[:, 0] = False
    sigma = 0.0 if sigma is None else float(sigma)
    return Trajectories(
        states=v,
        clean=v.clone(),
        z=torch.zeros(v.shape[0], dtype=torch.float64),
        t=t,
        initial=v[:, 0].clone(),
        sigma=sigma,
        noise_mask=mask,
    )


class CellCulture(PDESpec):
    """``d_t u = theta * (diffusion) + f(u)`` with Neumann boundaries.

    ``variant="linear"`` is the paper's SemiPDE 1 (``theta lap(u)``),
    ``"porous"`` its SemiPDE 2 (``theta d_x((u/K) d_x u)``).  The unknown
    mechanism reads the state only, so ``feature_dim = 1``; ``K`` enters the
    porous operator as the maximum monolayer density and is *not* estimated.

    There is no ground truth for real data, so :meth:`truth` is a *schema*
    carrying the initial guess, not a value to score against: ``D = 0.5``
    (``1250`` um^2/h) is the order of magnitude the legacy fit reports, and
    every method is started there through ``theta_init="truth"``.  Starting
    from zero is not an option here: Adam moves each coordinate by roughly its
    learning rate, and a network that learns at ``1e-3`` would need a thousand
    steps just to reach a diffusivity the data can distinguish from zero.
    Nothing in the study compares an estimate with this number.
    """

    name = "cell"

    def __init__(self, grid: Grid, variant: str = "linear", K: float = 1.7e-3,
                 theta_ref: float = 0.5, scale: float = STATE_SCALE) -> None:
        if variant not in ("linear", "porous"):
            raise ValueError(f"variant must be 'linear' or 'porous', got {variant!r}")
        super().__init__(grid)
        self.variant = variant
        self.K = float(K)
        self.theta_ref = float(theta_ref)
        self.scale = float(scale)

    # -- operators -------------------------------------------------------------
    def _diffusion(self, u: torch.Tensor, theta: Theta) -> torch.Tensor:
        D = theta["D"]
        if self.variant == "linear":
            return D * ops.laplacian(u, self._spacing, self._periodic)
        # ``d_x((u/K) d_x u)`` on the *physical* state: ``u`` here is already
        # the scaled ``v``, and ``v / K_scaled = u / K`` with
        # ``K_scaled = K / scale``.  The flux form is the conservative one --
        # ``divergence(coeff * gradient(u))`` leaks mass at the Neumann ends,
        # and the fit then inflates ``D`` to pay for the leak.
        return D * ops.flux_divergence(
            u / (self.K / self.scale), u, self._spacing, self._periodic
        )

    def drift(self, u, z, theta, f):
        rhs = self._diffusion(u, theta)
        if f is not None:
            rhs = rhs + f(u, z)
        return rhs

    def initial(self, rng: np.random.Generator, n: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            "the cell-culture data are observed, not generated: build a "
            "Trajectories with cell_trajectories()"
        )

    def truth(self) -> Theta:
        # Schema only -- see the class docstring.
        return {"D": torch.tensor(self.theta_ref, dtype=torch.float64)}

    def truth_f(self, u, z):
        """A *nominal* logistic mechanism, in the scaled state.

        There is no ``f_0`` for real data.  This exists because Baseline 1's
        wrapper reports its implied mechanism as the zero function *shaped
        like* this one, and because the spec-scoring helpers ask for a
        reference.  Nothing in the study compares a fitted ``f`` with it: the
        report quotes the fitted mechanism's own coefficients.
        """
        return 0.05 * u * (1.0 - u / (self.K / self.scale))


class LogisticCell(CellCulture):
    """Fisher--KPP / Porous--Fisher: the logistic mechanism is *known*.

    This is the paper's Benchmark 1/2.  ``theta = (D, lamda)`` is estimated and
    ``f`` is off in the sense of the unknown mechanism; the logistic term is
    added by :meth:`drift` from the estimated ``lamda`` so that the comparison
    is "same operator, parametric growth law" rather than "no growth".
    """

    def __init__(self, grid: Grid, variant: str = "linear", K: float = 1.7e-3,
                 theta_ref: float = 0.5, lamda_ref: float = 0.05,
                 scale: float = STATE_SCALE) -> None:
        super().__init__(grid, variant=variant, K=K, theta_ref=theta_ref,
                         scale=scale)
        self.lamda_ref = float(lamda_ref)

    def drift(self, u, z, theta, f):
        rhs = self._diffusion(u, theta)
        # ``f`` is None for this model by construction (the mechanism is
        # parametric), so the logistic is added here.
        return rhs + theta["lamda"] * u * (1.0 - u / (self.K / self.scale))

    def truth(self) -> Theta:
        return {
            "D": torch.tensor(self.theta_ref, dtype=torch.float64),
            "lamda": torch.tensor(self.lamda_ref, dtype=torch.float64),
        }


def theta_to_um2h(theta: float) -> float:
    """``D`` from the dimensionless units back to the paper's ``um^2/h``."""
    return float(theta) * UNIT_UM ** 2


def cell_case(
    *,
    n_train: int = 12,
    n_val: int = 3,
    n_test: int = 3,
    sigma: Optional[float] = None,
    variant: str = "linear",
    hidden: Sequence[int] = (16, 64, 64, 16),
    ode_step: float = 0.25,
    epochs: int = 200,
    init: str = "gaussian",
) -> Tuple[CellCulture, CaseConfig]:
    """The real-data case and a :class:`CaseConfig` sized for it.

    ``M`` is ``n_train`` -- the curves the estimator is fitted on -- and the
    held-out curves are split ``n_val``/``n_test``.  The 18 curves are only
    split by position in the legacy order, never shuffled: the replicates are
    interchangeable, and a random split would make the number depend on a seed
    for no gain.

    ``init`` defaults to the paper's scheme, but the 48 h window needs
    ``"gaussian_zero_out"``: the random initial network has an ``O(1)`` slope in
    ``u`` even though its value is zero, and the reverse-time adjoint of the
    rollout multiplies that slope by ``exp(48 * ||df/du||)`` -- gradient norms
    of ``1e46`` and a fit that diverges at epoch 149.  See docs/realdata_cell.md
    Sec. 2.1.
    """
    grid = Grid.from_ranges([(0.0, LENGTH)], [N_X], periodic=False)
    time = TimeGrid(TIMES_H[0], TIMES_H[-1], len(TIMES_H), include_endpoint=True)
    sigma = (replicate_sigma() if sigma is None else float(sigma)) / STATE_SCALE
    spec = CellCulture(grid, variant=variant)
    config = CaseConfig(
        name=f"cell-{variant}",
        grid=grid,
        time=time,
        d_y=1,
        M=int(n_train),
        n_val=int(n_val),
        n_test=int(n_test),
        sigma=sigma,
        hidden=tuple(hidden),
        ode_step=float(ode_step),
        epochs=int(epochs),
        theta_init="truth",
        init=str(init),
    )
    return spec, config
