"""PDE specifications.

A case is described by the *drift* form of Eq. (2) of the paper,

.. math::

    \\partial_t u_z = G_z(u_z;\\theta) - F_f(u_z),

with ``G`` collecting the known mechanism (the direction the parameter
``theta`` indexes) and ``F`` the shared unknown mechanism, assembled from a
function ``f``.  Writing the model as an explicit drift makes the sign
convention unambiguous and lets Baseline 1 be obtained by passing ``f = 0``
rather than by regenerating data.

Every spec carries

``d_y``      number of observed fields;
``drift``    ``(u, z, theta, f) -> du/dt`` with ``u`` of shape
             ``(batch, d_y, *grid)`` and ``z`` broadcastable to it;
``initial``  a sampler ``(rng, n) -> (n, d_y, *grid)`` for the initial data;
``truth``    the ground-truth ``theta`` and ``f`` used for scoring.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .grids import Grid
from . import operators as ops

Theta = Dict[str, torch.Tensor]
NuisanceFn = Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

#: The input blocks ``inner_inputs`` selects from, in the order
#: :meth:`PDESpec.inner_features` concatenates them.  All three are legitimate
#: under ``main.tex`` line 409 ("the inputs ``V_z`` of ``f_0`` are selected from
#: the location, state and state derivatives"); only the first two were
#: reachable before, and the third is the one that matters when a class's
#: representability floor is what sets an interval's width -- see
#: ``docs/size.md``.
_INNER_TOKENS = frozenset({"state", "location", "derivatives"})


def parse_inner_inputs(text: Optional[str]) -> Optional[str]:
    """Canonicalise an ``inner_inputs`` spelling, for use as an ``argparse`` type.

    Kept here rather than in a CLI so that the command line cannot admit a
    spelling :meth:`PDESpec.set_inner_inputs` would reject -- the two used to be
    separate lists, and the one that mattered (`--inner-inputs`) was the shorter
    of the two.  ``None``/``matched``/``state`` all mean "the estimator's own
    class", as in :meth:`set_inner_inputs`.
    """
    if text is None:
        return None
    text = str(text).strip()
    if text in ("", "none", "None", "matched", "state"):
        return None
    tokens = tuple(t.strip() for t in text.split(",") if t.strip())
    unknown = [t for t in tokens if t not in _INNER_TOKENS]
    if unknown:
        raise ValueError(
            f"inner_inputs takes a combination of {sorted(_INNER_TOKENS)} "
            f"(or 'matched'), got {text!r} (unrecognised: {unknown})"
        )
    return ",".join(tokens)


class PDESpec(ABC):
    """Base class for a SemiPDE case."""

    name: str = "unnamed"
    d_y: int = 1
    #: Number of pointwise features the unknown mechanism reads; see
    #: :meth:`features`.  For ``F_f(u)`` depending only on the state this is
    #: ``d_y``; a mechanism that also depends on the spatial coordinate (the
    #: shared forcing ``q(x)`` of the forced-Burgers case) asks for more.
    feature_dim: int = 1

    @property
    def mechanism_out_dim(self) -> int:
        """Components the unknown mechanism's network emits.

        Normally ``d_y`` -- one scalar field per observed field.  Case 3 is the
        exception: it observes one field and learns the two-component drift
        vector ``f(x)`` that enters ``-div(f(x) u)``.  :meth:`drift` is free to
        combine whatever it gets, so the two dimensions need not agree.
        """
        return self.d_y

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self._spacing = grid.spacing
        self._periodic = grid.periodic
        # ``coordinates`` is called from inside ``drift`` (the ``q(x)`` and
        # ``g(x)`` profiles read it), so it runs once per integration substep --
        # tens of thousands of times per reference solve.  Building the mesh and
        # copying it to the device each time dominated every run in this
        # codebase; the tensor is a read-only function of ``(batch, dtype,
        # device)``, so one cached entry per key is enough.
        self._coords_cache: Dict[Tuple[int, torch.dtype, str], torch.Tensor] = {}

    # -- model -----------------------------------------------------------------
    @abstractmethod
    def drift(
        self,
        u: torch.Tensor,
        z: torch.Tensor,
        theta: Theta,
        f: NuisanceFn,
    ) -> torch.Tensor:
        """Return ``du/dt`` for state ``u`` under input ``z``.

        ``f`` is ``None`` when the unknown mechanism is switched off
        (Baseline 1 and the ``F = 0`` experiment).
        """

    @abstractmethod
    def initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample ``n`` initial conditions, shape ``(n, d_y, *grid.shape)``."""

    # -- observation operator --------------------------------------------------
    #: Flat indices (into the trailing grid axes) of the points the data
    #: actually observe, or ``None`` for "the whole grid".
    #:
    #: The simulated cases observe every grid point, so the state the rollout
    #: returns *is* the sample.  An experiment that installs six thermocouples
    #: along a rod is the opposite: the PDE has to be integrated on the whole
    #: rod -- the unobserved stretch is a heat sink and sets the dynamics -- while
    #: the likelihood only sees the sensors.  Without the split, a full-grid
    #: residual would count the 10 unobserved nodes as 10 extra observations,
    #: inflating the ``n`` of Theorems 2--4 by that factor and deflating every
    #: interval by its square root.
    obs_index: Optional[Sequence[int]] = None

    def observe(self, u: torch.Tensor) -> torch.Tensor:
        """Project a predicted field onto the observed points.

        ``u`` is ``(..., d_y, *grid)``; the result keeps the leading axes and
        indexes the last one.  This is what :meth:`ProfilingModel.forward` hands
        back, so every loss and every Algorithm-2 residual is computed on the
        same points the data are.
        """
        if self.obs_index is None:
            return u
        return u[..., list(int(i) for i in self.obs_index)]

    # -- unknown-mechanism features --------------------------------------------
    def features(
        self,
        u: torch.Tensor,
        coordinates: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Pointwise inputs to the unknown mechanism.

        Parameters
        ----------
        u
            ``(B, d_y, *grid)``.
        coordinates
            ``(B, *grid, n_spatial)`` physical coordinates, already broadcast to
            the batch and expanded along the sample axes.
        z
            ``(B,)`` or broadcastable to ``(B, 1, ...)``.

        Returns
        -------
        ``(B, *grid, feature_dim)``.

        The default hands the state vector at each grid point to ``f``, i.e.
        ``F_f(u)(x) = f(u(x))``.  Specs whose nuisance mechanism also depends on
        ``x`` override this.
        """
        return u.movedim(1, -1)

    # -- Algorithm 2's inner class ---------------------------------------------
    #: How the inner ``h_j`` of Algorithm 2 reads the sample; ``None`` is the
    #: paper's construction (``h_j`` reads exactly what ``f_hat`` reads).  Set
    #: it through :meth:`set_inner_inputs` so the width stays in step.
    inner_inputs: Optional[str] = None
    #: Input width for the inner ``h_j``; ``None`` means "the estimator's own",
    #: i.e. :attr:`feature_dim`.
    inner_feature_dim: Optional[int] = None

    def set_inner_inputs(self, inner_inputs: Optional[str]) -> None:
        """Configure the inner class; see :meth:`inner_features`.

        Any combination of ``"location"``, ``"state"`` and ``"derivatives"`` is
        accepted, in any order; ``None`` -- spelled ``"matched"``, or
        ``"state"`` where that is what the matched class reads -- restores the
        class matched to the estimator, which is the paper's construction.

        ``"derivatives"`` adds every observed field's gradient and Laplacian, so
        ``h_j`` can read the *source* of the target it is trying to match
        instead of having to reproduce it.  On a diffusion-type drift that
        collapses the class's representability floor by three to four orders of
        magnitude, which is what makes the floor--ceiling window wide enough
        that ``tilde lambda_n`` can be *chosen* rather than merely bounded (see
        ``docs/size.md`` Sec. 5.3f).

        The width is computed here rather than by hand, so a spec cannot end up
        with a feature map and a network width that disagree -- that mismatch is
        silent, and the network simply stops reading the column it was built
        for.
        """
        if inner_inputs in (None, "matched", "state"):
            self.inner_inputs, self.inner_feature_dim = None, None
            return
        tokens = tuple(t.strip() for t in inner_inputs.split(",") if t.strip())
        if not tokens:
            # An empty request is not "no override" -- everything that means
            # that is spelled ``None``/``"matched"`` above -- and a class with no
            # inputs at all is not representable, so it is a mistake, not a
            # default.  ``parse_inner_inputs`` maps ``""`` here to ``None`` first.
            raise ValueError(
                f"inner_inputs must be None, 'matched', 'state', or a "
                f"combination of {sorted(_INNER_TOKENS)}, got {inner_inputs!r}"
            )
        unknown = [t for t in tokens if t not in _INNER_TOKENS]
        if unknown:
            raise ValueError(
                f"inner_inputs must be None, 'matched', 'state' or a "
                f"combination of {sorted(_INNER_TOKENS)}, got {inner_inputs!r} "
                f"(unrecognised: {unknown})"
            )
        self.inner_inputs = ",".join(tokens)
        n_loc = len(self.grid.shape)
        # The width is the sum of the blocks ``inner_features`` concatenates, in
        # that method's order; the two have to agree or the network silently
        # stops reading the trailing columns.
        width = 0
        if "state" in tokens:
            width += self.d_y
        if "location" in tokens:
            width += n_loc
        if "derivatives" in tokens:
            width += self.d_y * (n_loc + 1)  # the gradient, then the Laplacian
        self.inner_feature_dim = width

    def inner_features(
        self,
        u: torch.Tensor,
        coordinates: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Inputs Algorithm 2's ``h_j`` is allowed to read.

        ``None``/the default is :meth:`features` -- ``h_j`` reads exactly what
        the fitted mechanism reads, which is the paper's ``S_n`` and what every
        published number used.  No case ships widened: the one calibration that
        lands on nominal coverage is on Case 1's *supplement* initial profile,
        and on the legacy profile that ``tab:simu:para`` reports the same switch
        moves ``se`` the other way.  Widen explicitly, per configuration.

        A spec may widen this **on its own**, leaving the estimator alone.  That
        is a deliberate deviation from the paper's construction, so it is
        worth being exact about what it does and does not buy:

        The profile score is ``psi = d_theta l - d_f l[h_0]``, the chain rule for
        ``d_theta l(theta, f_hat(theta))``, so ``h_0 = d_theta f_hat(theta_0)``
        -- the sensitivity of the *fitted nuisance* to ``theta``.  Under the
        paper's ``S_n`` that direction lives in the estimator's own class, and
        the projection normal equations ``E[(d_theta u - d_f u[h_0])^T
        d_f u[h]] = 0`` fix it.  A nuisance reading only the state cannot move
        in a location-dependent way, so an ``h_j`` that *is* allowed to read the
        location minimises over directions the estimator can never take:
        ``M_hat = d_theta u - d_f u[h_hat]`` comes out smaller than the ``M_0``
        of the estimator actually being run, and ``Sigma_hat_eff =
        sigma_hat^2 (mean M_hat M_hat^T)^{-1}`` too **large**.  The interval is
        conservative, not exact, and it is not the paper's Proposition.

        It is also not identified: ``se`` rises with the width of the class and
        has no upper bound, while ``theta_hat`` -- which never sees ``h_j`` -- is
        unchanged.  Measured on Case 1 at ``M = 64``, ``n = 10240``, paired on
        the same fits (``scripts/probe_inner_class.py``):

        ==============  ====================  ====================
        inner class     supplement, 20 reps   legacy, 48 reps
        ==============  ====================  ====================
        ``None``        0.668                 0.434
        ``location``    1.072                 0.336
        both            2.061                 0.447
        ==============  ====================  ====================

        (``se / sd``; ``theta_hat`` is bit-identical across all six cells.)
        The two columns are two different statistical problems, and they do not
        agree even in *sign*: on the supplement's initial profile the widening
        buys the coverage the matched class was missing (C.R. 0.80/0.95/0.95
        against a matched 0.65/0.75/0.85), while on the legacy profile that
        ``tab:simu:para`` reports it makes ``se`` *smaller* (C.R.90 0.542 ->
        0.396) -- so the widening is a calibration for one configuration, not a
        correction that travels.  Handing ``h_j`` the state as well over-covers
        on the supplement profile, which is the non-identification showing up as
        an unbounded lever.  This is a class chosen to make the coverage right,
        not derived from the paper's ``Sigma_eff``, and it must be reported
        alongside any interval computed with it.  See ``docs/size.md``.
        """
        if self.inner_inputs is None:
            return self.features(u, coordinates, z)
        state = u.movedim(1, -1)
        # The block order is ``location, state, derivatives``: the first two are
        # in the order :meth:`features` uses (``cat([coordinates, state])``), so
        # a widened class whose tokens are a subset of the mechanism's own keeps
        # the column order every published run used.  Getting this backwards is
        # invisible under the exact solve -- a total-degree basis is invariant
        # under a permutation of its columns -- but builds a differently wired
        # inner *network*, which is the solve the paper itself specifies.
        #
        # ``derivatives`` is the gradient of every observed field followed by
        # its Laplacian, computed with the same stencils the drift uses; the
        # point of it is that ``h_j`` can then *take* the target's source rather
        # than have to imitate it (see the note on ``inner_inputs``).
        blocks = []
        if "location" in self.inner_inputs:
            blocks.append(coordinates)
        if "state" in self.inner_inputs:
            blocks.append(state)
        if "derivatives" in self.inner_inputs:
            # ``ops.gradient`` returns one field per spatial axis; they are
            # stacked along the field axis, which is how ``inner_feature_dim``
            # counts them (``d_y * n_loc`` gradient columns, then ``d_y``).
            grad = torch.cat(ops.gradient(u, self._spacing, self._periodic), dim=1)
            blocks.append(grad.movedim(1, -1))
            blocks.append(ops.laplacian(u, self._spacing, self._periodic).movedim(1, -1))
        return torch.cat(blocks, dim=-1)

    # -- scoring ---------------------------------------------------------------
    @abstractmethod
    def truth(self) -> Theta:
        """Ground-truth parameter values, keyed as in ``drift``."""

    def truth_f(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Ground-truth unknown mechanism, used for ``||f_hat - f_0||``."""
        raise NotImplementedError(f"{type(self).__name__} has no analytic f_0")

    def truth_u(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------
    def coordinates(
        self,
        batch: int,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        """Physical coordinates broadcast to ``(batch, *grid, n_spatial)``.

        Needed by specs whose unknown mechanism reads ``x`` rather than ``u``
        (Case 3, and the ``q(x)`` of forced Burgers), including inside
        :meth:`truth_f`, which only receives the state.
        """
        key = (int(batch), dtype, str(device))
        hit = self._coords_cache.get(key)
        if hit is not None:
            return hit
        stacked = np.stack(self.grid.mesh(), axis=-1)  # (*grid, n_spatial)
        out = torch.as_tensor(stacked, dtype=dtype)
        out = out.unsqueeze(0).expand(batch, *out.shape)
        if device is not None:
            out = out.to(device)
        self._coords_cache[key] = out
        return out

    @property
    def spacing(self) -> Sequence[float]:
        return self._spacing

    @property
    def periodic(self) -> bool:
        return self._periodic


# ---------------------------------------------------------------------------
# Case 1: reaction--diffusion (Example 1 of the paper, no auxiliary variable)
# ---------------------------------------------------------------------------
class ReactionDiffusion(PDESpec):
    """``d_t u = theta * laplacian(u) + f(u)`` on a periodic interval.

    The paper's Example 1 with ``f(u) = u (1 - u)``.  The paper prints
    ``theta = 3e-3``; the benchmark default is ``1e-2``.  ``f`` carries no
    theta, so a purely parametric fit (``f == 0``) can only compensate for the
    reaction by inflating theta and its absolute error is then roughly flat in
    ``theta_true`` -- measured 2e-2..4e-2 over the range tried -- which makes
    the comparison degenerate at the lower end and a tie at the upper one (see
    :func:`cases.case1_rd`).  Pass ``theta_true=3e-3`` to reproduce the paper,
    or ``3e-2`` for the setting this case carried on 2026-10-22/23.

    The initial condition is the one printed in the supplement,
    ``u(x, 0) = 0.5 sin(c1 |x| + c2) + 0.5`` with ``c1 ~ U(0.5, 1)`` and
    ``c2 ~ U(-1, 1)``.  The legacy generator used ``0.5 sin(c1 (1 - |x|) + c2) + 0.5``
    with ``c1 ~ U(1, 2)`` and ``c2 ~ U(0, 2pi)`` instead -- the paper's constants
    are still visible, commented out, in ``legacy/RD_dataset.py`` -- which makes
    the field wavier and changes the realised data.  Pass ``legacy_ic=True`` to
    reproduce the published tables; the default follows the paper.

    ``mechanism_inputs``
        Which inputs ``F_f`` -- and therefore ``h_j``, which perturbs ``F_f``
        -- is allowed to read.  ``"state"`` (the default) is the narrow
        selection ``F_f(u) = f(u(x))`` that the published Case-1 tables were
        produced with.  ``"location,state"`` widens it to ``f(x, u(x))``, the
        selection ``main.tex`` line 409 describes ("selected from the location,
        state and state derivatives") and the one ``ForcedBurgers`` and
        ``NavierStokes3D`` already use.

        The distinction is not cosmetic for the *variance*: a pointwise
        mechanism with the state-only selection admits exactly
        ``d_f u[h](x) = f'(u(x)) h(u(x))``, so the class available to ``h_j``
        is the functions of the **state**, while the target ``d_theta u`` is a
        field on ``(x, t, z)``.  Measured on the truth mechanism
        (``scripts/probe_class.py``), a state-only Chebyshev class explains
        15-24% of ``d_theta u``'s variance and saturates by degree 12, versus
        69-80% for a class carrying the coordinate -- which is why
        ``Sigma_eff`` comes out too small and the intervals too narrow for this
        case.  See ``docs/size.md``.
    """

    name = "rd"
    d_y = 1

    #: The legacy generator's constants; see the class docstring.
    LEGACY_C1 = (1.0, 2.0)
    LEGACY_C2 = (0.0, 2 * math.pi)

    def __init__(
        self,
        grid: Grid,
        theta_true: float = 1e-2,
        ic_c1: Sequence[float] = (0.5, 1.0),
        ic_c2: Sequence[float] = (-1.0, 1.0),
        legacy_ic: bool = False,
        mechanism_inputs: str = "state",
        inner_inputs: Optional[str] = None,
    ) -> None:
        super().__init__(grid)
        self.theta_true = float(theta_true)
        if legacy_ic:
            ic_c1, ic_c2 = self.LEGACY_C1, self.LEGACY_C2
        self.ic_c1 = tuple(ic_c1)
        self.ic_c2 = tuple(ic_c2)
        self.legacy_ic = bool(legacy_ic)
        if mechanism_inputs not in ("state", "location,state"):
            raise ValueError(
                f"mechanism_inputs must be 'state' or 'location,state', "
                f"got {mechanism_inputs!r}"
            )
        self.mechanism_inputs = mechanism_inputs
        # ``read`` from the spec rather than the class attribute, so the
        # widened selection is visible to the network builder.
        self.feature_dim = 1 + (1 if "location" in mechanism_inputs else 0)

        # An instance value is what makes the inner net read a *different* input
        # set from the fitted mechanism; see ``PDESpec.set_inner_inputs``.
        self.set_inner_inputs(inner_inputs)

    def features(self, u, coordinates, z):
        """``f``'s inputs: the state, plus the location if it was selected."""
        state = u.movedim(1, -1)
        if self.mechanism_inputs == "state":
            return state
        return torch.cat([coordinates, state], dim=-1)

    def drift(self, u, z, theta, f):
        lap = ops.laplacian(u, self._spacing, self._periodic)
        rhs = theta["theta"] * lap
        if f is not None:
            rhs = rhs + f(u, z)
        return rhs

    def initial(self, rng, n):
        shape = (n, self.d_y, *self.grid.shape)
        out = np.empty(shape, dtype=np.float64)
        coords = self.grid.mesh()  # one array per spatial axis
        for i in range(n):
            c1 = rng.uniform(*self.ic_c1)
            c2 = rng.uniform(*self.ic_c2)
            # ``1 - |x|`` toggles the mirror-symmetric argument the legacy
            # generator used; it is a different field, not a phase shift.
            profile = 1.0 - np.abs(coords[0]) if self.legacy_ic else np.abs(coords[0])
            arg = c1 * profile
            for extra in coords[1:]:
                arg = arg + c1 * (1.0 - np.abs(extra) if self.legacy_ic else np.abs(extra))
            out[i, 0] = 0.5 * np.sin(arg + c2) + 0.5
        return out

    def truth(self) -> Theta:
        return {"theta": torch.tensor(self.theta_true, dtype=torch.float64)}

    def truth_f(self, u, z):
        return u * (1.0 - u)

    def truth_u(self, u, z):  # pragma: no cover - not used for scoring
        return u


# ---------------------------------------------------------------------------
# Case 2: FitzHugh--Nagumo (two coupled reaction--diffusion equations)
# ---------------------------------------------------------------------------
class FitzHughNagumo(PDESpec):
    """``d_t u = theta_1 lap(u) + f_1(u, v)``, ``d_t v = theta_2 lap(v) + f_2(u, v)``.

    Simulation Case 2 of the paper: ``f_1 = u - u^3 - 5e-3 - v``,
    ``f_2 = u - v``, ``theta = (1e-3, 3e-3)``.  The excitation ``u - u^3 - 5e-3``
    is what makes the two fields genuinely coupled rather than two independent
    diffusions.
    """

    name = "fhn"
    d_y = 2
    feature_dim = 2  # the mechanism reads the whole state vector (u, v)

    def __init__(
        self,
        grid: Grid,
        theta_true: Sequence[float] = (1e-3, 3e-3),
        ic_slope: Sequence[float] = (1.0, 2.0),
        ic_phase: Sequence[float] = (0.0, 2 * np.pi),
        inner_inputs: Optional[str] = None,
    ) -> None:
        super().__init__(grid)
        if len(theta_true) != 2:
            raise ValueError("FitzHughNagumo needs two diffusivities")
        self.theta_true = tuple(float(t) for t in theta_true)
        self.ic_slope = tuple(ic_slope)
        self.ic_phase = tuple(ic_phase)
        # State-only mechanism, so the inner class can be widened to reach the
        # coordinate; see ``PDESpec.set_inner_inputs``.
        self.set_inner_inputs(inner_inputs)

    def _diffusivity(self, theta: Theta, u: torch.Tensor) -> torch.Tensor:
        stack = torch.stack(
            [
                torch.as_tensor(theta["theta1"], dtype=u.dtype, device=u.device),
                torch.as_tensor(theta["theta2"], dtype=u.dtype, device=u.device),
            ]
        )
        return stack.reshape(1, self.d_y, *([1] * (u.ndim - 2)))

    def drift(self, u, z, theta, f):
        rhs = self._diffusivity(theta, u) * ops.laplacian(
            u, self._spacing, self._periodic
        )
        if f is not None:
            rhs = rhs + f(u, z)
        return rhs

    def initial(self, rng, n):
        out = np.empty((n, self.d_y, *self.grid.shape), dtype=np.float64)
        mesh = self.grid.mesh()
        for i in range(n):
            for field in range(self.d_y):
                c1 = rng.uniform(*self.ic_slope)
                c2 = rng.uniform(*self.ic_slope)
                c3 = rng.uniform(*self.ic_phase)
                arg = c1 * (1.0 - np.abs(mesh[0])) + c2 * (1.0 - np.abs(mesh[1])) + c3
                out[i, field] = 0.5 * np.sin(arg) + 0.5
        return out

    def truth(self) -> Theta:
        return {
            "theta1": torch.tensor(self.theta_true[0], dtype=torch.float64),
            "theta2": torch.tensor(self.theta_true[1], dtype=torch.float64),
        }

    def truth_f(self, u, z):
        return self._reaction(u)

    @staticmethod
    def _reaction(u: torch.Tensor) -> torch.Tensor:
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        return torch.cat([u1 - u1 ** 3 - 5e-3 - u2, u1 - u2], dim=1)


# ---------------------------------------------------------------------------
# Case 3: Nernst--Planck
# ---------------------------------------------------------------------------
class NernstPlanck(PDESpec):
    """``d_t u = theta_1 lap(u) + theta_2 div(E u) - div(f(x) u)`` on ``[-1,1]^2``.

    Simulation Case 3.  ``E = (1 - u(2-|x|)/2, 1 - u(2-|y|)/2)`` is the known
    (electrostatic) drift and the unknown mechanism is the divergence of a
    *vector-valued* field ``f`` times the density:

    .. math:: \\mathcal F_f(u) = -\\nabla\\cdot(f(x)\\,u),
        \\quad f = \\bigl((1-|x|)^2+(1-|y|),\\ (1-|y|)^2+(1-|x|)\\bigr)/10 .

    Two consequences shape the implementation.  The mechanism reads the
    *coordinate*, not the state, so :meth:`features` is overridden; and it emits
    two components while only one field is observed, so
    :attr:`mechanism_out_dim` is 2 and :meth:`drift` is what turns the network
    output into the scalar right-hand side.
    """

    name = "np"
    d_y = 1
    feature_dim = 2  # the mechanism reads (x, y)

    def __init__(
        self,
        grid: Grid,
        theta_true: Sequence[float] = (7e-3, 5.33e-3),
        ic_slope: Sequence[float] = (1.0, 2.0),
        ic_phase: Sequence[float] = (0.0, 2 * np.pi),
        inner_inputs: Optional[str] = None,
    ) -> None:
        super().__init__(grid)
        if grid.dim != 2:
            raise ValueError("NernstPlanck is defined on a two-dimensional grid")
        self.theta_true = tuple(float(t) for t in theta_true)
        self.ic_slope = tuple(ic_slope)
        self.ic_phase = tuple(ic_phase)
        self.set_inner_inputs(inner_inputs)

    @property
    def mechanism_out_dim(self) -> int:
        return 2

    def features(self, u, coordinates, z):
        """The mechanism is ``f(x)``: hand it the coordinate, ignoring ``u``."""
        batch = u.shape[0]
        coords = coordinates
        if coords.shape[0] != batch:
            coords = self.coordinates(batch, u.dtype, u.device)
        return coords

    def _coordinate_axes(self, u: torch.Tensor) -> torch.Tensor:
        """Coordinates as ``(B, n_spatial, *grid)``.

        Needed whenever a coordinate multiplies a state: ``u`` is
        ``(B, d_y, *grid)`` while :meth:`coordinates` is ``(B, *grid, n_spatial)``,
        and broadcasting those two directly silently produces a tensor with an
        extra spatial axis instead of raising.
        """
        coords = self.coordinates(u.shape[0], u.dtype, u.device)
        return coords.movedim(-1, 1)

    def _source_vector(self, u: torch.Tensor) -> tuple:
        """The field ``f(x)`` of the paper, one component per axis.

        ``f_a = ((1-|x_a|)^2 + (1-|x_b|)) / 10`` with ``b`` the other axis, so
        each component mixes a squared term in its own coordinate with a linear
        term in the other.
        """
        coords = self._coordinate_axes(u)
        dec = [1.0 - coords[:, a : a + 1].abs() for a in range(self.grid.dim)]
        return tuple((dec[a] ** 2 + dec[1 - a]) / 10.0 for a in range(self.grid.dim))

    def _electric_field(self, u: torch.Tensor) -> list:
        """``E_a = 1 - u (2 - |x_a|) / 2``.  The known electrostatic drift."""
        coords = self._coordinate_axes(u)
        return [
            1.0 - u * (2.0 - coords[:, a : a + 1].abs()) / 2.0
            for a in range(self.grid.dim)
        ]

    def drift(self, u, z, theta, f):
        spacing = self._spacing
        rhs = theta["theta1"] * ops.laplacian(u, spacing, self._periodic)
        e_field = self._electric_field(u)
        rhs = rhs + theta["theta2"] * ops.divergence(
            [e * u for e in e_field], spacing, self._periodic
        )
        if f is not None:
            fvec = f(u, z)  # (B, 2, *grid)
            flux = [fvec[:, a : a + 1] * u for a in range(self.grid.dim)]
            rhs = rhs - ops.divergence(flux, spacing, self._periodic)
        return rhs

    def initial(self, rng, n):
        out = np.empty((n, self.d_y, *self.grid.shape), dtype=np.float64)
        mesh = self.grid.mesh()
        for i in range(n):
            c1 = rng.uniform(*self.ic_slope)
            c2 = rng.uniform(*self.ic_slope)
            c3 = rng.uniform(*self.ic_phase)
            arg = c1 * (1.0 - np.abs(mesh[0])) + c2 * (1.0 - np.abs(mesh[1])) + c3
            out[i, 0] = np.sin(arg) + 1.0
        return out

    def truth(self) -> Theta:
        return {
            "theta1": torch.tensor(self.theta_true[0], dtype=torch.float64),
            "theta2": torch.tensor(self.theta_true[1], dtype=torch.float64),
        }

    def truth_f(self, u, z):
        return torch.cat(self._source_vector(u), dim=1)


# ---------------------------------------------------------------------------
# Case 4: three-dimensional Navier--Stokes momentum equation
# ---------------------------------------------------------------------------
class NavierStokes3D(PDESpec):
    """Simulation Case 4, on ``[-1,1]^3`` with three velocity components.

    .. math::

        \\partial_t u = -10^{-3}\\,u\\cdot\\nabla u
            + \\theta_1\\{\\Delta u - \\tfrac13\\nabla(\\nabla\\cdot u)\\}
            - \\theta_2\\nabla p + f_1(x) + \\nabla f_2(x,u,\\nabla\\cdot u)

    with the pressure closure ``p = u_x(1-|x|) + u_y(1-|y|) + u_z(1-|z|)``.  The
    unknown mechanism is ``F = -f_1 - \\nabla f_2``, so the network reads the
    seven pointwise features ``(x, y, z, u_x, u_y, u_z, \\nabla\\cdot u)`` and
    emits three components.

    Note on the source: ``f_1`` is written out explicitly in the paper, yet it is
    part of ``F``, so it is *not* used in :meth:`drift` -- it appears only in
    :meth:`truth_f`, which is what the data are generated from.  The fifth
    component of the supplement's pressure formula is printed as ``1-|x|`` for
    the ``z`` term; we use ``1-|z|``, which is the only reading consistent with
    the stated pattern.
    """

    name = "ns3d"
    d_y = 3
    feature_dim = 7  # (x, y, z, u, div u)

    def __init__(
        self,
        grid: Grid,
        theta_true: Sequence[Sequence[float]] = ((6e-3, 7e-3, 8e-3), (3e-3, 4e-3, 5e-3)),
        advection_scale: float = 1e-3,
        ic_slope: Sequence[float] = (1.0, 2.0),
        ic_phase: Sequence[float] = (0.0, 2 * np.pi),
        inner_inputs: Optional[str] = None,
    ) -> None:
        super().__init__(grid)
        if grid.dim != 3:
            raise ValueError("NavierStokes3D is defined on a three-dimensional grid")
        self.theta_true = tuple(tuple(float(x) for x in row) for row in theta_true)
        self.advection_scale = float(advection_scale)
        self.ic_slope = tuple(ic_slope)
        self.ic_phase = tuple(ic_phase)
        self.set_inner_inputs(inner_inputs)

    @property
    def mechanism_out_dim(self) -> int:
        return 3

    def _divergence(self, u: torch.Tensor) -> torch.Tensor:
        return ops.divergence(
            [u[:, a : a + 1] for a in range(self.d_y)], self._spacing, self._periodic
        )

    def features(self, u, coordinates, z):
        batch = u.shape[0]
        coords = coordinates
        if coords.shape[0] != batch:
            coords = self.coordinates(batch, u.dtype, u.device)
        div_u = self._divergence(u).movedim(1, -1)  # (B, *grid, 1)
        return torch.cat([coords, u.movedim(1, -1), div_u], dim=-1)

    def _pressure(self, u: torch.Tensor) -> torch.Tensor:
        coords = self.coordinates(u.shape[0], u.dtype, u.device).movedim(-1, 1)
        p = None
        for a in range(self.d_y):
            term = u[:, a : a + 1] * (1.0 - coords[:, a : a + 1].abs())
            p = term if p is None else p + term
        return p

    def _vector(self, theta, key: str, u: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(theta[key], dtype=u.dtype, device=u.device).reshape(
            1, self.d_y, *([1] * (u.ndim - 2))
        )

    def drift(self, u, z, theta, f):
        spacing = self._spacing
        lap = ops.laplacian(u, spacing, self._periodic)
        div_u = self._divergence(u)
        grad_div = torch.cat(
            ops.gradient(div_u, spacing, self._periodic), dim=1
        )
        rhs = -self.advection_scale * ops.advection(u, spacing, self._periodic)
        rhs = rhs + self._vector(theta, "theta1", u) * (lap - grad_div / 3.0)
        grad_p = torch.cat(
            ops.gradient(self._pressure(u), spacing, self._periodic), dim=1
        )
        rhs = rhs - self._vector(theta, "theta2", u) * grad_p
        if f is not None:
            rhs = rhs + f(u, z)
        return rhs

    def initial(self, rng, n):
        out = np.empty((n, self.d_y, *self.grid.shape), dtype=np.float64)
        mesh = self.grid.mesh()
        for i in range(n):
            for field in range(self.d_y):
                arg = 0.0
                for axis in range(self.grid.dim):
                    arg = arg + rng.uniform(*self.ic_slope) * (
                        1.0 - np.abs(mesh[axis])
                    )
                arg = arg + rng.uniform(*self.ic_phase)
                out[i, field] = np.sin(arg) + 1.0
        return out

    def truth(self) -> Theta:
        return {
            "theta1": torch.tensor(self.theta_true[0], dtype=torch.float64),
            "theta2": torch.tensor(self.theta_true[1], dtype=torch.float64),
        }

    def truth_f(self, u, z):
        return -self._forcing(u) - self._grad_f2(u)

    def _forcing(self, u: torch.Tensor) -> torch.Tensor:
        coords = self.coordinates(u.shape[0], u.dtype, u.device)
        s = sum(coords[..., a : a + 1].abs() for a in range(self.grid.dim))
        profile = 0.1 * (torch.sin(2 * np.pi * s) + torch.cos(2 * np.pi * s))
        return profile.expand(-1, *([-1] * (u.ndim - 2)), self.d_y).movedim(-1, 1)

    def _grad_f2(self, u: torch.Tensor) -> torch.Tensor:
        coords = self.coordinates(u.shape[0], u.dtype, u.device)
        s = sum(coords[..., a : a + 1].abs() for a in range(self.grid.dim)).movedim(-1, 1)
        squared = sum(u[:, a : a + 1] ** 2 for a in range(self.d_y))
        f2 = (3.0 - s) * self._divergence(u) / (10.0 * (1.0 + squared))
        return torch.cat(ops.gradient(f2, self._spacing, self._periodic), dim=1)


# ---------------------------------------------------------------------------
# Example 2: forced viscous Burgers with an observed input
# ---------------------------------------------------------------------------
class ForcedBurgers(PDESpec):
    r"""``d_t u_z - nu d_xx u_z + u_z d_x u_z = theta z g(x) + q(x)`` on the circle.

    The paper's Example 2.  ``nu > 0`` and the forcing profile ``g`` are known,
    ``theta`` calibrates the controlled forcing, and ``q`` is a shared unknown
    background source that must be recovered from data at **both** input values.
    The input ``Z`` enters only through the known term ``theta Z g(x)`` -- it is
    not an argument of ``q``.

    Two structural points:

    * The unknown mechanism is ``q(x)``, a function of the *coordinate*, so
      :meth:`features` hands the network ``x`` rather than the state.
    * ``g`` is typically chosen to overlap ``q_0`` in at least one Fourier mode
      (here both contain a ``sin x`` component).  That overlap is the whole point
      of the design: on a single input the map
      ``(theta, q) -> (theta + c, q - c z_0 g)`` leaves every observation
      unchanged, so ``theta`` is identified only by the *difference* between the
      two forcing directions.  Sampling one input with probability 1 is therefore
      a genuine negative control, not a harder version of the same problem.
    """

    name = "burgers"
    d_y = 1
    feature_dim = 1  # the mechanism reads the coordinate x

    def __init__(
        self,
        grid: Grid,
        nu: float = 0.05,
        theta_true: float = 0.5,
        g_modes: Sequence[tuple] = ((1.0, "sin"),),
        q_modes: Sequence[tuple] = ((0.3, "sin"), (0.2, "cos2"), (0.1, "sin3")),
        ic_amplitude: float = 0.0,
        inner_inputs: Optional[str] = None,
    ) -> None:
        #: The defaults above are the paper's raw Example-2 numbers, transcribed
        #: as published.  They are *not* the runnable configuration: at
        #: ``nu = 0.05`` on the paper's ``[0, 2.5]`` window the solve runs away.
        #: ``cases.case_burgers`` is the rescaled set the experiments use, and
        #: its docstring derives the rescaling.
        super().__init__(grid)
        if grid.dim != 1:
            raise ValueError("ForcedBurgers is one-dimensional")
        self.nu = float(nu)
        self.theta_true = float(theta_true)
        self.g_modes = tuple(g_modes)
        self.q_modes = tuple(q_modes)
        #: ``u_z(0, x) = 0`` in the paper.  A nonzero value is available for
        #: experiments that need more than the two input levels to distinguish
        #: curves, but it is off by default so that the published setting is
        #: reproduced exactly.
        self.ic_amplitude = float(ic_amplitude)
        self.ic_slope = (0.5, 1.5)
        self.ic_phase = (0.0, 2 * np.pi)
        # ``f`` here is ``q(x)``, so the class is already location-reading and
        # the only widening available is to hand ``h_j`` the state as well; see
        # ``PDESpec.set_inner_inputs``.
        self.set_inner_inputs(inner_inputs)

    # -- the two profiles ------------------------------------------------------
    @staticmethod
    def _profile(coords: torch.Tensor, modes) -> torch.Tensor:
        x = coords[..., 0]
        out = torch.zeros_like(x)
        for amplitude, kind in modes:
            if kind == "sin":
                out = out + amplitude * torch.sin(x)
            elif kind == "cos":
                out = out + amplitude * torch.cos(x)
            elif kind == "sin2":
                out = out + amplitude * torch.sin(2 * x)
            elif kind == "cos2":
                out = out + amplitude * torch.cos(2 * x)
            elif kind == "sin3":
                out = out + amplitude * torch.sin(3 * x)
            elif kind == "cos3":
                out = out + amplitude * torch.cos(3 * x)
            else:
                raise ValueError(f"unknown Fourier mode {kind!r}")
        return out

    def g_profile(self, coords: torch.Tensor) -> torch.Tensor:
        return self._profile(coords, self.g_modes)

    def q_profile(self, coords: torch.Tensor) -> torch.Tensor:
        return self._profile(coords, self.q_modes)

    # -- model -----------------------------------------------------------------
    def features(self, u, coordinates, z):
        """The mechanism is ``q(x)``, so the network reads the coordinate."""
        if coordinates.shape[0] != u.shape[0]:
            coordinates = self.coordinates(u.shape[0], u.dtype, u.device)
        return coordinates

    def drift(self, u, z, theta, f):
        coords = self.coordinates(u.shape[0], u.dtype, u.device)
        d_x = ops.d1(u, 0, self._spacing[0], self._periodic)
        d_xx = ops.d2(u, 0, self._spacing[0], self._periodic)
        rhs = self.nu * d_xx - u * d_x
        zz = torch.as_tensor(z, dtype=u.dtype, device=u.device).reshape(-1, 1)
        theta_val = torch.as_tensor(theta["theta"], dtype=u.dtype, device=u.device)
        rhs = rhs + theta_val * zz.reshape(-1, 1, *([1] * (u.ndim - 2))) * self.g_profile(
            coords
        ).unsqueeze(1)
        if f is not None:
            rhs = rhs + f(u, z)  # q(x), broadcast over the batch
        return rhs

    def initial(self, rng, n):
        out = np.zeros((n, self.d_y, *self.grid.shape), dtype=np.float64)
        if self.ic_amplitude == 0.0:
            return out
        mesh = self.grid.mesh()[0]
        for i in range(n):
            c1 = rng.uniform(*self.ic_slope)
            c2 = rng.uniform(*self.ic_phase)
            out[i, 0] = self.ic_amplitude * np.sin(c1 * mesh + c2)
        return out

    def truth(self) -> Theta:
        return {"theta": torch.tensor(self.theta_true, dtype=torch.float64)}

    def truth_f(self, u, z):
        coords = self.coordinates(u.shape[0], u.dtype, u.device)
        return self.q_profile(coords).unsqueeze(1).expand(-1, 1, *self.grid.shape)

    def effective_information(self, config, sigma: float, pi: float) -> float:
        r"""Analytic ``I_eff`` of the supplement, at the zero background.

        .. math::
            I_{\mathrm{eff}} = 4\pi(1-\pi)\sigma^{-2}E_X|\mathcal Kg(X)|^2,

        with :math:`\mathcal K` the heat response of the linearised problem, which
        we obtain by integrating ``d_t u = nu d_xx u + g`` from a zero initial
        condition with a unit forcing scale.  Only the space--time average of
        :math:`|\mathcal Kg|^2` enters.

        The formula holds only at ``(theta_0, q_0) = (0, 0)`` with a common
        conditional design, so it is a supplement calibration for the zero
        background -- **not** a variance formula for an arbitrary Burgers flow.
        """
        from .data import solve

        u0 = torch.zeros(1, self.d_y, *self.grid.shape, dtype=torch.float64)
        trajectory = solve(
            _LinearBurgers(self.grid, self.nu, self.g_profile(self.coordinates(1))),
            u0,
            config.time,
            {"theta": torch.tensor(1.0, dtype=torch.float64)},
            f=None,
            ode_step=config.data_step,
        )
        # ``trajectory`` is (1, T, 1, *grid); the t=0 slice is identically zero,
        # so averaging over the remaining times is an unweighted space--time mean.
        mean_square = float(trajectory[0, 1:].pow(2).mean())
        return 4 * math.pi * (1 - pi) * mean_square / (sigma ** 2)


class _LinearBurgers(PDESpec):
    """``d_t u = nu d_xx u + theta g(x)``: the linearisation behind ``I_eff``."""

    name = "burgers-linear"
    d_y = 1
    feature_dim = 1

    def __init__(self, grid, nu, g_field):
        super().__init__(grid)
        self.nu = float(nu)
        self.g_field = g_field

    def drift(self, u, z, theta, f):
        d_xx = ops.d2(u, 0, self._spacing[0], self._periodic)
        return self.nu * d_xx + theta["theta"] * self.g_field

    def initial(self, rng, n):
        return np.zeros((n, self.d_y, *self.grid.shape))

    def truth(self):
        return {"theta": torch.tensor(1.0, dtype=torch.float64)}

    def truth_f(self, u, z):
        return torch.zeros_like(u)


# ---------------------------------------------------------------------------
# the new simulation case with an auxiliary variable
# ---------------------------------------------------------------------------
class AdvectionDiffusionZ(PDESpec):
    r"""``d_t u_z + (v_0 + z v_1) d_x u_z - D d_xx u_z = f(u_z)`` on the circle.

    The new Case designed for this revision.  The auxiliary variable ``Z``
    changes a *known* transport operator -- the advection speed is
    ``v_0 + z v_1`` -- while the unknown reaction ``f`` is shared across both
    inputs.  That is the structure of the SemiPDE model stripped to its
    essentials: ``Z`` moves a known mechanism and must not become an argument of
    the nonparametric part.

    It is deliberately the simplest member of the family that still exercises
    every ingredient the theory uses:

    * an unknown ``f(u)`` that depends on the state, not the coordinate;
    * a known operator whose coefficient is switched by ``Z``;
    * a randomly generated initial condition per trajectory, so the ``M`` curves
      genuinely differ rather than being repeats of two solutions.

    Choosing ``v_1`` large enough that the two inputs produce visibly different
    solutions is what makes ``theta``-free identification of ``f`` possible; if
    ``v_1 = 0`` the input is inert and the case degenerates to Case 1 with
    advection.
    """

    name = "advz"
    d_y = 1

    def __init__(
        self,
        grid: Grid,
        v0: float = 0.5,
        v1: float = 0.4,
        diffusivity: float = 5e-2,
        ic_slope: Sequence[float] = (1.0, 3.0),
        ic_phase: Sequence[float] = (0.0, 2 * np.pi),
        ic_amplitude: float = 0.5,
        inner_inputs: Optional[str] = None,
    ) -> None:
        super().__init__(grid)
        if grid.dim != 1:
            raise ValueError("AdvectionDiffusionZ is one-dimensional")
        self.v0 = float(v0)
        self.v1 = float(v1)
        self.diffusivity = float(diffusivity)
        self.ic_slope = tuple(ic_slope)
        self.ic_phase = tuple(ic_phase)
        self.ic_amplitude = float(ic_amplitude)
        # ``f`` reads the state only, by design (``Z`` must not become an
        # argument of the nonparametric part), so the inner class can be widened
        # to the coordinate; see ``PDESpec.set_inner_inputs``.
        self.set_inner_inputs(inner_inputs)

    def drift(self, u, z, theta, f):
        zz = torch.as_tensor(z, dtype=u.dtype, device=u.device).reshape(-1, 1)
        speed = (self.v0 + zz * self.v1).reshape(-1, 1, *([1] * (u.ndim - 2)))
        d_x = ops.d1(u, 0, self._spacing[0], self._periodic)
        d_xx = ops.d2(u, 0, self._spacing[0], self._periodic)
        rhs = -speed * d_x + self.diffusivity * d_xx
        if f is not None:
            rhs = rhs + f(u, z)
        return rhs

    def initial(self, rng, n):
        out = np.empty((n, self.d_y, *self.grid.shape), dtype=np.float64)
        mesh = self.grid.mesh()[0]
        for i in range(n):
            c1 = rng.uniform(*self.ic_slope)
            c2 = rng.uniform(*self.ic_phase)
            out[i, 0] = self.ic_amplitude * np.sin(c1 * mesh + c2) + 1.0
        return out

    def truth(self) -> Theta:
        """No free parameter: ``v_0, v_1, D`` are known and ``f`` is the target.

        The parameter block is kept (as an empty dict) so that the machinery can
        still be exercised; the case is about recovering ``f`` and about whether
        the switched transport helps or hurts, not about a scalar ``theta``.  A
        researcher who wants a parameter can expose ``D`` by moving it into
        :meth:`drift` from ``theta``; the hook is left deliberately visible.
        """
        return {}

    def truth_f(self, u, z):
        return u * (1.0 - u)


# ---------------------------------------------------------------------------
# Case 1 with a switched transport operator (the parameter-carrying variant)
# ---------------------------------------------------------------------------
class AdvectionDiffusionThetaZ(AdvectionDiffusionZ):
    """``AdvectionDiffusionZ`` with the diffusivity promoted to a parameter.

    ``d_t u + (v_0 + z v_1) d_x u = theta d_xx u + f(u)`` with ``f`` shared and
    ``theta`` unknown, so the case carries both an estimated parameter and a
    nonparametric nuisance.  ``v_0`` and ``v_1`` stay known.
    """

    name = "advz-theta"

    def __init__(self, grid: Grid, theta_true: float = 5e-2, **kwargs) -> None:
        super().__init__(grid, diffusivity=theta_true, **kwargs)
        self.theta_true = float(theta_true)

    def drift(self, u, z, theta, f):
        zz = torch.as_tensor(z, dtype=u.dtype, device=u.device).reshape(-1, 1)
        speed = (self.v0 + zz * self.v1).reshape(-1, 1, *([1] * (u.ndim - 2)))
        d_x = ops.d1(u, 0, self._spacing[0], self._periodic)
        d_xx = ops.d2(u, 0, self._spacing[0], self._periodic)
        rhs = -speed * d_x + theta["theta"] * d_xx
        if f is not None:
            rhs = rhs + f(u, z)
        return rhs

    def truth(self) -> Theta:
        return {"theta": torch.tensor(self.theta_true, dtype=torch.float64)}
