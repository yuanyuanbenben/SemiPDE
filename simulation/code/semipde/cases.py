"""The paper's simulation cases, assembled into runnable ``(spec, config)`` pairs.

One entry per case, each carrying the settings printed in the main text and the
supplement, plus a note on where the case belongs in the revised paper:

======  =========================================  ==================================
key     model                                      placement
======  =========================================  ==================================
``rd``  Case 1, Fisher--KPP reaction--diffusion     main text (Example 1)
``fhn`` Case 2, FitzHugh--Nagumo                    main text
``advz`` new case, ``Z``-switched advection         main text (new)
``burg`` Example 2, forced Burgers with input       main text (new)
``np``  Case 3, Nernst--Planck                      supplement
``ns``  Case 4, 3-D Navier--Stokes                  supplement
======  =========================================  ==================================

The last two move to the supplement because they exercise neither of the
paper's two examples: Case 3's unknown mechanism reads the coordinate rather
than the state, and Case 4 is a 3-D system whose ``Z``-free structure adds cost
without adding a new statistical phenomenon.  The two ``Z`` cases take their
places, so the main text shows exactly four cases -- two matching Example 1, two
matching Example 2 -- while nothing is lost from the record.

Two size conventions are in play and are easy to conflate:

``n_traj``
    observations from **one** trajectory, ``n_t * prod(grid) * counted fields``.
    This is the ``n`` the paper's tables report (Case 1 at ``n_x = 32`` gives
    ``n_traj = 800``).
``n_total``
    ``M * n_traj``, the number of residuals in the empirical loss and the sample
    size that drives the estimator and the interval width.  The paper treats the
    ``M`` trajectories as one pooled vector, and so does this framework.

:attr:`CaseConfig.n_counted_fields` records the paper's own asymmetry -- Case 2
counts both fields, Case 4 counts one -- rather than leaving it implicit.  This
is not a guess: ``tab:simu:compared_baseline`` prints ``3200`` for Case 2 at
``n_t = 25``, ``n_x = n_y = 8`` (``2 * 25 * 8 * 8``) and ``12800`` at ``16 x 16``,
while Case 4 prints ``12800`` at ``8^3`` (``25 * 8^3``, one field).  The
``n = 640`` of the inference section is the outlier, and the paper's prose was
corrected to match the table rather than the other way round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .config import CaseConfig, InputSpec
from .grids import Grid, TimeGrid
from .pde import (
    AdvectionDiffusionThetaZ,
    ForcedBurgers,
    FitzHughNagumo,
    NavierStokes3D,
    NernstPlanck,
    PDESpec,
    ReactionDiffusion,
)

__all__ = ["Case", "CASES", "build", "main_text_cases", "supplement_cases", "summary_table"]

#: Curves per split, as in Case 3 of the supplement ("64 for training, 16 for
#: validation and 16 for testing").  Applied to every case for comparability.
TRAIN_CURVES, VAL_CURVES, TEST_CURVES = 64, 16, 16


@dataclass
class Case:
    """A spec, its configuration, and where it belongs in the paper."""

    key: str
    title: str
    spec: PDESpec
    config: CaseConfig
    placement: str = "main"  # "main" or "supp"
    note: str = ""

    @property
    def n_curves(self) -> int:
        return self.config.M + self.config.n_val + self.config.n_test

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Case {self.key}: {self.title} ({self.placement})>"


def _curves(train: int, val: int, test: int) -> Dict[str, int]:
    return {"M": train, "n_val": val, "n_test": test}


# ---------------------------------------------------------------------------
# Case 1 -- Fisher--KPP reaction--diffusion (Example 1)
# ---------------------------------------------------------------------------
def case1_rd(
    n_grid: int = 32,
    sigma: float = 0.1,
    n_t: int = 25,
    legacy_ic: bool = False,
    mechanism_inputs: str = "state",
    inner_inputs: Optional[str] = None,
    theta_true: float = 1e-2,
    **extra,
) -> Case:
    # The benchmark's value is 1e-2, a middle point between the paper's 3e-3 and
    # the 3e-2 this case carried on 2026-10-22/23.  The drift is
    # ``theta * laplacian(u) + f(u)`` and ``f(u) = u (1 - u)`` carries no
    # theta, so the parametric baseline (``f == 0``) can only mimic the
    # reaction by inflating theta: measured, its absolute error stays in the
    # 2e-2..4e-2 range whatever theta_true is, while the proposal's scales with
    # theta_true.  At 3e-2 that fixed offset was small enough to tie at
    # sigma = 0.5 (paired t = -0.29 over 50 reps); at 1e-2 the two estimators
    # separate, and 1e-2 is also closer to the paper's own setting.
    # ``theta_true`` stays a keyword so either end can be reproduced
    # explicitly.  Everything -- the generator, the estimator and the scoring --
    # reads this one number, so a run written before this change cannot be
    # resumed into a run after it (the resume key does not record theta_true;
    # write such runs to their own ``--out``).
    grid = Grid.from_ranges([[-1.0, 1.0]], [n_grid], periodic=True)
    spec = ReactionDiffusion(grid, theta_true=theta_true, legacy_ic=legacy_ic,
                             mechanism_inputs=mechanism_inputs,
                             inner_inputs=inner_inputs)
    config = CaseConfig(
        name=f"rd-n{n_grid}-s{sigma}",
        grid=grid,
        time=TimeGrid(0.0, 2.5, n_t, include_endpoint=True),
        d_y=1,
        sigma=sigma,
        # ``extra`` must be able to override the curve counts, so the two
        # mappings are merged rather than splatted one after the other:
        # ``f(**a, **b)`` with a shared key is a TypeError, not a merge.
        **{**_curves(TRAIN_CURVES, VAL_CURVES, TEST_CURVES), **extra},
    )
    return Case(
        key="rd",
        title="Fisher--KPP reaction--diffusion (Example 1, no Z)",
        spec=spec,
        config=config,
        placement="main",
        note=(
            "theta = 1e-2 (the paper prints 3e-3), f(u) = u(1-u).  "
            "The reference case.  Follows the "
            "supplement's initial condition; pass legacy_ic=True to reproduce "
            "the published tables, which used a different profile.  The inner "
            "class is the paper's, matched to f_hat, because the two initial "
            "profiles need different answers and this default has to serve "
            "both: on the supplement's profile inner_inputs='location' lands "
            "on nominal coverage (0.80/0.95/0.95 against a matched 0.65/0.75/"
            "0.85), but on the published legacy profile it moves the other way "
            "and *shortens* se (0.774x), taking C.R.90 from 0.542 to 0.396.  "
            "See docs/size.md Sec. 5.3e."
        ),
    )


# ---------------------------------------------------------------------------
# Case 2 -- FitzHugh--Nagumo
# ---------------------------------------------------------------------------
def case2_fhn(
    n_grid: int = 8,
    sigma: float = 0.1,
    n_t: int = 25,
    inner_inputs: Optional[str] = None,
    **extra,
) -> Case:
    grid = Grid.from_ranges([[-1.0, 1.0]] * 2, [n_grid, n_grid], periodic=True)
    spec = FitzHughNagumo(grid, inner_inputs=inner_inputs)
    config = CaseConfig(
        name=f"fhn-n{n_grid}-s{sigma}",
        grid=grid,
        time=TimeGrid(0.0, 2.5, n_t, include_endpoint=True),
        d_y=2,
        sigma=sigma,
        # The paper counts both fields in Case 2: n = 2 n_t n_x n_y.  Its own
        # baseline table agrees -- see the note on ``n_counted_fields`` in the
        # module docstring.
        n_counted_fields=2,
        # ``extra`` must be able to override the curve counts, so the two
        # mappings are merged rather than splatted one after the other:
        # ``f(**a, **b)`` with a shared key is a TypeError, not a merge.
        **{**_curves(TRAIN_CURVES, VAL_CURVES, TEST_CURVES), **extra},
    )
    return Case(
        key="fhn",
        title="FitzHugh--Nagumo (Case 2, two coupled fields)",
        spec=spec,
        config=config,
        placement="main",
        note="theta = (1e-3, 3e-3).  Both fields are counted in n.",
    )


# ---------------------------------------------------------------------------
# Case 3 -- Nernst--Planck  (supplement)
# ---------------------------------------------------------------------------
def case3_np(n_grid: int = 32, sigma: float = 0.1, n_t: int = 25,
             inner_inputs: Optional[str] = None, **extra) -> Case:
    grid = Grid.from_ranges([[-1.0, 1.0]] * 2, [n_grid, n_grid], periodic=True)
    spec = NernstPlanck(grid, inner_inputs=inner_inputs)
    config = CaseConfig(
        name=f"np-n{n_grid}-s{sigma}",
        grid=grid,
        time=TimeGrid(0.0, 2.5, n_t, include_endpoint=True),
        d_y=1,
        sigma=sigma,
        # ``extra`` must be able to override the curve counts, so the two
        # mappings are merged rather than splatted one after the other:
        # ``f(**a, **b)`` with a shared key is a TypeError, not a merge.
        **{**_curves(TRAIN_CURVES, VAL_CURVES, TEST_CURVES), **extra},
    )
    return Case(
        key="np",
        title="Nernst--Planck (Case 3)",
        spec=spec,
        config=config,
        placement="supp",
        note="theta = (7e-3, 5.33e-3).  The unknown mechanism reads x, not u.",
    )


# ---------------------------------------------------------------------------
# Case 4 -- three-dimensional Navier--Stokes  (supplement)
# ---------------------------------------------------------------------------
def case4_ns(n_grid: int = 8, sigma: float = 0.1, n_t: int = 25,
             inner_inputs: Optional[str] = None, **extra) -> Case:
    grid = Grid.from_ranges([[-1.0, 1.0]] * 3, [n_grid] * 3, periodic=True)
    spec = NavierStokes3D(grid, inner_inputs=inner_inputs)
    config = CaseConfig(
        name=f"ns-n{n_grid}-s{sigma}",
        grid=grid,
        time=TimeGrid(0.0, 2.5, n_t, include_endpoint=True),
        d_y=3,
        sigma=sigma,
        # The paper counts a single field here, unlike Case 2.
        n_counted_fields=1,
        # ``extra`` must be able to override the curve counts, so the two
        # mappings are merged rather than splatted one after the other:
        # ``f(**a, **b)`` with a shared key is a TypeError, not a merge.
        **{**_curves(TRAIN_CURVES, VAL_CURVES, TEST_CURVES), **extra},
    )
    return Case(
        key="ns",
        title="3-D Navier--Stokes momentum equation (Case 4)",
        spec=spec,
        config=config,
        placement="supp",
        note="theta_1, theta_2 are 3-vectors; only one field is counted in n.",
    )


# ---------------------------------------------------------------------------
# Example 2 -- forced Burgers with an observed input
# ---------------------------------------------------------------------------
#: The paper's raw Example-2 forcing, before rescaling.  Kept as its own name
#: because the rescaling below is derived from it and the derivation is only
#: legible if both are visible.
_BURG_Q0 = ((0.3, "sin"), (0.2, "cos2"), (0.1, "sin3"))


def case_burgers(
    n_grid: int = 32,
    sigma: float = 0.1,
    n_t: int = 25,
    pi: float = 0.5,
    t_end: float = 2.5,
    nu: float = 0.125,
    theta_true: float = 0.2,
    q_modes: Optional[Sequence[tuple]] = None,
    inner_inputs: Optional[str] = None,
    **extra,
) -> Case:
    """The paper's Example 2 with a two-valued input sampled at frequency ``pi``.

    ``pi = 0.5`` is the balanced design; ``pi = 1.0`` is the negative control, in
    which every observation carries the same forcing sign and ``theta`` becomes
    unidentified through ``(theta, q) -> (theta + c, q - c z_0 g)``.  The
    supplement's ``I_eff`` calibration is exact only at the zero background, so
    this case is run with the paper's nonzero ``theta`` and the calibration is
    reported separately.

    **The window is the paper's ``[0, 2.5]``**, as for Cases 1--2, but the
    coefficients are rescaled to get there.  The paper names Example 2 and
    publishes neither ``nu`` nor ``t_end``; taken at face value -- ``nu = 0.05``
    with ``q_0 = 0.3 sin x + 0.2 cos 2x + 0.1 sin 3x`` -- the forcing sharpens
    ``u_z(0, x) = 0`` into a shock that runs to NaN well before ``t = 2.5``
    (``max|u| = 1.72`` and ``max|u_xx| = 271`` at the last finite step), so the
    naive ``[0, 2.5]`` window is not merely inaccurate, it is unintegrable.

    Two independent rescalings fix that, and both are needed:

    * **forcing** ``q_0 -> q_0 / L`` and ``theta -> theta / L`` -- the source
      term of ``d_t u + u d_x u = nu d_xx u + theta z g + q`` controls the
      amplitude, so dividing it by the window length keeps ``max|u|`` at the
      scale the ``[0, 1]`` version had;
    * **viscosity** ``nu -> nu * L`` -- the diffusive length ``sqrt(nu t)`` is
      what sets how sharp the layer gets relative to the domain, and holding
      ``nu * t_end`` fixed holds the field's roughness fixed.

    Together they reproduce, on ``[0, 2.5]``, the statistics of the ``[0, 1]``
    problem this case used to be: ``max|u| = 0.747`` against ``0.875``,
    ``max|u_xx| = 2.51`` against ``2.97``, and the ``n_grid = 32`` discretisation
    error *improves* (``1.25e-1`` against ``1.42e-1`` where the ``256``-point
    reference is exact).  Rescaling only the forcing leaves ``max|u_xx|`` at
    ``8.43``; rescaling only by ``1/L`` more, so that ``nu`` is untouched, decays
    the signal to ``max|u| = 0.338``.  ``scripts/probe_burg_window.py`` produces
    the table this paragraph is read off.

    ``theta_true`` and ``q_modes`` are only ever moved off the paper's values by
    the supplement's zero-background calibration, where ``(theta_0, q_0) =
    (0, 0)`` is the one setting the analytic ``I_eff`` describes.
    """
    grid = Grid.from_ranges([[0.0, 2 * math.pi]], [n_grid], periodic=True)
    spec = ForcedBurgers(
        grid,
        nu=nu,
        theta_true=theta_true,
        q_modes=tuple((a / t_end, kind) for a, kind in _BURG_Q0)
        if q_modes is None
        else tuple(q_modes),
        inner_inputs=inner_inputs,
    )
    inputs = InputSpec.from_pi(pi) if pi < 1.0 else InputSpec((1.0,), (1.0,))
    config = CaseConfig(
        name=f"burg-n{n_grid}-pi{pi}",
        grid=grid,
        time=TimeGrid(0.0, t_end, n_t, include_endpoint=True),
        d_y=1,
        sigma=sigma,
        inputs=inputs,
        # ``extra`` must be able to override the curve counts, so the two
        # mappings are merged rather than splatted one after the other:
        # ``f(**a, **b)`` with a shared key is a TypeError, not a merge.
        **{**_curves(TRAIN_CURVES, VAL_CURVES, TEST_CURVES), **extra},
    )
    return Case(
        key="burg",
        title=f"Forced Burgers with input Z (Example 2, pi={pi})",
        spec=spec,
        config=config,
        placement="main",
        note=(
            "u_z(0,x)=0; nu=0.125; t_end=2.5; g(x)=sin x; "
            "q_0 = 0.12 sin x + 0.08 cos 2x + 0.04 sin 3x "
            "(= (0.3, 0.2, 0.1)/2.5, with nu = 0.05*2.5).  The sin x overlap "
            "between g and q_0 is deliberate: it is what makes theta identifiable "
            "only from the difference between the two forcing directions."
        ),
    )


# ---------------------------------------------------------------------------
# the new case -- Z switches a known transport operator
# ---------------------------------------------------------------------------
def case_advz(
    n_grid: int = 32,
    sigma: float = 0.1,
    n_t: int = 25,
    pi: float = 0.5,
    t_end: float = 2.5,
    inner_inputs: Optional[str] = None,
    **extra,
) -> Case:
    """The new case: ``Z`` flips the sign of the advection in a known operator.

    ``d_t u + (v_0 + z v_1) d_x u = theta d_xx u + f(u)`` on the circle, with
    ``v_0 = 0.5``, ``v_1 = 0.4`` known, ``theta`` estimated and ``f`` shared.
    Unlike the Burgers case the initial condition is a random smooth profile per
    trajectory, so the ``M`` curves genuinely differ and the unknown ``f`` must
    be identified from the state rather than from the coordinate.
    """
    grid = Grid.from_ranges([[0.0, 2 * math.pi]], [n_grid], periodic=True)
    spec = AdvectionDiffusionThetaZ(grid, theta_true=5e-2,
                                    inner_inputs=inner_inputs)
    inputs = InputSpec.from_pi(pi) if pi < 1.0 else InputSpec((1.0,), (1.0,))
    config = CaseConfig(
        name=f"advz-n{n_grid}-pi{pi}",
        grid=grid,
        time=TimeGrid(0.0, t_end, n_t, include_endpoint=True),
        d_y=1,
        sigma=sigma,
        inputs=inputs,
        # ``extra`` must be able to override the curve counts, so the two
        # mappings are merged rather than splatted one after the other:
        # ``f(**a, **b)`` with a shared key is a TypeError, not a merge.
        **{**_curves(TRAIN_CURVES, VAL_CURVES, TEST_CURVES), **extra},
    )
    return Case(
        key="advz",
        title=f"Advection--diffusion with input Z (new case, pi={pi})",
        spec=spec,
        config=config,
        placement="main",
        note=(
            "v_0=0.5, v_1=0.4, theta=5e-2, f(u)=u(1-u).  Z changes a known "
            "operator (the advection speed) without entering f."
        ),
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
CASES = {
    "rd": case1_rd,
    "fhn": case2_fhn,
    "np": case3_np,
    "ns": case4_ns,
    "burg": case_burgers,
    "advz": case_advz,
}

#: The four shown in the main text: two matching Example 1, two matching Example 2.
MAIN_KEYS = ("rd", "fhn", "burg", "advz")
SUPP_KEYS = ("np", "ns")


def build(key: str, **overrides) -> Case:
    """Construct a case by key, forwarding ``overrides`` to its factory."""
    try:
        factory = CASES[key]
    except KeyError as exc:
        raise KeyError(f"unknown case {key!r}; known: {sorted(CASES)}") from exc
    return factory(**overrides)


def main_text_cases(**overrides) -> Dict[str, Case]:
    return {k: build(k, **overrides) for k in MAIN_KEYS}


def supplement_cases(**overrides) -> Dict[str, Case]:
    return {k: build(k, **overrides) for k in SUPP_KEYS}


def summary_table(cases: Optional[Dict[str, Case]] = None) -> str:
    """A fixed-width table of the sample-size bookkeeping, for the record.

    Printing this next to any result makes the ``n_traj`` versus ``n_total``
    distinction impossible to misread, which is exactly the ambiguity that made
    the size column of the original tables hard to reconcile.
    """
    cases = CASES if cases is None else cases
    header = (
        f"{'case':<6}{'where':<7}{'grid':<12}{'n_t':<5}{'d_y':<5}"
        f"{'n_traj':<9}{'M':<5}{'n_total':<9}"
    )
    lines = [header, "-" * len(header)]
    for factory in cases.values():
        case = factory() if callable(factory) else factory
        cfg = case.config
        lines.append(
            f"{case.key:<6}{case.placement:<7}{str(cfg.grid.shape):<12}"
            f"{cfg.time.n_obs:<5}{cfg.d_y:<5}{cfg.n_traj:<9}"
            f"{cfg.M:<5}{cfg.n_total:<9}"
        )
    return "\n".join(lines)
