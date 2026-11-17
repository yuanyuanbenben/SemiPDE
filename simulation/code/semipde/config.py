"""Case configuration.

This module is the single source of truth for the *size* of an experiment.
The legacy code conflated several different quantities behind the flags
``-n``, ``-ss`` and ``-ts``, and the paper reports yet another one.  The three
quantities are separated here:

``n_traj``
    Scalars observed along one trajectory,
    ``n_obs * prod(grid.shape) * n_observed_fields``.  This is the ``n`` that
    appears in the tables of the paper (e.g. Case 1 with ``n_t = 25`` and
    ``n_x = 32`` gives ``n_traj = 800``).

``M``
    Number of repeated trajectories, i.e. the paper's Remark 2 vector
    ``u(x) = (u_1(x), ..., u_M(x))^T``.  Held fixed and *not* counted in the
    reported ``n``.

``n_total``
    ``M * n_traj``.  This is the number of residuals in the empirical loss and
    therefore the ``n`` that enters the asymptotic statements
    (Theorems 2--4) and the confidence-interval width.  Reporting ``n_traj``
    while forming intervals with ``n_total`` is the inconsistency that the
    legacy ``RD_coverage_out.py`` fell into.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

from .grids import Grid, TimeGrid


@dataclass(frozen=True)
class InputSpec:
    """The auxiliary variable ``Z``.

    ``values``/``probs`` describe the marginal law of ``Z``.  ``Z`` never enters
    the nonparametric function ``f``; it acts through the known mechanism
    ``L_z``, the forcing, or the known initial/boundary data, per
    Eq. (2) of the paper.
    """

    values: Tuple[float, ...]
    probs: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.probs):
            raise ValueError("values and probs must have the same length")
        s = float(sum(self.probs))
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"probs must sum to 1, got {s}")

    @classmethod
    def balanced(cls) -> "InputSpec":
        return cls((-1.0, 1.0), (0.5, 0.5))

    @classmethod
    def from_pi(cls, pi: float) -> "InputSpec":
        """``P(Z = 1) = pi`` on ``Z in {-1, 1}``."""
        if not 0.0 <= pi <= 1.0:
            raise ValueError(f"pi must lie in [0, 1], got {pi}")
        return cls((-1.0, 1.0), (1.0 - pi, pi))

    @property
    def n_levels(self) -> int:
        return len(self.values)

    @property
    def pi(self) -> float:
        """``P(Z = 1)`` for the binary case; raises otherwise."""
        if self.values != (-1.0, 1.0):
            raise ValueError("pi is only defined for Z in {-1, 1}")
        return float(self.probs[1])


@dataclass
class CaseConfig:
    """Everything needed to define and fit one simulation case.

    Only the fields that feed the *statistical* notion of size are mandatory;
    the rest have sensible defaults matching the paper's implementation
    details (Euler step 0.001, widths ``[16, 64, 64, 16]``, Adam at 1e-3).
    """

    name: str
    grid: Grid
    time: TimeGrid
    d_y: int = 1

    # --- experiments / repetition -------------------------------------------------
    M: int = 64
    n_val: int = 16
    n_test: int = 16
    inputs: Optional[InputSpec] = None

    # --- observation noise --------------------------------------------------------
    sigma: float = 0.1

    # --- network ------------------------------------------------------------------
    #: Layer widths.  The paper's sieve condition puts a lower bound on the
    #: *width* as a function of ``n`` (supplement, Corollary
    #: ``repu_width_tolerance``); this is a fixed literal, so the code is at
    #: ``C_m = 0`` in that statement rather than at the ``C_m > 0`` the
    #: corollary needs.  :mod:`semipde.sieve` implements the condition, measures
    #: the fitted radius ``R_n = sqrt(J(F))`` against it, and provides
    #: :func:`~semipde.sieve.scaled_hidden` to put a run on a width schedule.
    hidden: Sequence[int] = (16, 64, 64, 16)
    activation: str = "relu"
    repu_power: int = 1
    #: Apply the ``sqrt(c_l / m_j)`` and ``sqrt((l + 1) / d_f)`` factors of
    #: Eq. (5).  The theory assumes them; turning them off reproduces the
    #: legacy networks, which used Xavier initialisation and no normalisation.
    normalize: bool = True
    #: ``"gaussian"`` is the paper's scheme, ``"xavier"`` the legacy one.
    #: ``"gaussian_zero_out"`` is ``"gaussian"`` with the output layer zeroed,
    #: which makes the initial mechanism *and its slope in ``u``* vanish.  See
    #: ``FNN.reset_parameters``: on windows much longer than the paper's ``2.5``
    #: the random slope is amplified by the reverse-time adjoint and the
    #: backward pass overflows.
    init: str = "gaussian"
    #: One trunk with ``d_f`` output heads, rather than ``d_f`` independent
    #: networks reading the same features.
    #:
    #: The supplement describes one network per case, not one per output: its
    #: Case 2 paragraph (``supp.tex:537``) asks for "a two-dimensional input,
    #: hidden layers with widths 16, 64, 64, 16, and a two-dimensional output",
    #: and ``legacy/GFN_model.py`` builds exactly that -- a single
    #: ``nn.Sequential`` whose last layer is ``Linear(16, d_f)``.  Splitting it
    #: into ``d_f`` blocks is a strictly *larger* class: each component gets a
    #: private trunk, so the nuisance part is free to take shapes the shared
    #: version cannot reach.
    #:
    #: That matters on the two-field cases, where the mechanism and the
    #: parametric term compete for the same signal.  Measured on Case 2
    #: (``M = 64``, ``sigma = 0.1``, 50 repetitions), the independent-block
    #: version returns ``theta_hat`` 79% and 159% above the truth with a
    #: sampling spread of 1.5 and 3.5 parameter-lengths, against the paper's
    #: 2% and 6% for the same two columns -- an estimator this far from the
    #: published one is not the same estimator.  What the blocks cost is the
    #: constraint, not the capacity: the shared trunk's function class for
    #: ``(f_1, ..., f_{d_f})`` is the image of a single feature map, and the
    #: parametric direction is identified only to the extent that the nuisance
    #: class cannot mimic it.
    #:
    #: Single-output cases are unaffected -- the two constructions coincide bit
    #: for bit when ``d_f = 1``.  The default is the shared trunk, which is what
    #: the supplement specifies and what the legacy code did; ``False`` is kept
    #: because every measurement in ``docs/`` was made with it, and the
    #: comparison is worth being able to reproduce.
    mechanism_shared: bool = True
    #: Where the optimisation of ``theta`` starts: ``"zero"`` or ``"truth"``.
    #:
    #: ``"zero"`` is the legacy convention and therefore the one the published
    #: tables were produced under -- every legacy case declares its coefficient
    #: as ``nn.Parameter(torch.tensor(0.))`` (see ``legacy/RD_model.py``,
    #: ``NP_model.py``, ``GFN_model.py``, ``NS_model.py``).  It is also the only
    #: internally consistent choice here: ``J(F) = ||phi - phi_0||^2`` makes
    #: ``phi_0`` the *initial* network, so ``theta_0`` has to be the *initial*
    #: parameter, and starting the two references at different things -- one at
    #: the initialisation, the other at the truth -- is not a convention any
    #: statement in the paper covers.
    #:
    #: ``"truth"`` is available because the local analysis is a perturbation
    #: around ``(theta_0, phi_0)`` and it is worth being able to see that regime.
    #: It must not be used for a bias or coverage study: the estimator would
    #: start at zero error, which is the quantity under test.
    theta_init: str = "zero"

    # --- numerics -----------------------------------------------------------------
    #: The scheme and step of the **model's** rollout during training and
    #: inference.  Unlike :attr:`data_step`, this one *does* enter the
    #: estimator: the rollout is what the data are fitted against, so its
    #: discretisation error is a bias in ``theta_hat``, not an accuracy loss.
    #:
    #: The supplement's implementation paragraphs (one per case) all specify
    #: Euler at ``0.001``.  The legacy ``*_model.py`` used ``0.02``, and the
    #: published tables were produced with that; the earlier note here claimed
    #: the two agree to five significant figures, which is true for Case 1 but
    #: **false for Example 2 (forced Burgers)**.  Measured there (``M = 256``,
    #: ``sigma = 0.1``, 20 repetitions, ``lambda_n = 0``)::
    #:
    #:     ode_method   ode_step   theta bias    bias/SE    cost (drift evals)
    #:     euler        2e-2       -5.25e-04     0.98       48
    #:     euler        1e-2       -3.00e-04     0.56       96
    #:     rk4          2e-2        see runs/ode_trade_rk4     192
    #:
    #: ``SE`` there is ``5.37e-04``, so the legacy ``2e-2`` Euler contributes a
    #: bias of one full standard error -- enough on its own to pull a 95%
    #: interval's coverage down to ~0.85.  ``"rk4"`` at the same ``ode_step``
    #: costs 4x the drift evaluations of Euler (192 against 48) and reproduces
    #: the paper's ``1e-3`` Euler accuracy (1008 evaluations) at a fifth of its
    #: cost, so it is the default.  Set ``ode_method="euler"`` with
    #: ``ode_step=1e-3`` to reproduce the supplement literally.
    #:
    #: The step itself was then re-measured for RK4, since the bias argument
    #: above is about a *scheme* difference and says nothing about how coarse
    #: RK4 may be.  Against a reference solve at ``2e-3``, forced Burgers
    #: (``n_x = 32``, ``n_t = 25``, ``theta_0 = 0.5``), ``max|u - u_ref|``::
    #:
    #:     ode_step   substeps/0.1   max|u - u_ref|   / fit error (3.2e-3)
    #:     5e-3       20             9.741e-11        3.0e-08
    #:     1e-2       10             1.966e-09        6.1e-07
    #:     2e-2        5             3.218e-08        1.0e-05
    #:     5e-2        2             1.332e-06        4.2e-04
    #:     1e-1        1             2.291e-05        7.2e-03
    #:
    #: Two things follow.  First, the third column is the *reference* solver's
    #: error, not the estimator's: ``generate`` integrates the truth with
    #: :attr:`data_step`, and the two steps are equal by default, so a rollout
    #: error that is common to both cancels out of ``theta_hat`` entirely.  What
    #: is left is the fourth column -- how far the truth being fitted has moved
    #: from the exact solution of the PDE.  At ``5e-2`` that is ``4e-4`` of the
    #: fit's own error, i.e. nothing; at ``1e-1`` it is ``0.7%`` of it, and the
    #: comparison above is a maximum rather than an rms, so ``1e-1`` is nearly
    #: defensible -- but ``5e-2`` is the last value with four orders of margin.
    #: Second, RK4 is fourth order, so this costs nothing in accuracy per step
    #: and everything in *count*: ``5e-2`` takes two substeps per ``0.1``
    #: observation interval where ``2e-2`` takes five.  Every drift evaluation
    #: is a network evaluation on a small batch, so a fit is ~2.5x faster --
    #: which is what makes the 500-repetition size study of ``supp.tex:525``
    #: affordable at all.
    ode_method: str = "rk4"
    ode_step: float = 5e-2
    #: The reference solver's step when *generating* data.  This is the legacy
    #: ``delta_t``.  It only needs to be small enough that the reference
    #: trajectory is accurate -- it never enters the estimator -- so it should
    #: be set as large as accuracy allows, not as small as possible.  This is
    #: the opposite of :attr:`ode_step`, which is the *model's* rollout step and
    #: does bias the estimate.
    #:
    #: ``cases.py`` carried the legacy per-case values (``1e-3`` for Cases 1--2
    #: and Burgers/AdvZ, ``5e-3`` for Case 3).  Those were ``1e-3``-Euler-era
    #: settings kept "because they have not been re-measured"; they have now
    #: been re-measured, and RK4 at ``2e-2`` reproduces every one of them::
    #:
    #:     case   published step   max|u(step) - u(2e-2)|   relative
    #:     rd     1e-3             1.076e-10               1.1e-10
    #:     fhn    1e-3             3.291e-09               3.3e-09
    #:     np     5e-3             2.207e-08               6.5e-09
    #:     ns     2e-2             --                      --
    #:     burg   1e-3             3.164e-09               3.6e-09
    #:     advz   1e-3             2.304e-07               1.5e-07
    #:
    #: The largest deviation of the reference trajectory is ``1.5e-07`` against
    #: an observation noise of ``sigma = 0.1``, i.e. six orders below the noise,
    #: while the reference solve is by far the most expensive part of a run.  So
    #: every case -- including the ones ``cases.py`` sets explicitly, for
    #: visibility -- now uses the default.  ``ns`` always did.
    #:
    #: One further doubling of the step, to ``5e-2``, was measured on forced
    #: Burgers as part of the same exercise: ``max|u - u(2e-3)| = 1.332e-06``,
    #: still five orders below ``sigma``, and now ``4e-4`` of the fit's own
    #: error.  It must move *with* :attr:`ode_step` and not independently --
    #: the argument that the model's rollout error cancels rests on the data
    #: being generated at the step the rollout uses -- and the two defaults are
    #: therefore kept equal here.  ``cases.py`` no longer names either of them.
    data_step: float = 5e-2

    # --- optimisation -------------------------------------------------------------
    #: The training budget.  ``1000`` rather than the legacy ``2000 Adam steps
    #: on 16 curves`` it replaces.  It is *not* a fix for coverage: at 200
    #: repetitions on the ``n_x = 32``, ``n_t = 25`` grid, ``se/Std`` moves
    #: ``0.870 -> 0.921`` between ``200`` and ``1000`` epochs and then flattens,
    #: leaving C.R. at ``0.915`` against a nominal ``0.95``.  A ``4x`` larger
    #: sample reaches the same ``0.92``, so on *that* grid the residual is a
    #: shortfall in ``Sigma_eff`` itself, not in the fit.  ``1000`` is the budget
    #: where that plateau is reached.
    #:
    #: **The plateau is a property of the grid, not of the estimator.**  Move to
    #: the grid the paper builds its own coverage table on (``n_x = 16``,
    #: ``n_t = 10``, same ``M = 64``, same ``sigma``, 300 epochs) and group ``A``
    #: of ``scripts/probe_variance.py`` gives ``se/sd = 0.61`` on Fisher--KPP and
    #: ``0.86`` on forced Burgers -- not ``0.92``.  The observation count moves
    #: by ``5x`` between the two grids while the widths do not, so ``p / n``
    #: moves from ``0.12`` to ``0.62``.  Read this number as a property of the
    #: configuration it was taken on; ``docs/size.md`` is the current record.
    #:
    #: An earlier version of this note used 60 repetitions and reported
    #: ``0.948``; see ``docs/divergences.md`` Sec. 10 for why the diagnostic
    #: needs a few hundred.
    epochs: int = 1000
    inner_steps: int = 5
    lr: float = 1e-3
    #: The penalty weight of Eq. (3).  The default is ``0`` -- see the note
    #: below for why, and for the one metric on which a nonzero value wins.
    #:
    #: ``J(F) = ||phi - phi_0||^2`` is a *parameter-space* norm, and it **grows
    #: without bound** as the network learns, because the centred net starts at
    #: output identically zero and must move its weights a long way to build a
    #: nonzero mechanism.  Measured on Example 2 (``M = 64``, 200 epochs)::
    #:
    #:     epoch   L_n        J(F)      lambda_n*J at 1e-4   share of the objective
    #:     1       2.30e-02   2.29e-02  2.29e-06            0.0%
    #:     10      1.33e-02   1.92e-01  1.92e-05            0.1%
    #:     50      9.89e-03   2.72e+00  2.72e-04            2.7%
    #:     200     9.50e-03   1.29e+01  1.29e-03            13.6%
    #:
    #: So a *constant* ``lambda_n`` is not a tie-breaker -- it is a constraint
    #: that tightens exactly as the fit improves.  (``lambda_n = 1`` makes it
    #: the objective outright, suppressing the network back to ``phi_0``.)
    #: The theory asks for ``lambda_n -> 0`` at rate
    #: ``n^{-2(a+r)/D_s} (log n)^{(a+r)/(s+r)}``, which is the regime the
    #: default sits in.
    #:
    #: What that buys, on Example 2 (``M = 64``, 200 epochs, the *same* fit):
    #:
    #:     lambda_n   q rel.err   u approx   theta error   se/std (M=64 / 256)
    #:     0          0.074       5.2e-03    1.24e-03      0.975 / 1.034
    #:     1e-04      0.154       1.37e-02   8.94e-04      0.637 / 0.694
    #:     1e-02      0.382       4.5e-02    1.66e-03      --
    #:
    #: The tradeoff is real and worth stating rather than hiding: ``1e-4``
    #: gives a ~28% smaller *theta* error, because pinning the network towards
    #: ``phi_0`` stabilises the parameter direction.  It pays for that with a
    #: 2x worse mechanism and a 2.6x worse prediction, and it breaks the
    #: calibration (``se/std`` well below 1 means the reported interval is too
    #: short).  Since the estimand is the *joint* ``(theta, f)``, the default
    #: takes the mechanism and the calibration.  ``lambda_n=0.0`` also happens
    #: to be the legacy behaviour, which had the penalty commented out.
    lambda_n: float = 0.0
    #: Threshold ``tau`` of the gradient-norm stopping rule (Eq. 16): the run
    #: stops at the first epoch with ``eta*|grad L_n|^2 <= tau``.  ``None``
    #: disables it, which is the default and what the paper's implementation
    #: details do in practice (a fixed epoch budget, returning the
    #: best-validation iterate).
    #:
    #: **A fixed ``tau`` is the wrong shape of knob and the old default of
    #: ``1e-3`` was actively harmful.**  ``eta*|grad L_n|^2`` scales with the
    #: learning rate and with the loss magnitude, so one constant cannot mean
    #: the same thing on two cases.  Measured on Example 2 at the initial
    #: iterate ``w_0 = (theta_true, f = 0)``, ``|grad L_n|^2 = 6.004e-01``, so
    #: ``eta*|grad|^2 = 6.004e-04`` at ``lr = 1e-3`` -- already below ``1e-3``.
    #: The rule therefore fired at epoch 0 and returned the *untrained* initial
    #: iterate, with nothing but a ``RuntimeWarning`` to say so.  Example 2's
    #: driver sidestepped it by passing ``--stopping-tau 0``, but anything
    #: taking this default silently got a model that had not been fitted.
    #:
    #: If the rule is wanted, set ``tau`` relative to the problem rather than
    #: to a constant -- ``_gradient_norm_sq`` is public and the initial value
    #: is the natural scale.
    stopping_tau: Optional[float] = None

    # --- bookkeeping --------------------------------------------------------------
    #: Which fields are counted in ``n``.  The paper counts both fields in
    #: Case 2 (``n = 2 * n_t * n_x * n_y``) but only one in Case 4
    #: (``n = n_t * n_x * n_y * n_z``); that asymmetry is reproduced here
    #: explicitly instead of being implicit.
    n_counted_fields: Optional[int] = None
    #: How many *grid points* enter ``n``.  ``None`` is the whole grid, which is
    #: every simulated case.  A case with a partial observation operator
    #: (:attr:`~semipde.pde.PDESpec.obs_index`) must set it to the number of
    #: observed points: the loss and the interval are defined on those, and a
    #: full-grid count would inflate ``n`` by the ratio and deflate ``se`` by
    #: its square root.
    n_counted_points: Optional[int] = None
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hidden = tuple(int(h) for h in self.hidden)
        if self.inputs is not None and self.d_y < 1:
            raise ValueError("d_y must be positive")

    # -- size, made explicit -------------------------------------------------------
    @property
    def n_observed_fields(self) -> int:
        return self.d_y if self.n_counted_fields is None else int(self.n_counted_fields)

    @property
    def n_traj(self) -> int:
        """Scalars per trajectory --- the paper's ``n``."""
        points = (self.grid.size if self.n_counted_points is None
                  else int(self.n_counted_points))
        return self.time.n_obs * points * self.n_observed_fields

    @property
    def n_total(self) -> int:
        """Scalars in the empirical loss --- the ``n`` of Theorems 2--4."""
        return self.M * self.n_traj

    @property
    def dim(self) -> int:
        return self.grid.dim

    # -- convenience ---------------------------------------------------------------
    def with_(self, **kwargs) -> "CaseConfig":
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)

    def size_table(self) -> str:
        """Human-readable summary of the three sizes, for logs and READMEs."""
        return (
            f"{self.name}: n_t={self.time.n_obs}, grid={self.grid.shape}, "
            f"points_counted={self.grid.size if self.n_counted_points is None else self.n_counted_points}, "
            f"fields_counted={self.n_observed_fields} -> n_traj={self.n_traj}; "
            f"M={self.M} -> n_total={self.n_total}"
        )
