"""Phase 2b Field/Simulator integration tests.

These exercise the full step loop: derived timestep, operators, monitored
positivity floor, and the runtime guards.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fieldsim.field import Field
from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.lagrangian import Lagrangian
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_config import SimulationConfig
from fieldsim.simulator import Simulator
from fieldsim.sources.common import (
    ConsumptionSource,
    LogisticGrowthSource,
    RelaxationSource,
)
from fieldsim.stability import n_steps, rate_sum, stable_dt
from fieldsim.utils.constants import FERTILITY, FOOD, POPULATION
from fieldsim.utils.generators import build_bump_function, generate_random_bump_specs

# Parameters of the flagship agriculture system, used here to build small
# variants of the same dynamics.
D_P = 0.1
D_F = 0.1
CHI = 0.1
GAMMA = 0.5
R = 0.2
BETA = 0.02
L = 10.0


def _agriculture_terms():
    lagrangian_terms = [
        Diffusion(target=POPULATION, alpha=D_P),
        Diffusion(target=FOOD, alpha=D_F),
    ]
    flux_terms = [
        AdvectionAlongGradientFlux(
            target_field=POPULATION, gradient_field=FOOD, kappa=CHI
        ),
    ]
    sources = [
        LogisticGrowthSource(target=POPULATION, upper_limit=FOOD, gamma=GAMMA),
        RelaxationSource(target=FOOD, capacity_field=FERTILITY, rate=R),
        ConsumptionSource(target=FOOD, consumer=POPULATION, beta=BETA),
    ]
    return lagrangian_terms, flux_terms, sources


def _build(n, init_fns, safety=0.8, dt=None):
    dx = L / n
    fields = {
        POPULATION: Field(POPULATION, (n, n), dx=dx, init_fn=init_fns[POPULATION]),
        FOOD: Field(FOOD, (n, n), dx=dx, init_fn=init_fns[FOOD]),
        FERTILITY: Field(
            FERTILITY, (n, n), dx=dx, init_fn=init_fns[FERTILITY], is_dynamic=False
        ),
    }
    lagrangian_terms, flux_terms, sources = _agriculture_terms()
    lagrangian = Lagrangian()
    for term in lagrangian_terms:
        lagrangian.add_term(term)

    if dt is None:
        dt = stable_dt(fields, lagrangian_terms, flux_terms, sources, dx, safety=safety)

    simulator = Simulator(
        fields=fields,
        lagrangian=lagrangian,
        sources=sources,
        flux_terms=flux_terms,
        dt=dt,
    )
    return simulator, dx


# --------------------------------------------------------------------------
# The defining property the old softplus clamp destroyed
# --------------------------------------------------------------------------


def test_zero_stays_zero():
    """Zero initial data is an exact fixed point of the full agriculture system.

    The previous ``Field._clip`` (``vmin + softplus(v - vmin)``) fabricated
    ``ln 2 ~= 0.693`` in every cell of an all-zero field on every application.
    With that gone, the right-hand side vanishes identically at zero: diffusion
    of zero is zero, the donor-cell flux carries a zero density, the bounded
    logistic carries a factor P, consumption carries a factor F, and relaxation
    toward a zero capacity is zero.
    """
    zero = lambda X, Y: jnp.zeros_like(X)
    simulator, dx = _build(
        32,
        {POPULATION: zero, FOOD: zero, FERTILITY: zero},
        # No term imposes a positive advective/consumption rate here, but the
        # diffusion rate does, so stable_dt is well defined.
    )

    for _ in range(100):
        simulator.step()

    state = simulator.get_state()
    for name, values in state.items():
        array = np.asarray(values)
        assert np.array_equal(array, np.zeros_like(array)), f"{name} left zero"

    for name, diag in simulator.diagnostics.items():
        assert diag["cumulative_truncated_mass"] == 0.0, name
        assert diag["last_truncated_mass"] == 0.0, name


def test_nonnegativity_full_dynamics():
    """Full agriculture dynamics stay non-negative with negligible floor action."""
    n = 48
    rng = np.random.default_rng(7)

    pop_fn = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=8, bounds=((0, L), (0, L)), amp_range=(0.5, 2.0),
            sigma_range=(0.4, 1.2),
        )
    )
    food_fn = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=8, bounds=((0, L), (0, L)), amp_range=(0.5, 2.0),
            sigma_range=(0.6, 1.5),
        )
    )
    fert_fn = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=6, bounds=((0, L), (0, L)), amp_range=(0.5, 2.0),
            sigma_range=(0.8, 2.0),
        )
    )

    simulator, dx = _build(
        n, {POPULATION: pop_fn, FOOD: food_fn, FERTILITY: fert_fn}
    )

    for _ in range(300):
        simulator.step()

    state = simulator.get_state()
    for name in (POPULATION, FOOD):
        array = np.asarray(state[name])
        assert np.all(np.isfinite(array)), name
        assert array.min() >= 0.0, f"{name} min {array.min()}"

        mass = float(array.sum()) * dx ** 2
        assert mass > 0.0, name
        truncated = simulator.diagnostics[name]["cumulative_truncated_mass"]
        assert truncated < 1e-10 * mass, f"{name} truncated {truncated} of {mass}"


# --------------------------------------------------------------------------
# Validation and guards
# --------------------------------------------------------------------------


def test_bad_bc_type_raises_at_field_construction():
    with pytest.raises(ValueError, match="boundary condition"):
        Field("phi", (8, 8), dx=0.1, bc_type="leaky_neumann")


def test_field_rejects_bad_shape_and_dx():
    with pytest.raises(ValueError, match="shape"):
        Field("phi", (8,), dx=0.1)
    with pytest.raises(ValueError, match="dx must be positive"):
        Field("phi", (8, 8), dx=0.0)


def test_field_is_cell_centered():
    n, dx = 4, 0.25
    field = Field("phi", (n, n), dx=dx, init_fn=lambda X, Y: X)
    row = np.asarray(field.get_values())[0]
    np.testing.assert_allclose(row, (np.arange(n) + 0.5) * dx, rtol=1e-6)
    # Domain is [0, L], not [0, L - dx]: the last centre sits dx/2 from the edge.
    assert row[-1] == pytest.approx(n * dx - dx / 2)


def test_mismatched_shape_or_dx_raises():
    lagrangian = Lagrangian()
    lagrangian.add_term(Diffusion(target=POPULATION, alpha=D_P))

    def make(shape_a, dx_a, shape_b, dx_b):
        return Simulator(
            fields={
                POPULATION: Field(POPULATION, shape_a, dx=dx_a),
                FOOD: Field(FOOD, shape_b, dx=dx_b),
            },
            lagrangian=lagrangian,
            sources=[],
            flux_terms=[],
            dt=1e-3,
        )

    with pytest.raises(ValueError, match="one grid shape"):
        make((8, 8), 0.1, (16, 16), 0.1)
    with pytest.raises(ValueError, match="one grid spacing"):
        make((8, 8), 0.1, (8, 8), 0.2)


def test_simulator_rejects_nonpositive_dt():
    lagrangian = Lagrangian()
    with pytest.raises(ValueError, match="dt must be positive"):
        Simulator(
            fields={POPULATION: Field(POPULATION, (8, 8), dx=0.1)},
            lagrangian=lagrangian,
            sources=[],
            flux_terms=[],
            dt=0.0,
        )


def test_total_time_and_steps_validation():
    with pytest.raises(ValueError, match="total_time must be positive"):
        SimulationConfig(
            name="bad", field_defs={}, lagrangian_terms=[], flux_terms=[],
            sources=[], total_time=0.0,
        )
    with pytest.raises(ValueError, match="total_time must be positive"):
        n_steps(-1.0, 0.1)
    with pytest.raises(ValueError, match="dt must be positive"):
        n_steps(1.0, 0.0)
    assert n_steps(1.0, 0.3) == 4       # ceil, never truncating the run short
    assert n_steps(1e-9, 1.0) == 1      # always at least one step


def _smooth_ics():
    def pop_fn(X, Y):
        return 0.5 + 0.5 * jnp.exp(-((X - 0.3 * L) ** 2 + (Y - 0.5 * L) ** 2) / 2.0)

    def food_fn(X, Y):
        return 1.0 + 0.5 * jnp.exp(-((X - 0.7 * L) ** 2 + (Y - 0.5 * L) ** 2) / 2.0)

    def fert_fn(X, Y):
        return 1.0 + 0.25 * jnp.sin(2 * jnp.pi * X / L) * jnp.cos(2 * jnp.pi * Y / L)

    return {POPULATION: pop_fn, FOOD: food_fn, FERTILITY: fert_fn}


def test_oversized_dt_is_rejected_at_construction():
    """A dt above the bound raises in ``__init__``, before any step is taken."""
    ics = _smooth_ics()
    reference, _ = _build(32, ics)
    oversized = 20.0 * reference.dt

    with pytest.raises(RuntimeError, match="violates the stability bound"):
        _build(32, ics, dt=oversized)


def test_oversized_dt_trips_the_runtime_guard():
    """A dt that becomes unsafe *after* construction still stops the run.

    Construction now validates the timestep, so the only way to reach the
    periodic guard is a dt that was safe for the initial state and is not safe
    later.  That is exactly what state-dependent (advective/consumption) rates
    can do; here it is provoked deterministically by assigning to
    ``simulator.dt`` after construction, which ``step`` honours by recompiling.
    """
    ics = _smooth_ics()
    simulator, _ = _build(32, ics)
    simulator.dt = 20.0 * simulator.dt

    with pytest.raises(RuntimeError):
        for _ in range(25):
            simulator.step()

    assert simulator.step_count <= 25


def test_stability_guard_reports_dt_violation_directly():
    """check_state() itself rejects a dt above the bound, with a clear message."""
    ics = _smooth_ics()
    reference, _ = _build(32, ics)
    simulator, _ = _build(32, ics)
    simulator.dt = 5.0 * simulator.dt

    with pytest.raises(RuntimeError, match="violates the stability bound"):
        simulator.check_state()

    # ... and the safe timestep passes the same guard.
    reference.check_state()


def test_stability_guard_detects_non_finite_values():
    ics = _smooth_ics()
    simulator, _ = _build(32, ics)
    simulator.fields[POPULATION].set_values(
        simulator.fields[POPULATION].get_values().at[0, 0].set(jnp.nan)
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        simulator.check_state()


def test_derived_dt_matches_the_rate_bound():
    ics = _smooth_ics()
    simulator, dx = _build(48, ics, safety=0.8)
    values = simulator.get_state()
    total = rate_sum(
        values, simulator.lagrangian.terms, simulator.flux_terms,
        simulator.sources, dx,
    )
    assert simulator.dt * total == pytest.approx(0.8, rel=1e-9)

    # Diffusion alone already accounts for most of the bound at this resolution.
    assert total >= 4.0 * max(D_P, D_F) / dx ** 2


def test_stable_dt_requires_some_dynamics():
    with pytest.raises(ValueError, match="no term imposes a positive rate"):
        stable_dt({POPULATION: jnp.zeros((8, 8))}, [], [], [], 0.1)


def test_floor_diagnostics_record_injected_negative_mass():
    """The floor is *monitored*: a genuine negative excursion is reported.

    Since the step function is jit-compiled and its floor measurement stays on
    device, the guard is evaluated per step but *reported* when the pending
    window is synchronised -- here, explicitly.
    """
    ics = _smooth_ics()
    simulator, dx = _build(32, ics)

    values = simulator.fields[POPULATION].get_values()
    injected = -1.0
    simulator.fields[POPULATION].set_values(values.at[5, 5].set(injected))

    simulator.step()
    with pytest.raises(RuntimeError, match="Positivity floor truncated"):
        simulator.sync_diagnostics()
