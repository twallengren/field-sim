import jax.numpy as jnp
import numpy as np
import pytest

from fieldsim.field import Field
from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.initialization import Mulberry32, initialize_civilization
from fieldsim.lagrangian import Lagrangian
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_runner import SimulationRunner
from fieldsim.simulator import Simulator
from fieldsim.simulations.civilization import get_config
from fieldsim.source_term import SourceTerm
from fieldsim.sources.common import (
    CivilizationFoodSource,
    InfrastructureSource,
    SoilSource,
)
from fieldsim.utils.constants import (
    FERTILITY, FOOD, INFRASTRUCTURE, POPULATION, SOIL,
)


def test_mulberry_and_initializer_are_deterministic():
    rng = Mulberry32(0)
    assert [rng.random() for _ in range(3)] == pytest.approx(
        [0.26642920868471265, 0.0003297457005828619, 0.2232720274478197]
    )
    first = initialize_civilization(0x1_0000_0001, 16)
    second = initialize_civilization(1, 16)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert np.all(first[INFRASTRUCTURE] == 0.0)
    assert np.all(first[SOIL] == 1.0)


@pytest.mark.parametrize("bc_type", ["neumann", "periodic"])
def test_civilization_runs_nonnegative_and_soil_stays_bounded(bc_type):
    runner = SimulationRunner(get_config(
        seed=4, n=24, total_time=2.0, bc_type=bc_type,
        preset="overshoot",
    ), max_frames=8)
    runner.run()
    assert runner.simulator.time == pytest.approx(2.0)
    assert runner.history_times[-1] == pytest.approx(2.0)
    for name, values in runner.simulator.get_state().items():
        array = np.asarray(values)
        assert np.all(np.isfinite(array)), name
        assert array.min() >= 0.0, name
    soil = np.asarray(runner.simulator.get_state()[SOIL])
    assert soil.max() <= 1.0


def test_civilization_feedback_sources_have_expected_direction():
    shape = (2, 2)
    values = {
        POPULATION: jnp.full(shape, 2.0),
        FOOD: jnp.full(shape, 3.0),
        FERTILITY: jnp.ones(shape),
        SOIL: jnp.full(shape, 0.8),
        INFRASTRUCTURE: jnp.zeros(shape),
    }
    food_source = CivilizationFoodSource(
        FOOD, POPULATION, FERTILITY, SOIL, INFRASTRUCTURE,
        regrowth=0.4, consumption=0.1, investment=0.2,
        infra_boost=3.0, food_cost=0.2,
    )
    without_infra = np.asarray(food_source.evaluate(values))
    with_infra = np.asarray(food_source.evaluate({
        **values, INFRASTRUCTURE: jnp.full(shape, 2.0)
    }))
    assert np.all(with_infra > without_infra)

    infrastructure = InfrastructureSource(
        INFRASTRUCTURE, POPULATION, FOOD, investment=0.2, decay=0.03
    )
    assert np.all(np.asarray(infrastructure.evaluate(values)) > 0.0)
    soil = SoilSource(SOIL, POPULATION, recovery=0.02, erosion=0.1)
    assert np.all(np.asarray(soil.evaluate(values)) < 0.0)


def test_periodic_transport_conserves_mass_and_preserves_positivity(x64):
    n, dx = 12, 1.0 / 12
    rng = np.random.default_rng(3)
    population = jnp.asarray(rng.random((n, n)))
    attractant = jnp.asarray(rng.random((n, n)))
    flux = AdvectionAlongGradientFlux(
        POPULATION, FOOD, kappa=0.3, bc_type="periodic"
    )
    diffusion = Diffusion(POPULATION, 0.02, bc_type="periodic")
    lagrangian = Lagrangian()
    lagrangian.add_term(diffusion)
    values = {POPULATION: population, FOOD: attractant}
    rate = flux.max_rate(values, dx) + diffusion.max_rate(values, dx)
    dt = 0.8 / rate
    mass0 = float(population.sum())
    for _ in range(100):
        population = population + dt * (
            -lagrangian.functional_derivative(
                {POPULATION: population, FOOD: attractant}, dx, POPULATION
            )
            - flux.divergence({POPULATION: population, FOOD: attractant}, dx)
        )
    assert float(population.min()) >= 0.0
    assert float(population.sum()) == pytest.approx(mass0, rel=1e-12)


def test_adaptive_step_reacts_to_intervention_without_retrace():
    config = get_config(seed=0, n=16, total_time=1.0, preset="settlement")
    runner = SimulationRunner(config)
    simulator = runner.simulator
    simulator.step()
    cache_size = simulator._jitted_step._cache_size()
    old_dt = simulator.safe_timestep()

    # A sharp attractant intervention raises the donor-cell outflow bound.
    food = simulator.fields[FOOD].get_values()
    simulator.fields[FOOD].set_values(food.at[:, ::2].add(100.0))
    new_dt = simulator.safe_timestep()
    assert new_dt < old_dt
    simulator.step()
    assert simulator._jitted_step._cache_size() == cache_size
    assert simulator.dt == pytest.approx(new_dt)


def test_boundary_metadata_must_match():
    fields = {
        POPULATION: Field(POPULATION, (4, 4), bc_type="periodic"),
    }
    lagrangian = Lagrangian()
    lagrangian.add_term(Diffusion(POPULATION, 0.1, bc_type="neumann"))
    with pytest.raises(ValueError, match="Term .* boundary condition"):
        Simulator(fields, lagrangian, [], [], dt=0.01)


def test_tiny_advance_integrates_instead_of_only_moving_clock():
    field = Field("value", (2, 2))
    source = SourceTerm(
        "unit source", "value", lambda values: jnp.ones((2, 2)),
        max_rate_fn=lambda values: 0.0,
    )
    simulator = Simulator(
        {"value": field}, Lagrangian(), [source], [], adaptive=True
    )
    simulator.advance(1e-13)
    assert simulator.step_count == 1
    assert simulator.time == pytest.approx(1e-13)
    np.testing.assert_allclose(np.asarray(field.get_values()), 1e-13, rtol=1e-6)


@pytest.mark.parametrize("shape", [(2.5, 2), (float("inf"), 2)])
def test_field_requires_integer_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        Field("value", shape)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_truncation_tolerance_must_be_finite_nonnegative(tolerance):
    with pytest.raises(ValueError, match="truncation_tolerance"):
        Simulator(
            {"value": Field("value", (2, 2))}, Lagrangian(), [], [],
            dt=0.1, truncation_tolerance=tolerance,
        )
