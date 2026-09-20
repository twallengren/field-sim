import jax.numpy as jnp
import numpy as np
import pytest

from fieldsim.catalog import ecology_parameter_defaults, get_preset, load_catalog
from fieldsim.initialization import initialize_civilization, initialize_ecology
from fieldsim.simulation_runner import SimulationRunner
from fieldsim.simulations.ecology import get_config, resolve_parameters
from fieldsim.sources.common import ecology_budget_terms
from fieldsim.utils.constants import (
    CULTIVATION,
    FERTILITY,
    FOOD,
    POPULATION,
    SOIL,
    WATER,
    WATER_SOURCES,
)


def test_ecology_catalog_is_separate_and_complete():
    catalog = load_catalog()
    assert len(catalog["parameters"]) == 13
    assert set(ecology_parameter_defaults()) == {
        "dp", "df", "dw", "chiFood", "chiWater", "growth", "yield",
        "cultivationScale", "waterConsumption", "harvestWaterCost",
        "consumption", "spoilage", "foodSupport", "waterSupport",
        "replenishmentRate", "sourceCapacity", "soilRecovery", "erosion",
        "settlementErosion",
    }
    expected = set(ecology_parameter_defaults())
    for preset_id in ("water_settlement", "water_overuse", "soil_recovery"):
        preset = get_preset(preset_id)
        assert preset["model"] == "ecology"
        assert set(preset["parameters"]) == expected


def test_ecology_initializer_is_deterministic_bounded_and_preserves_legacy_fields():
    ecology = initialize_ecology(11, 20, source_capacity=2.5, cultivation_scale=0.7)
    again = initialize_ecology(11, 20, source_capacity=2.5, cultivation_scale=0.7)
    legacy = initialize_civilization(11, 20)
    for name in ecology:
        np.testing.assert_array_equal(ecology[name], again[name])
    for name in (POPULATION, FOOD, FERTILITY, SOIL):
        np.testing.assert_array_equal(ecology[name], legacy[name])
    assert ecology[WATER_SOURCES].min() >= 0.0
    assert ecology[WATER_SOURCES].max() <= 1.0
    np.testing.assert_allclose(ecology[WATER], 2.5 * ecology[WATER_SOURCES])
    np.testing.assert_allclose(
        ecology[CULTIVATION], ecology[POPULATION] / (ecology[POPULATION] + 0.7)
    )


def test_ecology_sources_have_expected_dry_and_wet_limits():
    p = ecology_parameter_defaults()
    base = {
        POPULATION: jnp.full((2, 2), 2.0),
        FOOD: jnp.full((2, 2), 3.0),
        WATER: jnp.zeros((2, 2)),
        SOIL: jnp.full((2, 2), 0.5),
        FERTILITY: jnp.ones((2, 2)),
        WATER_SOURCES: jnp.ones((2, 2)),
    }
    dry = ecology_budget_terms(base, p)
    assert np.all(np.asarray(dry["harvest"]) == 0.0)
    assert np.all(np.asarray(dry["domesticUse"]) == 0.0)
    assert np.all(np.asarray(dry["agriculturalUse"]) == 0.0)
    assert np.all(np.asarray(dry["recharge"]) > 0.0)
    assert np.all(np.asarray(dry["population"]) < 0.0)
    assert np.all(np.asarray(dry["soilRecovery"]) == 0.0)

    wet = ecology_budget_terms({**base, WATER: jnp.full((2, 2), 100.0)}, p)
    assert np.all(np.asarray(wet["recharge"]) == 0.0)
    assert np.all(np.asarray(wet["harvest"]) > 0.0)
    assert np.all(np.asarray(wet["domesticUse"]) > 0.0)


@pytest.mark.parametrize("bc_type", ["neumann", "periodic"])
def test_ecology_runs_nonnegative_bounded_and_updates_derived_field(bc_type):
    runner = SimulationRunner(get_config(
        seed=4, n=20, total_time=3.0, bc_type=bc_type,
        preset="water_overuse",
    ), max_frames=4)
    initial_cultivation = np.asarray(runner.simulator.get_state()[CULTIVATION])
    runner.run()
    state = runner.simulator.get_state()
    for name, values in state.items():
        array = np.asarray(values)
        assert np.all(np.isfinite(array)), name
        assert array.min() >= 0.0, name
    assert np.asarray(state[SOIL]).max() <= 1.0
    assert np.asarray(state[WATER_SOURCES]).max() <= 1.0
    expected = np.asarray(state[POPULATION]) / (
        np.asarray(state[POPULATION]) + resolve_parameters("water_overuse")["cultivationScale"]
    )
    np.testing.assert_allclose(np.asarray(state[CULTIVATION]), expected)
    assert not np.allclose(initial_cultivation, expected)
    assert all(
        diagnostic["cumulative_truncated_mass"] == 0.0
        for diagnostic in runner.simulator.diagnostics.values()
    )


def test_water_balance_terms_close_one_explicit_step(x64):
    runner = SimulationRunner(get_config(
        seed=5, n=12, total_time=1.0, preset="water_settlement", adaptive=False
    ))
    simulator = runner.simulator
    before = simulator.get_state()
    terms = ecology_budget_terms(before, resolve_parameters("water_settlement"))
    dt = simulator.dt
    dx2 = runner.dx ** 2
    initial_mass = float(jnp.sum(before[WATER])) * dx2
    recharge = dt * float(jnp.sum(terms["recharge"])) * dx2
    domestic = dt * float(jnp.sum(terms["domesticUse"])) * dx2
    agricultural = dt * float(jnp.sum(terms["agriculturalUse"])) * dx2
    simulator.step()
    current_mass = float(jnp.sum(simulator.get_state()[WATER])) * dx2
    assert current_mass == pytest.approx(
        initial_mass + recharge - domestic - agricultural, rel=1e-12, abs=1e-12
    )


def test_presets_produce_distinct_resource_feedbacks(x64):
    summaries = {}
    for preset in ("water_settlement", "water_overuse", "soil_recovery"):
        runner = SimulationRunner(get_config(
            seed=7, n=16, total_time=20.0, preset=preset
        ), max_frames=2)
        runner.run()
        state = runner.simulator.get_state()
        summaries[preset] = {
            WATER: float(jnp.mean(state[WATER])),
            SOIL: float(jnp.mean(state[SOIL])),
        }
    assert summaries["water_overuse"][WATER] < summaries["water_settlement"][WATER]
    assert summaries["water_settlement"][SOIL] < 0.8
    assert summaries["soil_recovery"][SOIL] > 0.25


def test_overuse_grows_then_declines_and_depleted_soil_recovers(x64):
    overuse = SimulationRunner(get_config(
        seed=7, n=16, total_time=60.0, preset="water_overuse"
    ), max_frames=12)
    overuse.run()
    population = [float(np.mean(frame[POPULATION])) for frame in overuse.history]
    peak = max(population)
    assert peak > 2.5 * population[0]
    assert population[-1] < 0.8 * peak

    recovery = SimulationRunner(get_config(
        seed=7, n=16, total_time=20.0, preset="soil_recovery"
    ), max_frames=4)
    assert float(jnp.mean(recovery.simulator.get_state()[SOIL])) == pytest.approx(0.25)
    recovery.run()
    assert float(jnp.mean(recovery.simulator.get_state()[SOIL])) > 0.33


@pytest.mark.parametrize("key", [
    "cultivationScale", "foodSupport", "waterSupport", "sourceCapacity",
])
def test_ecology_denominators_must_be_positive(key):
    with pytest.raises(ValueError, match="must be positive"):
        resolve_parameters(parameters={key: 0.0})
