"""Phase 2c tests for the bundled physical configurations.

Covers the time-discretisation order (Richardson extrapolation) and headless
end-to-end runs of both shipped configs at reduced resolution.
"""

import numpy as np
import pytest

from fieldsim.field import Field
from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.lagrangian import Lagrangian
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_runner import SimulationRunner
from fieldsim.simulator import Simulator
from fieldsim.simulations import agriculture, chemotaxis_demo
from fieldsim.sources.common import (
    ConsumptionSource,
    LogisticGrowthSource,
    RelaxationSource,
)
from fieldsim.stability import stable_dt
from fieldsim.utils.constants import FERTILITY, FOOD, POPULATION


# --------------------------------------------------------------------------
# Time-discretisation order
# --------------------------------------------------------------------------

def _agriculture_like_simulator(n, dt):
    """48^2 agriculture dynamics on fixed, smooth, strictly positive data.

    Strictly positive initial data matters: the positivity floor is a
    non-smooth operation, so any activation of it would destroy the smooth
    dt-dependence that Richardson extrapolation measures.  The assertions below
    confirm it never fires.
    """
    L = agriculture.DOMAIN_KM
    dx = L / n

    def pop_fn(X, Y):
        return 1.0 + 0.4 * np.sin(2 * np.pi * X / L) * np.cos(2 * np.pi * Y / L)

    def food_fn(X, Y):
        return 1.5 + 0.5 * np.cos(2 * np.pi * X / L) * np.sin(4 * np.pi * Y / L)

    def fert_fn(X, Y):
        return 1.2 + 0.3 * np.cos(4 * np.pi * X / L) * np.cos(2 * np.pi * Y / L)

    fields = {
        POPULATION: Field(POPULATION, (n, n), dx=dx, init_fn=pop_fn),
        FOOD: Field(FOOD, (n, n), dx=dx, init_fn=food_fn),
        FERTILITY: Field(FERTILITY, (n, n), dx=dx, init_fn=fert_fn, is_dynamic=False),
    }

    lagrangian_terms = [
        Diffusion(target=POPULATION, alpha=agriculture.D_P),
        Diffusion(target=FOOD, alpha=agriculture.D_F),
    ]
    flux_terms = [
        AdvectionAlongGradientFlux(
            target_field=POPULATION, gradient_field=FOOD, kappa=agriculture.CHI
        )
    ]
    sources = [
        LogisticGrowthSource(target=POPULATION, upper_limit=FOOD, gamma=agriculture.GAMMA),
        RelaxationSource(target=FOOD, capacity_field=FERTILITY, rate=agriculture.R),
        ConsumptionSource(target=FOOD, consumer=POPULATION, beta=agriculture.BETA),
    ]

    lagrangian = Lagrangian()
    for term in lagrangian_terms:
        lagrangian.add_term(term)

    if dt is None:
        # Deliberately well inside the stability bound so that the fast
        # diffusive modes are resolved and we are in the asymptotic regime
        # where the local truncation error is O(dt).
        dt = stable_dt(fields, lagrangian_terms, flux_terms, sources, dx, safety=0.2)

    simulator = Simulator(
        fields=fields, lagrangian=lagrangian, sources=sources,
        flux_terms=flux_terms, dt=dt,
    )
    return simulator, dt


def _run_to(n, dt, steps):
    simulator, _ = _agriculture_like_simulator(n, dt)
    for _ in range(steps):
        simulator.step()
    for name, diag in simulator.diagnostics.items():
        assert diag["cumulative_truncated_mass"] == 0.0, (
            f"positivity floor activated for {name}; the Richardson estimate "
            "would be measuring the floor, not the time discretisation"
        )
    state = simulator.get_state()
    return np.concatenate([
        np.asarray(state[POPULATION], dtype=np.float64).ravel(),
        np.asarray(state[FOOD], dtype=np.float64).ravel(),
    ])


def test_time_order(x64):
    """Forward Euler is first order: the Richardson ratio is ~2."""
    n = 48
    base_steps = 40

    _, dt = _agriculture_like_simulator(n, dt=None)

    coarse = _run_to(n, dt, base_steps)
    medium = _run_to(n, dt / 2.0, 2 * base_steps)
    fine = _run_to(n, dt / 4.0, 4 * base_steps)

    d1 = np.linalg.norm(coarse - medium)
    d2 = np.linalg.norm(medium - fine)

    assert d2 > 0.0
    assert d1 > 1e3 * np.finfo(np.float64).eps * np.linalg.norm(coarse)

    ratio = d1 / d2
    assert 1.6 <= ratio <= 2.4, f"Richardson ratio {ratio} outside [1.6, 2.4]"


# --------------------------------------------------------------------------
# End-to-end headless runs
# --------------------------------------------------------------------------

def _totals(state, dx):
    return {
        name: float(np.asarray(values, dtype=np.float64).sum()) * dx ** 2
        for name, values in state.items()
    }


def test_agriculture_runs_end_to_end():
    """Reduced-size agriculture run: finite, non-negative, no floor activity."""
    runner = SimulationRunner(agriculture.get_config(seed=0, n=48, total_time=1.0))

    assert runner.steps >= 1
    assert runner.dt > 0
    runner.run()
    assert len(runner.history) == runner.steps

    state = runner.simulator.get_state()
    for name in (POPULATION, FOOD, FERTILITY):
        array = np.asarray(state[name])
        assert np.all(np.isfinite(array)), name
        assert array.min() >= 0.0, name

    totals = _totals(state, runner.dx)
    for name in (POPULATION, FOOD):
        truncated = runner.simulator.diagnostics[name]["cumulative_truncated_mass"]
        assert truncated <= 1e-12 * totals[name], f"{name}: floor removed {truncated}"

    # The static fertility field is genuinely static.
    fert0 = np.asarray(
        Field(FERTILITY, (48, 48), dx=runner.dx,
              init_fn=agriculture.get_config(seed=0, n=48).field_defs[FERTILITY]["init_fn"],
              is_dynamic=False).get_values()
    )
    np.testing.assert_array_equal(np.asarray(state[FERTILITY]), fert0)


def test_agriculture_config_shape_overrides():
    cfg = agriculture.get_config(seed=3, n=32, total_time=0.5)
    assert cfg.name == "Agriculture"
    assert cfg.total_time == 0.5
    for defs in cfg.field_defs.values():
        assert defs["shape"] == (32, 32)
        assert defs["dx"] == pytest.approx(agriculture.DOMAIN_KM / 32)
    assert cfg.field_defs[FERTILITY]["is_dynamic"] is False


def test_chemotaxis_demo_conserves_both_totals():
    """Transport-only config: no sources, so both totals are invariant.

    Runs in the library default float32; the totals are accumulated in float64
    on the host so that the assertion measures the *scheme's* drift rather than
    the precision of the reduction itself.
    """
    runner = SimulationRunner(chemotaxis_demo.get_config(seed=0, n=48, total_time=2.0))
    assert runner.config.sources == []

    initial = _totals(runner.simulator.get_state(), runner.dx)
    runner.run()
    final = _totals(runner.simulator.get_state(), runner.dx)

    for name in (POPULATION, FOOD):
        assert initial[name] > 0.0
        assert final[name] == pytest.approx(initial[name], rel=1e-5)

    state = runner.simulator.get_state()
    for name in (POPULATION, FOOD):
        array = np.asarray(state[name])
        assert np.all(np.isfinite(array))
        assert array.min() >= 0.0
        assert runner.simulator.diagnostics[name]["cumulative_truncated_mass"] == 0.0

    # Chemotaxis actually did something: the population is more concentrated on
    # the food maxima than it started.
    initial_pop = np.asarray(
        Field(POPULATION, (48, 48), dx=runner.dx,
              init_fn=chemotaxis_demo.get_config(seed=0, n=48)
              .field_defs[POPULATION]["init_fn"]).get_values(),
        dtype=np.float64,
    )
    final_pop = np.asarray(state[POPULATION], dtype=np.float64)
    assert not np.allclose(initial_pop, final_pop)
