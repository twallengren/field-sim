"""Phase 4: the jit-compiled step must be numerically indistinguishable.

``Simulator._step_values`` is a pure function of the state dict, wrapped once in
``jax.jit``.  These tests pin the two properties that matter:

* jit changes speed, not answers (``jax.disable_jit()`` runs the *same* Python
  function eagerly, so this is a genuine A/B of the compiled executable against
  the interpreter);
* the host-side guards, which now run on the ``check_every`` cadence instead of
  every step, still fire.

There is deliberately no wall-clock assertion here -- timing thresholds are the
classic flaky CI test.  The speed claim is measured by hand and recorded in the
commit; what is *tested* is that consecutive steps go through one compiled
executable without retracing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fieldsim.field import Field
from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.lagrangian import Lagrangian
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulator import Simulator
from fieldsim.sources.common import (
    ConsumptionSource,
    LogisticGrowthSource,
    RelaxationSource,
)
from fieldsim.stability import stable_dt
from fieldsim.utils.constants import FERTILITY, FOOD, POPULATION

# Same dynamics as the flagship agriculture config, at a test-sized grid.
D_P = 0.1
D_F = 0.1
CHI = 0.1
GAMMA = 0.5
R = 0.2
BETA = 0.02
L = 10.0


def _smooth_ics():
    """Smooth, strictly positive initial data (no floor activity expected)."""

    def pop_fn(X, Y):
        return 0.5 + 0.5 * jnp.exp(-((X - 0.3 * L) ** 2 + (Y - 0.5 * L) ** 2) / 2.0)

    def food_fn(X, Y):
        return 1.0 + 0.5 * jnp.exp(-((X - 0.7 * L) ** 2 + (Y - 0.5 * L) ** 2) / 2.0)

    def fert_fn(X, Y):
        return 1.0 + 0.25 * jnp.sin(2 * jnp.pi * X / L) * jnp.cos(2 * jnp.pi * Y / L)

    return {POPULATION: pop_fn, FOOD: food_fn, FERTILITY: fert_fn}


def _build(n=32, dt=None, dt_factor=1.0):
    """Full agriculture dynamics on an ``n x n`` grid."""
    dx = L / n
    ics = _smooth_ics()
    fields = {
        POPULATION: Field(POPULATION, (n, n), dx=dx, init_fn=ics[POPULATION]),
        FOOD: Field(FOOD, (n, n), dx=dx, init_fn=ics[FOOD]),
        FERTILITY: Field(
            FERTILITY, (n, n), dx=dx, init_fn=ics[FERTILITY], is_dynamic=False
        ),
    }
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
    lagrangian = Lagrangian()
    for term in lagrangian_terms:
        lagrangian.add_term(term)

    if dt is None:
        dt = stable_dt(fields, lagrangian_terms, flux_terms, sources, dx)
    dt *= dt_factor

    simulator = Simulator(
        fields=fields, lagrangian=lagrangian, sources=sources,
        flux_terms=flux_terms, dt=dt,
    )
    return simulator, dx


def _run(simulator, steps):
    for _ in range(steps):
        simulator.step()
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in simulator.get_state().items()
    }


def test_jit_consistency(x64):
    """Jitted and eager stepping agree to machine precision over 50 steps."""
    steps = 50

    jitted, _ = _build()
    eager, _ = _build()

    # Identical initial state, by construction *and* by check.
    for name, values in jitted.get_state().items():
        np.testing.assert_array_equal(
            np.asarray(values), np.asarray(eager.get_state()[name])
        )
    assert jitted.dt == eager.dt

    jit_state = _run(jitted, steps)
    with jax.disable_jit():
        eager_state = _run(eager, steps)

    for name in jit_state:
        np.testing.assert_allclose(
            jit_state[name], eager_state[name], atol=1e-12, rtol=0.0,
            err_msg=f"field {name!r} diverged between the jitted and eager paths",
        )

    # Non-trivial evolution: the test would be vacuous if nothing moved.
    initial = np.asarray(_build()[0].get_state()[POPULATION], dtype=np.float64)
    assert np.max(np.abs(jit_state[POPULATION] - initial)) > 1e-6

    assert jitted.step_count == eager.step_count == steps
    assert set(jitted.diagnostics) == set(eager.diagnostics)
    for name, diag in jitted.diagnostics.items():
        other = eager.diagnostics[name]
        assert set(diag) == {"cumulative_truncated_mass", "last_truncated_mass"}
        for key, value in diag.items():
            assert value == pytest.approx(other[key], abs=1e-12), (name, key)


def test_diagnostics_flush_on_read_mid_window():
    """Reading ``diagnostics`` synchronises steps taken since the last check."""
    simulator, dx = _build()
    assert simulator.check_every == 25

    for _ in range(7):  # deliberately not a multiple of check_every
        simulator.step()
    assert len(simulator._pending) == 7

    diagnostics = simulator.diagnostics
    assert simulator._pending == []
    for name, diag in diagnostics.items():
        # Smooth positive data: the floor should never have engaged.
        assert diag["cumulative_truncated_mass"] == 0.0, name
        assert diag["last_truncated_mass"] == 0.0, name

    # Same object identity on every read, so stashed references stay live.
    assert simulator.diagnostics is diagnostics


def test_guards_still_fire_under_jit():
    """An oversized dt stops the run within one check window."""
    simulator, _ = _build(dt_factor=20.0)

    with pytest.raises(RuntimeError):
        for _ in range(simulator.check_every):
            simulator.step()

    assert simulator.step_count <= simulator.check_every


def test_truncation_guard_reports_its_window():
    """An injected negative excursion is attributed to its check window."""
    simulator, _ = _build()

    for _ in range(3):
        simulator.step()
    values = simulator.fields[POPULATION].get_values()
    simulator.fields[POPULATION].set_values(values.at[5, 5].set(-1.0))
    simulator.step()  # step 4 is the offending one
    simulator.step()

    with pytest.raises(RuntimeError, match="Positivity floor truncated") as excinfo:
        simulator.sync_diagnostics()

    message = str(excinfo.value)
    assert "at step 4" in message
    assert "within the last check window (steps 1-5" in message


def test_repeated_steps_reuse_one_compiled_executable():
    """Assertion-free smoke: consecutive steps hit the same jit cache entry.

    Not a performance assertion (those are CI-fragile); this only pins that the
    step function is compiled once and reused, which is the whole point of the
    phase.  ``_cache_size()`` counts distinct traced signatures for the wrapped
    callable.
    """
    simulator, _ = _build(n=16)

    simulator.step()
    cache_after_first = simulator._jitted_step._cache_size()
    assert cache_after_first == 1

    for _ in range(4):
        simulator.step()

    assert simulator._jitted_step._cache_size() == cache_after_first
    assert simulator.step_count == 5

    state = simulator.get_state()
    for name, values in state.items():
        assert np.all(np.isfinite(np.asarray(values))), name
