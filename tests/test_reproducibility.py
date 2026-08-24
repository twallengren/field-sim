"""Phase 1 reproducibility test.

`get_config(seed=...)` must be a pure function of its seed: no module-level
RNG consumption, and identical seeds must produce bit-identical initial
conditions and bit-identical trajectories.
"""

import numpy as np

from fieldsim.field import Field
from fieldsim.lagrangian import Lagrangian
from fieldsim.simulations.agriculture import get_config
from fieldsim.simulator import Simulator
from fieldsim.stability import stable_dt

N_STEPS = 5


def _build_simulator(config):
    fields = {
        name: Field(name=name, **kwargs)
        for name, kwargs in config.field_defs.items()
    }

    lagrangian = Lagrangian()
    for term in config.lagrangian_terms:
        lagrangian.add_term(term)

    dx = next(iter(fields.values())).dx
    dt = stable_dt(
        fields,
        config.lagrangian_terms,
        config.flux_terms,
        config.sources,
        dx,
        safety=config.safety,
    )

    simulator = Simulator(
        fields=fields,
        lagrangian=lagrangian,
        sources=config.sources,
        flux_terms=config.flux_terms,
        dt=dt,
    )
    return simulator


def test_get_config_is_seed_deterministic():
    cfg1 = get_config(seed=42)
    cfg2 = get_config(seed=42)

    for name in cfg1.field_defs:
        field1 = Field(name=name, **cfg1.field_defs[name])
        field2 = Field(name=name, **cfg2.field_defs[name])
        assert np.array_equal(field1.get_values(), field2.get_values())


def test_short_run_is_seed_deterministic():
    cfg1 = get_config(seed=42)
    cfg2 = get_config(seed=42)

    sim1 = _build_simulator(cfg1)
    sim2 = _build_simulator(cfg2)

    for _ in range(N_STEPS):
        sim1.step()
        sim2.step()

    state1 = sim1.get_state()
    state2 = sim2.get_state()

    assert state1.keys() == state2.keys()
    for name in state1:
        assert np.array_equal(np.asarray(state1[name]), np.asarray(state2[name]))
