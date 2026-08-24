"""Phase 0 packaging smoke test.

Verifies that the core fieldsim modules import cleanly under the new
src-layout package. This is intentionally shallow (no behavior assertions
about numerics) — deeper tests land in later phases.

Note: fieldsim.simulations.population_food is import-broken (references
modules deleted upstream) and is intentionally excluded here; it is slated
for deletion in Phase 1.
"""

import importlib


MODULES = [
    "fieldsim",
    "fieldsim.field",
    "fieldsim.simulator",
    "fieldsim.lagrangian",
    "fieldsim.flux_term",
    "fieldsim.source_term",
    "fieldsim.lagrangian_term",
    "fieldsim.simulation_runner",
    "fieldsim.simulation_config",
    "fieldsim.utils.generators",
    "fieldsim.lagrangians.common",
    "fieldsim.flux_terms.common",
    "fieldsim.sources.common",
    "fieldsim.simulations.agriculture",
]


def test_modules_import():
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None
