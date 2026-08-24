"""Phase 0 packaging smoke test.

Verifies that the core fieldsim modules import cleanly under the new
src-layout package. This is intentionally shallow (no behavior assertions
about numerics) — deeper tests land in later phases.

Note: the legacy, import-broken population/food simulation config module
and its dead source-term classes were deleted in Phase 1.
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
