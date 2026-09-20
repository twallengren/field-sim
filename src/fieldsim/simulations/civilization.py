"""Coupled population, food, infrastructure, and soil dynamics."""

from __future__ import annotations

import math

from fieldsim.catalog import get_preset, parameter_defaults
from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.initialization import DOMAIN_LENGTH, initialize_civilization
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_config import SimulationConfig
from fieldsim.sources.common import (
    CivilizationFoodSource,
    InfrastructureSource,
    LogisticGrowthSource,
    SoilSource,
)
from fieldsim.utils.constants import (
    FERTILITY,
    FOOD,
    INFRASTRUCTURE,
    POPULATION,
    SOIL,
)

DEFAULT_N = 64
DEFAULT_PRESET = "settlement"


def resolve_parameters(preset=DEFAULT_PRESET, parameters=None):
    """Resolve defaults, a catalog preset, and finite non-negative overrides.

    Catalog min/max values describe the browser controls. Python callers may
    intentionally explore finite values beyond those UI ranges.
    """
    selected = get_preset(preset)
    if selected["model"] != "civilization":
        raise ValueError(f"Preset {preset!r} is for model {selected['model']!r}.")
    resolved = parameter_defaults()
    resolved.update({key: float(value) for key, value in selected["parameters"].items()})
    if parameters:
        unknown = set(parameters) - set(resolved)
        if unknown:
            raise ValueError(f"Unknown civilization parameters: {sorted(unknown)}.")
        resolved.update({key: float(value) for key, value in parameters.items()})
    if any(not math.isfinite(value) or value < 0.0 for value in resolved.values()):
        raise ValueError("Civilization parameters must be finite and non-negative.")
    return resolved


def get_config(seed=0, n=DEFAULT_N, total_time=None, bc_type="neumann",
               preset=DEFAULT_PRESET, parameters=None, adaptive=True):
    """Build a civilization config from a catalog preset and optional overrides."""
    try:
        integer_n = int(n)
    except (TypeError, ValueError, OverflowError):
        integer_n = None
    if integer_n is None or integer_n != n or integer_n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}.")
    n = integer_n
    selected = get_preset(preset)
    p = resolve_parameters(preset, parameters)
    initial = initialize_civilization(seed, n, bc_type=bc_type)
    dx = DOMAIN_LENGTH / n
    grid = {"shape": (n, n), "dx": dx, "bc_type": bc_type}

    def init(name):
        values = initial[name]
        return lambda X, Y: values

    fields = {
        POPULATION: {**grid, "init_fn": init(POPULATION), "is_dynamic": True},
        FOOD: {**grid, "init_fn": init(FOOD), "is_dynamic": True},
        FERTILITY: {**grid, "init_fn": init(FERTILITY), "is_dynamic": False},
        INFRASTRUCTURE: {**grid, "init_fn": init(INFRASTRUCTURE), "is_dynamic": True},
        SOIL: {**grid, "init_fn": init(SOIL), "is_dynamic": True},
    }
    lagrangian_terms = [
        Diffusion(POPULATION, p["dp"], bc_type=bc_type),
        Diffusion(FOOD, p["df"], bc_type=bc_type),
    ]
    flux_terms = [
        AdvectionAlongGradientFlux(
            POPULATION, FOOD, p["chiFood"], bc_type=bc_type
        ),
        AdvectionAlongGradientFlux(
            POPULATION, INFRASTRUCTURE, p["chiInfra"], bc_type=bc_type
        ),
    ]
    sources = [
        LogisticGrowthSource(POPULATION, FOOD, gamma=p["growth"]),
        CivilizationFoodSource(
            FOOD, POPULATION, FERTILITY, SOIL, INFRASTRUCTURE,
            p["regrowth"], p["consumption"], p["investment"],
            p["infraBoost"], p["foodCost"],
        ),
        InfrastructureSource(
            INFRASTRUCTURE, POPULATION, FOOD, p["investment"], p["infraDecay"]
        ),
        SoilSource(SOIL, POPULATION, p["soilRecovery"], p["erosion"]),
    ]
    return SimulationConfig(
        name=f"Civilization — {selected['title']}",
        field_defs=fields,
        lagrangian_terms=lagrangian_terms,
        flux_terms=flux_terms,
        sources=sources,
        total_time=float(selected["duration"] if total_time is None else total_time),
        adaptive=adaptive,
        safety=0.8,
        max_dt=0.1,
    )
