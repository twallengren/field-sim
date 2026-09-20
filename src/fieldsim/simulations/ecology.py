"""Coupled settlement ecology with food, local water, and soil feedbacks."""

from __future__ import annotations

import math

from fieldsim.catalog import ecology_parameter_defaults, get_preset
from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.initialization import DOMAIN_LENGTH, initialize_ecology
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_config import SimulationConfig
from fieldsim.sources.common import (
    EcologyFoodSource,
    EcologyPopulationSource,
    EcologySoilSource,
    EcologyWaterSource,
)
from fieldsim.utils.constants import (
    CULTIVATION,
    FERTILITY,
    FOOD,
    POPULATION,
    SOIL,
    WATER,
    WATER_SOURCES,
)

DEFAULT_N = 64
DEFAULT_PRESET = "water_settlement"
_POSITIVE_PARAMETERS = {"cultivationScale", "foodSupport", "waterSupport", "sourceCapacity"}


def resolve_parameters(preset=DEFAULT_PRESET, parameters=None):
    """Resolve an ecology preset and finite, non-negative Python overrides."""
    selected = get_preset(preset)
    if selected["model"] != "ecology":
        raise ValueError(f"Preset {preset!r} is for model {selected['model']!r}.")
    resolved = ecology_parameter_defaults()
    resolved.update({key: float(value) for key, value in selected["parameters"].items()})
    if parameters:
        unknown = set(parameters) - set(resolved)
        if unknown:
            raise ValueError(f"Unknown ecology parameters: {sorted(unknown)}.")
        resolved.update({key: float(value) for key, value in parameters.items()})
    if any(not math.isfinite(value) or value < 0.0 for value in resolved.values()):
        raise ValueError("Ecology parameters must be finite and non-negative.")
    if any(resolved[key] <= 0.0 for key in _POSITIVE_PARAMETERS):
        raise ValueError(
            "cultivationScale, foodSupport, waterSupport, and sourceCapacity "
            "must be positive."
        )
    return resolved


def get_config(seed=0, n=DEFAULT_N, total_time=None, bc_type="neumann",
               preset=DEFAULT_PRESET, parameters=None, adaptive=True):
    """Build the ecology simulation from a catalog preset and overrides."""
    try:
        integer_n = int(n)
    except (TypeError, ValueError, OverflowError):
        integer_n = None
    if integer_n is None or integer_n != n or integer_n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}.")
    n = integer_n
    selected = get_preset(preset)
    p = resolve_parameters(preset, parameters)
    initial = initialize_ecology(
        seed, n, bc_type=bc_type,
        source_capacity=p["sourceCapacity"],
        cultivation_scale=p["cultivationScale"],
    )
    # The gallery scenarios start from distinct, reproducible conditions: the
    # overuse story begins with a small settlement that can grow before demand
    # draws water down, while the recovery story begins on depleted land.
    if preset == "water_overuse":
        initial[POPULATION] = 0.35 * initial[POPULATION]
        initial[CULTIVATION] = (
            initial[POPULATION] / (initial[POPULATION] + p["cultivationScale"])
        )
    elif preset == "soil_recovery":
        initial[SOIL] = 0.25 * initial[SOIL]
    dx = DOMAIN_LENGTH / n
    grid = {"shape": (n, n), "dx": dx, "bc_type": bc_type}

    def init(name):
        values = initial[name]
        return lambda X, Y: values

    fields = {
        POPULATION: {**grid, "init_fn": init(POPULATION), "is_dynamic": True},
        FOOD: {**grid, "init_fn": init(FOOD), "is_dynamic": True},
        WATER: {**grid, "init_fn": init(WATER), "is_dynamic": True},
        SOIL: {**grid, "init_fn": init(SOIL), "is_dynamic": True},
        FERTILITY: {**grid, "init_fn": init(FERTILITY), "is_dynamic": False},
        WATER_SOURCES: {
            **grid, "init_fn": init(WATER_SOURCES), "is_dynamic": False,
        },
    }
    lagrangian_terms = [
        Diffusion(POPULATION, p["dp"], bc_type=bc_type),
        Diffusion(FOOD, p["df"], bc_type=bc_type),
        Diffusion(WATER, p["dw"], bc_type=bc_type),
    ]
    flux_terms = [
        AdvectionAlongGradientFlux(
            POPULATION, FOOD, p["chiFood"], bc_type=bc_type
        ),
        AdvectionAlongGradientFlux(
            POPULATION, WATER, p["chiWater"], bc_type=bc_type
        ),
    ]
    sources = [
        EcologyPopulationSource(p),
        EcologyFoodSource(p),
        EcologyWaterSource(p),
        EcologySoilSource(p),
    ]
    return SimulationConfig(
        name=f"Settlement ecology — {selected['title']}",
        field_defs=fields,
        lagrangian_terms=lagrangian_terms,
        flux_terms=flux_terms,
        sources=sources,
        total_time=float(selected["duration"] if total_time is None else total_time),
        adaptive=adaptive,
        safety=0.8,
        max_dt=0.1,
        derived_fields={
            CULTIVATION: lambda values: (
                values[POPULATION] / (values[POPULATION] + p["cultivationScale"])
            )
        },
    )
