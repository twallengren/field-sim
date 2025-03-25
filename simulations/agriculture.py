from jax import numpy as jnp

from flux_terms.agriculture_flux_terms import PopulationFertilityAttractionFlux
from flux_terms.common import AdvectionAlongGradientFlux
from lagrangians.common import Diffusion
from simulation_config import SimulationConfig
from sources.common import ResourceLimitedDecaySource, LimitedGrowthSource, ResourceConsumptionSource
from utils.constants import POPULATION, FERTILITY, INDUSTRY, FOOD
from utils.generators import generate_random_bump_specs, build_bump_function

grid_dim = 10 # each edge is grid_dim km long

def initial_pop(x, y):
    return jnp.zeros_like(x, dtype=jnp.float32) + 1.0

def initial_industry(x, y):
    return jnp.zeros_like(x, dtype=jnp.float32)

# Create bumps in a 10x10 space
specs = generate_random_bump_specs(
    n_bumps=10,
    bounds=((0, 10), (0, 10)),
    amp_range=(20, 80),
    sigma_range=(0.4, 1.2),
    mode="cartesian"
)
bump_fn = build_bump_function(specs)
def initial_fertility(x, y):
    return bump_fn(x, y, t=0).astype(jnp.float32)

def initial_food(x, y):
    return initial_fertility(x, y)

def get_config():

    num_of_coordinates = 100
    num_years = 5
    days_per_frame = 5
    dt = days_per_frame / 365
    steps = int(num_years // dt)
    dx = grid_dim/num_of_coordinates

    return SimulationConfig(
        name="Population–Food Interaction",
        field_defs={
            POPULATION: {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_pop,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
            FERTILITY: {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_fertility,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
            INDUSTRY: {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_industry,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
            FOOD: {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_food,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
        },
        lagrangian_terms=[
            Diffusion(target=POPULATION, alpha=0.01)
        ],
        flux_terms=[
            ### POPULATION FLUXES ###
            AdvectionAlongGradientFlux(target_field=POPULATION, gradient_field=FOOD, kappa=0.5, scale_by_target=False),
        ],
        sources=[
            ### POPULATION SOURCES ###
            LimitedGrowthSource(target=POPULATION, upper_limit=FOOD, gamma=0.1),
            ResourceLimitedDecaySource(target=POPULATION, resource=FOOD, gamma=1.0),

            ### FOOD SOURCES ###
            ResourceConsumptionSource(target=FOOD, consumer=POPULATION, rho=0.1)
        ],
        dt=dt,
        steps=steps
    )
