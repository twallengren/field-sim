from jax import numpy as jnp
from simulation_config import SimulationConfig
from lagrangians.population_food_terms import (
    PopulationDiffusion,
    FoodDiffusion, FoodTransportToInfraTerm,
)
from sources.population_food_sources import (
    PopulationGrowthSource,
    FoodConsumptionSource, FoodDecaySource, ConstantFoodSource, PopulationDecaySource, FoodLimitedPopulationDecaySource,
    ResourceConsumptionSource, PopulationResourceGrowth, InfrastructureBuildSource
)
from utils.generators import generate_random_bump_specs, build_bump_function

grid_dim = 10 # each edge is grid_dim km long

def initial_pop(x, y):
    bump = 10 * jnp.exp(-((x - grid_dim/2)**2 + (y - grid_dim/2)**2))
    return bump

def initial_food(x, y):
    return 0*jnp.exp(-((x - grid_dim/2)**2 + (y - grid_dim/2)**2))

# Create bumps in a 10x10 space
specs = generate_random_bump_specs(
    n_bumps=10,
    bounds=((0, 10), (0, 10)),
    amp_range=(20, 80),
    sigma_range=(0.4, 1.2),
    mode="cartesian"
)
bump_fn = build_bump_function(specs)
def initial_resources(x, y): return bump_fn(x, y, t=0)

def farm_region(x, y, t):
    tol = 0.5
    period = 5*365
    x0 = 3*jnp.cos(2*jnp.pi*t/period) + grid_dim/2
    y0 = 3*jnp.sin(2*jnp.pi*t/period) + grid_dim/2
    return jnp.exp(-((x - x0)**2 + (y - y0)**2) / 0.1) > tol

class SourceMask:

    def __init__(self, mask_fn, step=1):
        self.mask_fn = mask_fn
        self.t = 0
        self.step = step

    def call_function(self, x, y):
        value = self.mask_fn(x, y, self.t)
        self.t += self.step
        return value


def get_config():

    num_of_coordinates = 100
    num_years = 5
    days_per_frame = 5
    dt = days_per_frame / 365
    steps = int(num_years // dt)
    dx = grid_dim/num_of_coordinates
    source_mask = SourceMask(farm_region, days_per_frame)

    return SimulationConfig(
        name="Population–Food Interaction",
        field_defs={
            "pop": {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_pop,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
            "food": {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_food,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
            "res": {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_resources,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
            "infra": {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": lambda x, y: jnp.zeros_like(x),  # Initialize with zeros
                "is_dynamic": True,
                "bc_type": "neumann"
            }
        },
        lagrangian_terms=[
            PopulationDiffusion(alpha=0.6),
            FoodDiffusion(alpha=0.7),
            FoodTransportToInfraTerm(kappa=0.01)
        ],
        flux_terms=[],
        sources=[
            PopulationGrowthSource(target="pop", gamma=1.0),
            PopulationDecaySource(target="pop", gamma=0.1),
            FoodLimitedPopulationDecaySource(target="pop", gamma=1.0),
            PopulationResourceGrowth(target="pop", gamma=2.0),
            FoodConsumptionSource(target="food", rho=1.0),
            FoodDecaySource(target="food", lamb=0.1),
            ConstantFoodSource(target="food", value=10.0, mask_fn=None),#source_mask.call_function),
            ResourceConsumptionSource(target="res", rate=0.01),
            InfrastructureBuildSource(target="infra", eta=0.1, decay=0.01),
        ],
        dt=dt,
        steps=steps
    )
