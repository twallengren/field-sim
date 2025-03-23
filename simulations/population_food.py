from jax import numpy as jnp
from simulation_config import SimulationConfig
from lagrangians.population_food_terms import (
    PopulationDiffusion,
    FoodDiffusion,
)
from sources.population_food_sources import (
    PopulationGrowthSource,
    FoodConsumptionSource, FoodDecaySource, FoodLimitedPopulationDecaySource, ConstantFoodSource
)

def initial_pop(x, y):
    bump1 = 3 * jnp.exp(-((x - 1.5)**2 + (y - 5)**2))
    bump2 = 1 * jnp.exp(-((x - 8.5)**2 + (y - 5)**2))
    return bump1 + bump2

def initial_food(x, y):
    return 0*jnp.exp(-((x - 5)**2 + (y - 5)**2))

def farm_region(x, y):
    return (jnp.exp(-((x - 5)**2 + (y - 7.5)**2) / 0.5) > 0.5) | (jnp.exp(-((x - 5)**2 + (y - 2.5)**2) / 0.5) > 0.5)

def get_config():
    return SimulationConfig(
        name="Population–Food Interaction",
        field_defs={
            "pop": {
                "shape": (100, 100),
                "dx": 0.1,
                "init_fn": initial_pop,
                "is_dynamic": True,
                "bc_type": "neumann"
            },
            "food": {
                "shape": (100, 100),
                "dx": 0.1,
                "init_fn": initial_food,
                "is_dynamic": True,
                "bc_type": "neumann"
            }
        },
        lagrangian_terms=[
            PopulationDiffusion(alpha=1.0),
            FoodDiffusion(alpha=2.0),
        ],
        sources=[
            PopulationGrowthSource(target="pop", gamma=0.5),
            FoodLimitedPopulationDecaySource(target="pop", delta=0.1),
            FoodConsumptionSource(target="food", rho=1.0),
            FoodDecaySource(target="food", lamb=0.2),
            ConstantFoodSource(target="food", value=1000.0, mask_fn=farm_region)
        ],
        dt=1/365,
        steps=365*5
    )
