import numpy as np

from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_config import SimulationConfig
from fieldsim.sources.common import LogisticGrowthSource
from fieldsim.utils.constants import POPULATION, FOOD
from fieldsim.utils.generators import generate_random_bump_specs, build_bump_function


def get_config(seed: int = 0):
    grid_dim = 10  # each edge is grid_dim km long

    rng = np.random.default_rng(seed)

    pop_specs = generate_random_bump_specs(
        rng,
        n_bumps=10,
        bounds=((0, 10), (0, 10)),
        amp_range=(1, 10),
        sigma_range=(0.2, 0.8),
    )
    pop_bump_fn = build_bump_function(pop_specs)
    def initial_pop(x, y):
        return pop_bump_fn(x, y)

    food_specs = generate_random_bump_specs(
        rng,
        n_bumps=10,
        bounds=((0, 10), (0, 10)),
        amp_range=(20, 80),
        sigma_range=(0.4, 1.2),
    )
    food_bump_fn = build_bump_function(food_specs)
    def initial_food(x, y):
        return food_bump_fn(x, y)

    num_of_coordinates = 250
    num_years = 10
    days_per_frame = 5
    dt = days_per_frame / 365
    steps = int(num_years // dt)
    dx = grid_dim / num_of_coordinates

    return SimulationConfig(
        name="Population–Food Interaction",
        field_defs={
            POPULATION: {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_pop,
                "is_dynamic": True,
                "bc_type": "neumann",
                "vmin": 0.0,
            },
            FOOD: {
                "shape": (num_of_coordinates, num_of_coordinates),
                "dx": dx,
                "init_fn": initial_food,
                "is_dynamic": True,
                "bc_type": "neumann",
                "vmin": 0.0
            },
        },
        lagrangian_terms=[
            Diffusion(target=POPULATION, alpha=0.1),
            Diffusion(target=FOOD, alpha=0.1)
        ],
        flux_terms=[
            AdvectionAlongGradientFlux(target_field=POPULATION, gradient_field=FOOD, kappa=0.5),
        ],
        sources=[
            ### POPULATION ###
            LogisticGrowthSource(target=POPULATION, upper_limit=FOOD, gamma=0.1),
        ],
        dt=dt,
        steps=steps
    )
