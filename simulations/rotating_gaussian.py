from jax import numpy as jnp
from simulation_config import SimulationConfig
from lagrangians.gradient_energy import GradientEnergyTerm
from sources.rotating_gaussian_source import RotatingGaussianSource

def gaussian(x, y):
    return jnp.exp(-((x - 5)**2 + (y - 5)**2))

def get_config():
    return SimulationConfig(
        name="Gradient Flow of Rotating Gaussian",
        field_defs={
            "rho": {
                "shape": (100, 100),
                "dx": 0.1,
                "init_fn": gaussian,
                "is_dynamic": True
            }
        },
        lagrangian_terms=[
            GradientEnergyTerm()
        ],
        sources=[
            RotatingGaussianSource(
                target="rho",
                amplitude=10.0,
                sigma=0.3,
                radius=2.0,
                omega=1.0,
                dx=0.1
            )
        ],
        dt=0.01,
        steps=100
    )
