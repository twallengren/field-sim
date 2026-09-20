r"""Chemotaxis demo: pure transport, both totals exactly conserved.

This is the sources-off subset of the agriculture model, kept as the
conservation reference case:

.. math::

    \partial_t P = D_P \nabla^2 P - \nabla\cdot(\chi\, P\, \nabla F)

    \partial_t F = D_F \nabla^2 F

on ``[0, L]^2`` with zero-flux boundaries.  Both right-hand sides are pure
divergences of fluxes that vanish on the domain boundary:

* the face-difference diffusion energy omits boundary faces, so every interior
  face moves exactly the same amount out of one cell and into its neighbour;
* the flux divergence zero-pads the interior-face arrays and differences them,
  so the cell-wise contributions telescope.

Hence ``sum(P) dx^2`` and ``sum(F) dx^2`` are invariant to round-off, and the
positivity floor must never activate (donor-cell upwind plus the CFL bound
keeps ``P >= 0`` structurally).  Any drift in either total is a bug, which is
exactly what ``tests/test_configs.py`` asserts.

``P`` still climbs the gradient of ``F`` (``chi > 0``) while ``F`` itself
spreads out, so the population concentrates on the food maxima and then relaxes
as those maxima flatten -- visually interesting despite carrying no sources.
"""

import numpy as np

from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_config import SimulationConfig
from fieldsim.utils.constants import FOOD, POPULATION
from fieldsim.utils.generators import build_bump_function, generate_random_bump_specs

#: Domain edge length [km].
DOMAIN_KM = 10.0

D_P = 0.05   # [km^2/yr] population dispersal
D_F = 0.02   # [km^2/yr] food spreading (slower, so the gradient survives)
CHI = 0.3    # [km^2/food/yr] chemotactic mobility

DEFAULT_N = 96
DEFAULT_TOTAL_TIME = 5.0


def get_config(seed: int = 0, n: int = DEFAULT_N,
               total_time: float = DEFAULT_TOTAL_TIME,
               bc_type: str = "neumann") -> SimulationConfig:
    """Build the transport-only chemotaxis configuration.

    Args:
        seed: seed for the initial-condition RNG (pure function of its args).
        n: number of cells per edge (``dx = DOMAIN_KM / n``).
        total_time: physical duration to integrate, in years.
    """
    try:
        integer_n = int(n)
    except (TypeError, ValueError, OverflowError):
        integer_n = None
    if integer_n is None or integer_n != n or integer_n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}.")
    n = integer_n
    L = DOMAIN_KM
    dx = L / n
    bounds = ((0.0, L), (0.0, L))

    rng = np.random.default_rng(seed)

    pop_fn = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=12, bounds=bounds, amp_range=(0.5, 2.0),
            sigma_range=(0.3, 0.8),
        )
    )
    food_fn = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=6, bounds=bounds, amp_range=(0.5, 2.0),
            sigma_range=(0.8, 1.6),
        )
    )

    def initial_pop(x, y):
        return pop_fn(x, y)

    def initial_food(x, y):
        return food_fn(x, y)

    grid = {"shape": (n, n), "dx": dx, "bc_type": bc_type}

    return SimulationConfig(
        name="Chemotaxis (transport only)",
        field_defs={
            POPULATION: {
                **grid,
                "units": "people/km^2",
                "init_fn": initial_pop,
                "is_dynamic": True,
            },
            FOOD: {
                **grid,
                "units": "food/km^2",
                "init_fn": initial_food,
                "is_dynamic": True,
            },
        },
        lagrangian_terms=[
            Diffusion(target=POPULATION, alpha=D_P, bc_type=bc_type),
            Diffusion(target=FOOD, alpha=D_F, bc_type=bc_type),
        ],
        flux_terms=[
            AdvectionAlongGradientFlux(
                target_field=POPULATION, gradient_field=FOOD, kappa=CHI,
                bc_type=bc_type,
            ),
        ],
        sources=[],
        total_time=total_time,
    )
