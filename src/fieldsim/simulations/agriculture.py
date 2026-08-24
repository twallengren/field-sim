r"""Agriculture: population and food on a fertile landscape.

Governing system
----------------
On the square domain ``[0, L]^2`` with ``L = 10 km``, time in years, and
zero-flux (homogeneous Neumann) boundaries on every field:

.. math::

    \partial_t P = D_P \nabla^2 P
                 - \nabla \cdot \left( \chi\, P\, \nabla F \right)
                 + \gamma\, P\, \frac{F - P}{F + P + \epsilon}

    \partial_t F = D_F \nabla^2 F
                 + r\,\bigl(K(x) - F\bigr)
                 - \beta\, P\, F

Fields
    ``P`` population density        [people / km^2]   (dynamic)
    ``F`` food density              [food / km^2]     (dynamic)
    ``K`` fertility / soil capacity [food / km^2]     (static)

Parameters and units
    ``D_P, D_F``  [km^2 / yr]                  random dispersal
    ``chi``       [km^2 / (food) / yr]         chemotactic mobility
    ``gamma``     [1 / yr]                     max per-capita growth rate
    ``r``         [1 / yr]                     soil regrowth rate
    ``beta``      [km^2 / person / yr]         per-capita consumption rate
    ``epsilon``   [food / km^2]                logistic regularisation, 1e-6

Signs and interpretation
    * ``-div(chi P grad F)`` with ``chi > 0`` moves people **up** the food
      gradient (attraction).  The flux is proportional to ``P``, so it vanishes
      where nobody lives; it is discretised donor-cell upwind, so it can never
      drain a cell below zero.
    * The growth term has per-capita rate ``gamma (F-P)/(F+P+eps)``, bounded in
      ``[-gamma, +gamma]``.  Equilibria are ``P = 0`` and ``P = F``: population
      grows toward the locally available food and starves (rate ``-gamma``)
      where there is none.  It is division-safe at ``F = 0``.
    * ``r(K - F)`` is soil regrowth toward the local fertility.  Being linear
      rather than logistic, it revives fully depleted cells.
    * ``-beta P F`` is consumption; it vanishes at ``F = 0`` and at ``P = 0``.

Conservation
    Both transport operators are exactly conservative (interior-face energy for
    diffusion, telescoping face-flux divergence for chemotaxis), so

        d/dt sum(P) dx^2 = sum of the growth source,
        d/dt sum(F) dx^2 = sum of regrowth minus consumption,

    with no boundary leakage.  See ``chemotaxis_demo`` for the sources-off case
    where both totals are conserved to machine precision.

Numerics
    Explicit Euler with ``dt`` derived from the combined stability/positivity
    bound (see :mod:`fieldsim.stability`)

        dt * ( 4 D_max/dx^2 + (max|u| + max|v|)/dx + s_max ) = 0.8

    At the default resolution ``n = 128`` (``dx = 0.078125 km``) diffusion
    dominates the bound and ``dt ~ 0.011 yr``, i.e. ~900 steps for a 10 year
    run.  The parameters below were chosen so that this bound is comfortable:
    the original configuration used food amplitudes of O(80), which pushed the
    advective rate to ~3000/yr and made the hardcoded ``dt = 5/365`` roughly
    3.4x over even the diffusion limit.
"""

import numpy as np

from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.lagrangians.common import Diffusion
from fieldsim.simulation_config import SimulationConfig
from fieldsim.sources.common import (
    ConsumptionSource,
    LogisticGrowthSource,
    RelaxationSource,
)
from fieldsim.utils.constants import FERTILITY, FOOD, POPULATION
from fieldsim.utils.generators import build_bump_function, generate_random_bump_specs

#: Domain edge length [km].
DOMAIN_KM = 10.0

#: Physical parameters (see module docstring for units).
D_P = 0.1
D_F = 0.1
CHI = 0.1
GAMMA = 0.5
R = 0.2
BETA = 0.02

#: Baseline fertility so that no part of the map is permanently dead.
FERTILITY_FLOOR = 0.25

DEFAULT_N = 128
DEFAULT_TOTAL_TIME = 10.0


def get_config(seed: int = 0, n: int = DEFAULT_N,
               total_time: float = DEFAULT_TOTAL_TIME) -> SimulationConfig:
    """Build the agriculture simulation configuration.

    Args:
        seed: seed for the initial-condition RNG. ``get_config`` is a pure
            function of its arguments; no module-level RNG is consumed.
        n: number of cells per edge (``dx = DOMAIN_KM / n``).
        total_time: physical duration to integrate, in years.
    """
    L = DOMAIN_KM
    dx = L / n
    bounds = ((0.0, L), (0.0, L))

    rng = np.random.default_rng(seed)

    pop_fn = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=10, bounds=bounds, amp_range=(0.5, 2.0),
            sigma_range=(0.3, 0.9),
        )
    )
    food_fn = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=10, bounds=bounds, amp_range=(0.5, 2.0),
            sigma_range=(0.5, 1.3),
        )
    )
    fertility_bumps = build_bump_function(
        generate_random_bump_specs(
            rng, n_bumps=8, bounds=bounds, amp_range=(0.5, 2.0),
            sigma_range=(0.8, 2.0),
        )
    )

    def initial_pop(x, y):
        return pop_fn(x, y)

    def initial_food(x, y):
        return food_fn(x, y)

    def fertility(x, y):
        return FERTILITY_FLOOR + fertility_bumps(x, y)

    grid = {"shape": (n, n), "dx": dx, "bc_type": "neumann"}

    return SimulationConfig(
        name="Agriculture",
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
            FERTILITY: {
                **grid,
                "units": "food/km^2",
                "init_fn": fertility,
                "is_dynamic": False,
            },
        },
        lagrangian_terms=[
            Diffusion(target=POPULATION, alpha=D_P),
            Diffusion(target=FOOD, alpha=D_F),
        ],
        flux_terms=[
            # chi > 0: people move up the food gradient.
            AdvectionAlongGradientFlux(
                target_field=POPULATION, gradient_field=FOOD, kappa=CHI
            ),
        ],
        sources=[
            LogisticGrowthSource(target=POPULATION, upper_limit=FOOD, gamma=GAMMA),
            RelaxationSource(target=FOOD, capacity_field=FERTILITY, rate=R),
            ConsumptionSource(target=FOOD, consumer=POPULATION, beta=BETA),
        ],
        total_time=total_time,
    )
