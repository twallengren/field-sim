import jax.numpy as jnp

from source_term import SourceTerm
from utils.constants import POPULATION, FOOD


### POPULATION SOURCES ###

class LimitedGrowthSource(SourceTerm):
    def __init__(self, target=POPULATION, upper_limit=FOOD, gamma=1.0):

        def expression_fn(fields):
            T = fields[target].get_values()
            U = fields[upper_limit].get_values()
            sigmoid = 1 / (1 + jnp.exp(-10 * (U - T)))
            return T * sigmoid

        super().__init__(name=f"{target} growth", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)

class DecaySource(SourceTerm):
    def __init__(self, target=POPULATION, gamma=1.0):

        def expression_fn(fields):
            T = fields[target].get_values()
            return -T

        super().__init__(name=f"{target} decay", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)
        self.gamma = gamma

class ResourceLimitedDecaySource(SourceTerm):
    def __init__(self, target=POPULATION, resource=FOOD, gamma=1.0):

        def expression_fn(fields):
            T = fields[target].get_values()
            R = fields[resource].get_values()
            sigmoid = 1 / (1 + jnp.exp(-10 * (T - R)))
            return -(T - R) * sigmoid

        super().__init__(name=f"{resource} limited {target} decay", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)
        self.gamma = gamma

### FOOD SOURCES ###

class ResourceConsumptionSource(SourceTerm):
    def __init__(self, target=FOOD, consumer=POPULATION, rho=1.0):

        def expression_fn(fields):
            C = fields[consumer].get_values()
            T = fields[target].get_values()
            sigmoid = 1 / (1 + jnp.exp(-10 * (T - 0.1)))
            return -C * sigmoid

        super().__init__(name="Food Consumption", target_field_name=target, expression_fn=expression_fn, coefficient=rho)
