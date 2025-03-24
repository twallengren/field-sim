import jax.numpy as jnp

from source_term import SourceTerm

class PopulationGrowthSource(SourceTerm):
    def __init__(self, target, gamma=1.0):
        super().__init__(name="Population Growth", target_field_name=target, expression_fn=None)
        self.gamma = gamma

    def evaluate(self, fields):
        P = fields["pop"].get_values()
        F = fields["food"].get_values()
        sigmoid = 1 / (1 + jnp.exp(-10 * (F - P)))
        return self.gamma * P * sigmoid

class PopulationDecaySource(SourceTerm):
    def __init__(self, target, gamma=1.0):
        super().__init__(name="Population Decay", target_field_name=target, expression_fn=None)
        self.gamma = gamma

    # death rate
    def evaluate(self, fields):
        P = fields["pop"].get_values()
        return -self.gamma * P

class FoodLimitedPopulationDecaySource(SourceTerm):
    def __init__(self, target, gamma=1.0):
        super().__init__(name="Population Decay", target_field_name=target, expression_fn=None)
        self.gamma = gamma

    # If P-F is greater than 0, we decay
    def evaluate(self, fields):
        P = fields["pop"].get_values()
        F = fields["food"].get_values()
        sigmoid = 1 / (1 + jnp.exp(-10 * (P - F)))
        return -self.gamma * P * (P - F + 1) * sigmoid

class PopulationResourceGrowth(SourceTerm):
    def __init__(self, target, gamma=1.0):
        super().__init__(name="Population Resource Growth", target_field_name=target, expression_fn=None)
        self.gamma = gamma

    def evaluate(self, fields):
        P = fields["pop"].get_values()
        R = fields["res"].get_values()
        sigmoid = 1 / (1 + jnp.exp(-10 * (R - 0.5)))
        return self.gamma * P * sigmoid

class FoodConsumptionSource(SourceTerm):
    def __init__(self, target, rho=1.0):
        super().__init__(name="Food Consumption", target_field_name=target, expression_fn=None)
        self.rho = rho

    def evaluate(self, fields):
        P = fields["pop"].get_values()
        F = fields["food"].get_values()
        sigmoid = 1 / (1 + jnp.exp(-10 * (F - 0.5)))
        return -self.rho * P * sigmoid


class FoodDecaySource(SourceTerm):
    def __init__(self, target, lamb=0.05):
        super().__init__(name="Food Decay", target_field_name=target, expression_fn=None)
        self.lamb = lamb

    def evaluate(self, fields):
        F = fields["food"].get_values()
        return -self.lamb * F

class ConstantFoodSource(SourceTerm):
    def __init__(self, target, value=1.0, mask_fn=None):
        """
        Constant source of food per unit area.
        If mask_fn is provided, it's applied as a spatial filter (returns 0 or 1).
        """
        self.value = value
        self.mask_fn = mask_fn
        super().__init__(name="Constant Food Source", target_field_name=target, expression_fn=None)

    def evaluate(self, fields):
        F = fields[self.target].get_values()
        shape = F.shape

        # Default: uniform source everywhere
        source = jnp.ones(shape) * self.value

        # If masked, apply spatial mask
        if self.mask_fn is not None:
            dx = fields[self.target].dx
            x = jnp.arange(shape[1]) * dx
            y = jnp.arange(shape[0]) * dx
            X, Y = jnp.meshgrid(x, y, indexing="xy")
            mask = self.mask_fn(X, Y)
            source *= mask

        return source

class ResourceConsumptionSource(SourceTerm):
    def __init__(self, target, rate=1.0):
        """
        Models resource depletion due to population presence:
        dR/dt = -rate * P
        """
        self.rate = rate
        super().__init__(name="Resource Consumption", target_field_name=target, expression_fn=None)

    def evaluate(self, fields):
        P = fields["pop"].get_values()
        return -self.rate * P


class RenewableResourceRegenerationSource(SourceTerm):
    def __init__(self, target, regen_rate=0.1, carrying_capacity=100.0):
        """
        Logistic resource regeneration:
        dR/dt = regen_rate * R * (1 - R / carrying_capacity)
        """
        self.regen_rate = regen_rate
        self.carrying_capacity = carrying_capacity
        super().__init__(name="Renewable Resource Regeneration", target_field_name=target, expression_fn=None)

    def evaluate(self, fields):
        R = fields[self.target].get_values()
        return self.regen_rate * R * (1 - R / self.carrying_capacity)

import jax.numpy as jnp
from source_term import SourceTerm

class InfrastructureBuildSource(SourceTerm):
    def __init__(self, target, eta=1.0, decay=0.01):
        """
        Infrastructure growth: dI/dt = eta * P * R - decay * I
        """
        self.eta = eta
        self.decay = decay
        super().__init__(name="Infrastructure Growth", target_field_name=target, expression_fn=None)

    def evaluate(self, fields):
        P = fields["pop"].get_values()
        R = fields["res"].get_values()
        I = fields[self.target].get_values()
        return self.eta * P * R - self.decay * I
