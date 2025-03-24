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
        return -self.gamma * (P-F) * sigmoid


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
