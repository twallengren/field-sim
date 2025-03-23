import jax.numpy as jnp

class LagrangianTerm:
    def __init__(self, name, expression_fn, coefficient=1.0):
        self.name = name
        self.expression_fn = expression_fn  # takes dict of Field objects
        self.coefficient = coefficient

    def evaluate(self, fields: dict) -> jnp.ndarray:
        """
        Evaluate this term across the entire grid.
        `fields` is a dict of {name: Field instance}
        """
        return self.coefficient * self.expression_fn(fields)
