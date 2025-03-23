import jax.numpy as jnp

class SourceTerm:
    def __init__(self, name, target_field_name, expression_fn, coefficient=1.0):
        """
        Parameters:
        - name: str — description for debugging
        - target_field_name: str — which field this term updates
        - expression_fn: Callable[[Dict[str, Field]], jnp.ndarray]
        - coefficient: float — scaling factor
        """
        self.name = name
        self.target = target_field_name
        self.expression_fn = expression_fn
        self.coefficient = coefficient

    def evaluate(self, fields: dict) -> jnp.ndarray:
        """
        Evaluate source/sink term on the grid using current field states.
        """
        return self.coefficient * self.expression_fn(fields)

    def __repr__(self):
        return f"SourceTerm(name={self.name}, target={self.target}, coeff={self.coefficient})"
