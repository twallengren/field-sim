import jax.numpy as jnp

class FluxTerm:
    def __init__(self, name, target_field_name, flux_fn, coefficient=1.0):
        """
        Args:
            name: Descriptive name of the flux.
            target_field_name: The field whose continuity equation this affects.
            flux_fn: Function(fields) -> (Jx, Jy), each shape (ny, nx)
            coefficient: Scalar multiplier on the flux (default 1.0)
        """
        self.name = name
        self.target_field_name = target_field_name
        self.flux_fn = flux_fn
        self.coefficient = coefficient

    def divergence(self, fields: dict, dx: float) -> jnp.ndarray:
        """
        Compute ∇ · J, where J = (Jx, Jy), as a scalar field.
        """
        Jx, Jy = self.flux_fn(fields)
        div_x = jnp.gradient(Jx, dx, axis=1)
        div_y = jnp.gradient(Jy, dx, axis=0)
        return self.coefficient * (div_x + div_y)
