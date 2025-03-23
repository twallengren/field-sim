import jax.numpy as jnp
from jax import vmap

import jax.numpy as jnp

class Field:
    def __init__(self, name, shape, dx=1.0, units=None, is_dynamic=True, init_fn=None):
        """
        Represents a scalar field defined over a 2D spatial domain.

        Args:
            name: Identifier for the field.
            shape: Tuple (ny, nx) giving grid size.
            dx: Grid spacing (assumed uniform).
            units: Optional string for units (for clarity).
            is_dynamic: Whether this field evolves over time.
            init_fn: Function f(x, y) to generate initial values. If None, defaults to zeros.
        """
        self.name = name
        self.shape = shape
        self.dx = dx
        self.units = units
        self.is_dynamic = is_dynamic

        # Generate initial condition from init_fn(x, y)
        self.values = self._initialize(init_fn)

    def _initialize(self, fn):
        ny, nx = self.shape
        if fn is None:
            return jnp.zeros((ny, nx))

        x = jnp.arange(nx) * self.dx
        y = jnp.arange(ny) * self.dx
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        return fn(X, Y)

    def get_values(self):
        return self.values

    def set_values(self, new_values):
        self.values = new_values

    def gradient(self):
        """
        Compute spatial gradient using central differences.
        Returns: tuple (df/dx, df/dy), each with shape = self.shape
        """
        f = self.values
        dx = self.dx

        df_dx = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dx)
        df_dy = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)

        return df_dx, df_dy

    def apply_neumann_bc(self):
        """
        Apply zero-flux (Neumann) boundary conditions by copying inward neighbors.
        """
        f = self.values
        f = f.at[0, :].set(f[1, :])           # top
        f = f.at[-1, :].set(f[-2, :])         # bottom
        f = f.at[:, 0].set(f[:, 1])           # left
        f = f.at[:, -1].set(f[:, -2])         # right
        self.values = f
