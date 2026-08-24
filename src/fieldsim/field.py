import jax.numpy as jnp
from jax.nn import softplus

class Field:
    def __init__(self, name, shape, dx=1.0, units=None, is_dynamic=True, init_fn=None, bc_type="neumann", vmin=None, vmax=None):
        self.name = name
        self.shape = shape
        self.dx = dx
        self.units = units
        self.is_dynamic = is_dynamic
        self.vmin = vmin
        self.vmax = vmax
        self.bc_type = bc_type
        self.values = self._initialize(init_fn)
        self._clip()

    def _initialize(self, fn):
        ny, nx = self.shape
        if fn is None:
            return jnp.zeros((ny, nx))

        x = jnp.arange(nx) * self.dx
        y = jnp.arange(ny) * self.dx
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        return fn(X, Y)

    def _clip(self):
        if self.vmin is not None and self.vmax is not None:
            # Use a smooth clipping based on scaled tanh
            k = 10.0  # Controls sharpness of transition
            span = self.vmax - self.vmin
            x_scaled = (self.values - self.vmin) / span
            x_squashed = jnp.tanh(k * (x_scaled - 0.5))  # maps smoothly into (-1, 1)
            x_shifted = 0.5 * (x_squashed + 1.0)  # maps into (0, 1)
            self.values = self.vmin + span * x_shifted
        elif self.vmin is not None:
            self.values = self.vmin + softplus(self.values - self.vmin)
        elif self.vmax is not None:
            self.values = self.vmax - softplus(self.vmax - self.values)

    def get_values(self):
        return self.values

    def set_values(self, new_values):
        self.values = new_values
        self._clip()

    def gradient(self):
        df_dy, df_dx = jnp.gradient(self.values, self.dx)
        return df_dx, df_dy

    def laplacian(self):
        df2_dx = jnp.gradient(jnp.gradient(self.values, self.dx, axis=1), self.dx, axis=1)
        df2_dy = jnp.gradient(jnp.gradient(self.values, self.dx, axis=0), self.dx, axis=0)
        return df2_dx + df2_dy

    def apply_bc(self):
        if self.bc_type == "neumann":
            self.apply_neumann_bc()
        elif self.bc_type == "leaky_neumann":
            self.apply_leaky_neumann_bc()
        else:
            raise ValueError(f"Unsupported boundary condition type: {self.bc_type}")
        self._clip()

    def apply_neumann_bc(self):
        f = self.values
        f = f.at[0, :].set(f[1, :])
        f = f.at[-1, :].set(f[-2, :])
        f = f.at[:, 0].set(f[:, 1])
        f = f.at[:, -1].set(f[:, -2])
        self.values = f

    def apply_leaky_neumann_bc(self, eta=0.1):
        f = self.values
        dx = self.dx
        f = f.at[:, 0].set(f[:, 1] - eta * dx)
        f = f.at[:, -1].set(f[:, -2] - eta * dx)
        f = f.at[0, :].set(f[1, :] - eta * dx)
        f = f.at[-1, :].set(f[-2, :] - eta * dx)
        self.values = f
