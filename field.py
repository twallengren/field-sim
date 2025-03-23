import jax.numpy as jnp

class Field:
    def __init__(self, name, shape, dx=1.0, units=None, is_dynamic=True, init_fn=None, bc_type="neumann"):
        self.name = name
        self.shape = shape
        self.dx = dx
        self.units = units
        self.is_dynamic = is_dynamic
        self.values = self._initialize(init_fn)
        self.bc_type = bc_type

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
        Returns df/dx, df/dy using second-order central differences.
        """
        df_dy, df_dx = jnp.gradient(self.values, self.dx)
        return df_dx, df_dy

    def laplacian(self):
        """
        Computes Laplacian using finite differences: ∇²f = d²f/dx² + d²f/dy²
        """
        df2_dx = jnp.gradient(jnp.gradient(self.values, self.dx, axis=1), self.dx, axis=1)
        df2_dy = jnp.gradient(jnp.gradient(self.values, self.dx, axis=0), self.dx, axis=0)
        return df2_dx + df2_dy

    def apply_bc(self):
        if self.bc_type == "neumann":
            self.apply_neumann_bc()
        elif self.bc_type == "leaky_neumann":
            self.apply_leaky_neumann_bc()
        else:
            raise ValueError(f"Unsupported boundary condition type: {type}")

    def apply_neumann_bc(self):
        f = self.values
        f = f.at[0, :].set(f[1, :])         # top
        f = f.at[-1, :].set(f[-2, :])       # bottom
        f = f.at[:, 0].set(f[:, 1])         # left
        f = f.at[:, -1].set(f[:, -2])       # right
        self.values = f

    def apply_leaky_neumann_bc(self, eta=0.1):
        """
        Applies a directional Neumann BC where the derivative normal to the boundary
        is set to -eta (i.e. controlled outflow), while the tangential derivative is 0.
        """
        f = self.values
        dx = self.dx

        # Left/right boundaries (∂f/∂x = ±η)
        f = f.at[:, 0].set(f[:, 1] - eta * dx)  # left (outflow: -η)
        f = f.at[:, -1].set(f[:, -2] - eta * dx)  # right (outflow: -η)

        # Top/bottom boundaries (∂f/∂y = ±η)
        f = f.at[0, :].set(f[1, :] - eta * dx)  # top
        f = f.at[-1, :].set(f[-2, :] - eta * dx)  # bottom

        self.values = f

