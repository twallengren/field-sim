import jax.numpy as jnp

from source_term import SourceTerm


class RotatingGaussianSource(SourceTerm):
    def __init__(self, target, amplitude=1.0, sigma=0.5, radius=3.0, omega=1.0, dx=0.1):
        super().__init__(name="Rotating Source", target_field_name=target, expression_fn=None)
        self.amplitude = amplitude
        self.sigma = sigma
        self.radius = radius
        self.omega = omega
        self.dx = dx
        self.t = 0.0  # internal time counter

    def evaluate(self, fields):
        f = fields[self.target].get_values()
        ny, nx = f.shape

        x = jnp.arange(nx) * self.dx
        y = jnp.arange(ny) * self.dx
        X, Y = jnp.meshgrid(x, y, indexing="xy")

        # Grid center
        x_center = (nx - 1) * self.dx / 2
        y_center = (ny - 1) * self.dx / 2

        # Rotating peak center
        x0 = x_center + self.radius * jnp.cos(self.omega * self.t)
        y0 = y_center + self.radius * jnp.sin(self.omega * self.t)

        G = self.amplitude * jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * self.sigma**2))

        self.t += self.dx  # crude proxy for time step

        return G