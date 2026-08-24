import jax.numpy as jnp
import numpy as np


def generate_random_bump_specs(
    rng: np.random.Generator,
    n_bumps,
    bounds=((0, 10), (0, 10)),
    amp_range=(10, 100),
    sigma_range=(0.3, 1.0),
):
    specs = []

    for _ in range(n_bumps):
        amp = rng.uniform(*amp_range)
        sigma = rng.uniform(*sigma_range)
        x = rng.uniform(*bounds[0])
        y = rng.uniform(*bounds[1])
        specs.append({
            "type": "cartesian",
            "x": x,
            "y": y,
            "amp": amp,
            "sigma": sigma
        })

    return specs


def build_bump_function(bump_specs):
    def bump_fn(x, y):
        result = jnp.zeros_like(x)

        for spec in bump_specs:
            amp = spec.get("amp", 1.0)
            sigma = spec.get("sigma", 1.0)
            x0 = spec["x"]
            y0 = spec["y"]

            bump = amp * jnp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            result += bump

        return result

    return bump_fn
