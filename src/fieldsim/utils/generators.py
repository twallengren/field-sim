import jax.numpy as jnp
import numpy as np

def generate_random_bump_specs(
    n_bumps,
    bounds=((0, 10), (0, 10)),
    amp_range=(10, 100),
    sigma_range=(0.3, 1.0),
    mode="cartesian",
    center=(5, 5),
    r_range=(1, 4),
    theta_range=(0, 2 * np.pi),
):
    specs = []

    for _ in range(n_bumps):
        amp = np.random.uniform(*amp_range)
        sigma = np.random.uniform(*sigma_range)

        if mode == "cartesian":
            x = np.random.uniform(*bounds[0])
            y = np.random.uniform(*bounds[1])
            specs.append({
                "type": "cartesian",
                "x": x,
                "y": y,
                "amp": amp,
                "sigma": sigma
            })

        elif mode == "polar":
            r = np.random.uniform(*r_range)
            theta = np.random.uniform(*theta_range)
            specs.append({
                "type": "polar",
                "center": center,
                "r": r,
                "theta": theta,
                "amp": amp,
                "sigma": sigma
            })

        else:
            raise ValueError("Mode must be 'cartesian' or 'polar'")

    return specs


def build_bump_function(bump_specs):
    def bump_fn(x, y, t=0):
        result = jnp.zeros_like(x)

        for spec in bump_specs:
            amp = spec.get("amp", 1.0)
            sigma = spec.get("sigma", 1.0)

            if spec["type"] == "cartesian":
                x0 = spec["x"](t) if callable(spec["x"]) else spec["x"]
                y0 = spec["y"](t) if callable(spec["y"]) else spec["y"]

            elif spec["type"] == "polar":
                cx, cy = spec["center"]
                r = spec["r"](t) if callable(spec["r"]) else spec["r"]
                theta = spec["theta"](t) if callable(spec["theta"]) else spec["theta"]
                x0 = cx + r * jnp.cos(theta)
                y0 = cy + r * jnp.sin(theta)

            else:
                raise ValueError(f"Unknown bump type: {spec['type']}")

            bump = amp * jnp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            result += bump

        return result

    return bump_fn
