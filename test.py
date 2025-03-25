import jax.numpy as jnp
import matplotlib.pyplot as plt
from field import Field  # Your Field class with clipping logic

# Initial condition function: generates values from -2 to +3
def initial_fertility(x, y):
    return x - 2  # Values range from -2 to 3 across the domain

# Define a mock config
dx = 0.1
config = {
    "fertility": {
        "shape": (100, 100),
        "dx": dx,
        "init_fn": initial_fertility,
        "is_dynamic": True,
        "bc_type": "neumann",
        "vmin": 0.0,
        "vmax": 1.0
    }
}

# Create field from config
fields = {
    name: Field(name=name, **kwargs)
    for name, kwargs in config.items()
}

# Extract fertility field
fertility_field = fields["fertility"]

# Plot
plt.figure(figsize=(10, 4))

# Original (unclipped) values for reference
x = jnp.arange(100) * dx
X, Y = jnp.meshgrid(x, x, indexing="xy")
original = initial_fertility(X, Y)

plt.subplot(1, 2, 1)
plt.title("Initial (Unclipped) Fertility")
plt.imshow(original, origin="lower", cmap="coolwarm")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Field Values (Clipped)")
plt.imshow(fertility_field.get_values(), origin="lower", cmap="coolwarm")
plt.colorbar()

plt.tight_layout()
plt.show()
