import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from field import Field
from lagrangian import Lagrangian
from simulator import Simulator
from source_term import SourceTerm


# Assuming these are imported from your local module
# from your_module import Field, Lagrangian, LagrangianTerm, SourceTerm, Simulator

class GradientEnergyTerm:
    def evaluate(self, fields):
        f = fields["rho"].get_values()
        df_dx, df_dy = fields["rho"].gradient()
        return 0.5 * (df_dx**2 + df_dy**2)

def gaussian_init(x, y):
    return jnp.exp(-((x - 5)**2 + (y - 5)**2))

def main():
    # Create initial field
    field = Field(name="rho", shape=(100, 100), dx=0.1, init_fn=gaussian_init)
    fields = {"rho": field}

    # Lagrangian setup
    lagrangian = Lagrangian()
    lagrangian.add_term(GradientEnergyTerm())

    # Source terms
    sources = [
        SourceTerm(
            name="ZeroSource",
            target_field_name="rho",
            expression_fn=lambda fields: jnp.zeros_like(fields["rho"].get_values())+1
        )
    ]

    # Simulation setup
    sim = Simulator(fields=fields, lagrangian=lagrangian, sources=sources, dt=0.01)

    # Run and store results
    n_steps = 100
    results = []

    for _ in range(n_steps):
        sim.step()
        results.append(sim.get_state()["rho"])

    # Visualize as video
    fig, ax = plt.subplots()
    im = ax.imshow(results[0], cmap='viridis', origin='lower', interpolation='nearest')

    def update(frame):
        im.set_array(results[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(results), interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
