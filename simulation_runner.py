import matplotlib.pyplot as plt
import matplotlib.animation as animation

from field import Field
from lagrangian import Lagrangian
from simulation_config import SimulationConfig
from simulator import Simulator


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.fields = {
            name: Field(name=name, **kwargs)
            for name, kwargs in config.field_defs.items()
        }

        self.lagrangian = Lagrangian()
        for term in config.lagrangian_terms:
            self.lagrangian.add_term(term)

        self.simulator = Simulator(
            fields=self.fields,
            lagrangian=self.lagrangian,
            sources=config.sources,
            dt=config.dt
        )
        self.history = []

    def run(self):
        for _ in range(self.config.steps):
            self.simulator.step()
            self.history.append(self.simulator.get_state())

    def animate(self, field_names, alpha=0.5, cmap_list=None):
        if isinstance(field_names, str):
            field_names = [field_names]

        if cmap_list is None:
            cmap_list = ['Reds', 'Greens', 'Blues'][:len(field_names)]

        fig, ax = plt.subplots()

        ims = []
        for i, name in enumerate(field_names):
            im = ax.imshow(self.history[0][name], cmap=cmap_list[i], origin='lower', alpha=alpha)
            ims.append(im)

        def update(frame):
            for i, name in enumerate(field_names):
                ims[i].set_array(self.history[frame][name])
            return ims

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=50, blit=True)
        plt.title(self.config.name)
        plt.show()

