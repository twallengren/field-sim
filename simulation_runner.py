import numpy as np
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

    def animate(self, field_names, alpha=0.5, cmap_list=None, absolute=False, split=False, interval=1):

        if isinstance(field_names, str):
            field_names = [field_names]

        if cmap_list is None:
            cmap_list = ['Reds', 'Greens', 'Blues'][:len(field_names)]

        # Determine color limits if absolute coloring is enabled
        vmin_vmax = {}
        if absolute:
            for name in field_names:
                all_frames = np.array([np.array(frame[name]) for frame in self.history])
                vmin_vmax[name] = (np.min(all_frames), np.max(all_frames))

        # Create subplots
        if split:
            fig, axes = plt.subplots(1, len(field_names), figsize=(5 * len(field_names), 5))
            if len(field_names) == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots()
            axes = [ax] * len(field_names)  # overlayed

        ims = []
        for i, name in enumerate(field_names):
            ax = axes[i]
            vmin, vmax = (None, None)
            if absolute:
                vmin, vmax = vmin_vmax[name]

            im = ax.imshow(
                self.history[0][name],
                cmap=cmap_list[i],
                origin='lower',
                alpha=alpha,
                vmin=vmin,
                vmax=vmax
            )
            ax.set_title(name)
            ims.append(im)

        def update(frame):
            for i, name in enumerate(field_names):
                ims[i].set_array(self.history[frame][name])

            progress = f"{100 * frame / (len(self.history) - 1):.1f}%"
            fig.suptitle(f"{self.config.name} – {progress} complete")
            return ims

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=interval, blit=True)
        plt.tight_layout()
        plt.show()

