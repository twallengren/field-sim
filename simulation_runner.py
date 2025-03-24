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

    def animate(
            self,
            field_names,
            alpha=0.5,
            cmap_list=None,
            absolute=False,
            split=False,
            interval=1,
            figsize_per_plot=(4, 4),
            show_colorbars=False,
            fontsize=12,
            split_rows=None,
            split_cols=None,
    ):

        if isinstance(field_names, str):
            field_names = [field_names]

        if cmap_list is None or len(cmap_list) < len(field_names):
            default_cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys']
            cmap_list = (cmap_list or []) + default_cmaps[len(cmap_list or []):len(field_names)]

        # Determine color limits if absolute coloring is enabled
        vmin_vmax = {}
        if absolute:
            for name in field_names:
                all_frames = np.array([np.array(frame[name]) for frame in self.history])
                vmin_vmax[name] = (np.min(all_frames), np.max(all_frames))

        # Create subplots
        if split:
            num_fields = len(field_names)

            if split_rows is None and split_cols is None:
                split_rows = 1
                split_cols = num_fields
            elif split_rows is None:
                split_rows = int(np.ceil(num_fields / split_cols))
            elif split_cols is None:
                split_cols = int(np.ceil(num_fields / split_rows))

            fig, axes = plt.subplots(
                split_rows,
                split_cols,
                figsize=(figsize_per_plot[0] * split_cols, figsize_per_plot[1] * split_rows)
            )

            axes = np.array(axes).reshape(-1)
            axes = axes[:num_fields]
        else:
            fig, ax = plt.subplots()
            axes = [ax] * len(field_names)

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
            ax.set_title(name, fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 2)
            if show_colorbars:
                fig.colorbar(im, ax=ax, shrink=0.7)
            ims.append(im)

        def update(frame):
            for i, name in enumerate(field_names):
                ims[i].set_array(self.history[frame][name])
            progress = f"{100 * frame / (len(self.history) - 1):.1f}%"
            fig.suptitle(f"{self.config.name} – {progress} complete", fontsize=fontsize + 1)
            return ims

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=interval, blit=True)
        plt.tight_layout()
        plt.show()



