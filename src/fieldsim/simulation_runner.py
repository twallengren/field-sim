import math

import numpy as np

from fieldsim.field import Field
from fieldsim.lagrangian import Lagrangian
from fieldsim.simulation_config import SimulationConfig
from fieldsim.simulator import Simulator
from fieldsim.stability import n_steps, stable_dt

#: Default cycling palette used by ``animate`` when the caller does not (or
#: does not fully) specify ``cmap_list``.  Cycled with modulo, so any number
#: of fields is supported without an ``IndexError``.
_DEFAULT_CMAPS = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys']


class SimulationRunner:
    """Builds the fields/operators from a config, derives dt, and integrates.

    History is bounded: ``run()`` records at most ``max_frames + 1`` snapshots
    (step 0 plus every ``stride``-th step, always including the final step),
    each stored as a float32 host (``numpy``) array, rather than one full-size
    JAX array of every single step.  Per-field running min/max are tracked as
    the snapshots are recorded, so ``animate(absolute=True)`` never has to
    materialise the whole history into one array just to find its extrema.
    """

    def __init__(self, config: SimulationConfig, max_frames: int = 150):
        if not max_frames > 0:
            raise ValueError(f"max_frames must be positive, got {max_frames!r}.")

        self.config = config
        self.max_frames = int(max_frames)
        self.fields = {
            name: Field(name=name, **kwargs)
            for name, kwargs in config.field_defs.items()
        }

        self.lagrangian = Lagrangian()
        for term in config.lagrangian_terms:
            self.lagrangian.add_term(term)

        dx_values = {float(f.dx) for f in self.fields.values()}
        if len(dx_values) != 1:
            raise ValueError(
                f"All fields must share one grid spacing dx; got {sorted(dx_values)}."
            )
        self.dx = dx_values.pop()

        # The timestep comes from the physics, evaluated on the initial state.
        try:
            derived_dt = stable_dt(
                self.fields,
                config.lagrangian_terms,
                config.flux_terms,
                config.sources,
                self.dx,
                safety=config.safety,
            )
        except ValueError as exc:
            if not config.adaptive or "no term imposes a positive rate" not in str(exc):
                raise
            derived_dt = config.max_dt
        self.dt = min(config.max_dt, derived_dt) if config.adaptive else derived_dt
        self.steps = n_steps(config.total_time, self.dt)

        self.simulator = Simulator(
            fields=self.fields,
            lagrangian=self.lagrangian,
            sources=config.sources,
            flux_terms=config.flux_terms,
            dt=self.dt,
            adaptive=config.adaptive,
            safety=config.safety,
            max_dt=config.max_dt,
        )
        self.history = []
        self.history_times = []
        self.history_steps = []
        self.diagnostics = None
        self.stride = max(1, math.ceil(self.steps / self.max_frames))
        self._running_min = {}
        self._running_max = {}
        self._ani = None

    def _snapshot(self):
        """Record one float32 host-array snapshot and update running extrema."""
        frame = {}
        for name, array in self.simulator.get_state().items():
            values = np.asarray(array, dtype=np.float32)
            frame[name] = values
            local_min = float(values.min())
            local_max = float(values.max())
            if name in self._running_min:
                self._running_min[name] = min(self._running_min[name], local_min)
                self._running_max[name] = max(self._running_max[name], local_max)
            else:
                self._running_min[name] = local_min
                self._running_max[name] = local_max
        self.history.append(frame)
        self.history_times.append(float(self.simulator.time))
        self.history_steps.append(int(self.simulator.step_count))

    def run(self):
        """Integrate the full run, recording a bounded, strided history."""
        self._snapshot()  # step 0, before any stepping.
        if self.config.adaptive:
            target = self.config.total_time
            frame_interval = target / self.max_frames
            next_frame_time = frame_interval
            start_step = self.simulator.step_count
            while self.simulator.time < target:
                remaining = target - self.simulator.time
                if (
                    self.simulator.step_count > start_step
                    and remaining <= 1e-12 * max(1.0, abs(target))
                ):
                    self.simulator.time = target
                    break
                self.simulator.step(remaining=remaining)
                is_last = self.simulator.time >= target
                if is_last or self.simulator.time >= next_frame_time:
                    self._snapshot()
                    while next_frame_time <= self.simulator.time:
                        next_frame_time += frame_interval
            if self.history_times[-1] != target:
                self._snapshot()
            self.steps = self.simulator.step_count
            self.dt = self.simulator.dt
            self.stride = None
        else:
            for step in range(1, self.steps + 1):
                self.simulator.step()
                is_last = step == self.steps
                if is_last or step % self.stride == 0:
                    self._snapshot()
        self.diagnostics = self.simulator.diagnostics
        return self.history

    def value_range(self, field_name):
        """Streaming ``(min, max)`` over every recorded snapshot of a field."""
        return self._running_min[field_name], self._running_max[field_name]

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
            save_path=None,
            fps=None,
    ):
        """Build (and optionally save) a matplotlib animation of the history.

        Importing ``matplotlib.pyplot``/``matplotlib.animation`` is deferred to
        this call so that callers who want a headless (Agg) backend can set
        ``matplotlib.use("Agg")`` *before* the first ``import pyplot`` anywhere
        in the process.

        ``interval`` is the interactive playback speed in milliseconds/frame
        (``FuncAnimation``'s own knob) and is used as-is for on-screen
        display; it is unrelated to real wall-clock recording time, since
        frames are subsampled by ``max_frames``/``stride``, not by time.
        For *saved* animations (``save_path`` set), ``interval`` alone is a
        poor fps source -- the default ``interval=1`` would translate to a
        literal 1000 fps GIF/MP4, which most players either choke on or
        silently clamp. ``fps`` controls the saved file's frame rate
        directly: pass an explicit value, or leave it ``None`` to derive a
        sane one from ``interval`` capped at 30 (see ``_save_animation``).
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if not self.history:
            raise RuntimeError("animate() called before run(); history is empty.")

        if isinstance(field_names, str):
            field_names = [field_names]
        num_fields = len(field_names)

        if cmap_list is None:
            cmap_list = []
        # Cycle the default palette with modulo so any field count works.
        resolved_cmaps = list(cmap_list) + [
            _DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)]
            for i in range(max(0, num_fields - len(cmap_list)))
        ]

        # Determine color limits if absolute coloring is enabled, using the
        # running min/max accumulated during run() -- no history stacking.
        vmin_vmax = {}
        if absolute:
            for name in field_names:
                vmin_vmax[name] = self.value_range(name)

        # Create subplots.
        if split:
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
            axes = np.atleast_1d(np.array(axes)).reshape(-1)
            axes = axes[:num_fields]
        else:
            # Automatic grid of *separate* axes -- one per field -- instead of
            # stacking every field on the same axes at partial alpha.
            split_cols = int(np.ceil(np.sqrt(num_fields))) or 1
            split_rows = int(np.ceil(num_fields / split_cols))
            fig, axes = plt.subplots(
                split_rows,
                split_cols,
                figsize=(figsize_per_plot[0] * split_cols, figsize_per_plot[1] * split_rows)
            )
            axes = np.atleast_1d(np.array(axes)).reshape(-1)
            for extra_ax in axes[num_fields:]:
                extra_ax.set_visible(False)
            axes = axes[:num_fields]

        ims = []
        for i, name in enumerate(field_names):
            ax = axes[i]
            vmin, vmax = (None, None)
            if absolute:
                vmin, vmax = vmin_vmax[name]

            im = ax.imshow(
                self.history[0][name],
                cmap=resolved_cmaps[i],
                origin='lower',
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(name, fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 2)
            if show_colorbars:
                fig.colorbar(im, ax=ax, shrink=0.7)
            ims.append(im)

        n_frames = len(self.history)
        suptitle = fig.suptitle(f"{self.config.name} – 0.0% complete", fontsize=fontsize + 1)

        def update(frame):
            for i, name in enumerate(field_names):
                ims[i].set_array(self.history[frame][name])
            denom = max(1, n_frames - 1)
            progress = f"{100 * frame / denom:.1f}%"
            suptitle.set_text(f"{self.config.name} – {progress} complete")
            return ims + [suptitle]

        # blit=False: fig.suptitle is mutated every frame and is not part of
        # the blitted-artist bookkeeping, so blit=True silently failed to
        # repaint it.  Keep a strong reference on self so the animation is not
        # garbage collected before it plays (a classic FuncAnimation pitfall).
        self._ani = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=interval, blit=False
        )
        plt.tight_layout()

        if save_path is not None:
            self._save_animation(save_path, animation, plt, fig, interval, fps)
            plt.close(fig)
            return self._ani

        plt.show()
        return self._ani

    def _save_animation(self, save_path, animation, plt, fig, interval, fps=None):
        save_path = str(save_path)
        if fps is None:
            # Derive a sane saved-file frame rate from the interactive
            # ``interval`` (ms/frame), but cap it: with the default
            # ``interval=1`` a literal ``1000/interval`` fps is quantized
            # nonsense for a saved GIF/MP4 (most players don't support it
            # and it bloats file size for no visual benefit). 30 fps is a
            # conventional ceiling; callers who want something else should
            # pass ``fps=`` explicitly.
            fps = min(30, max(1, round(1000.0 / max(interval, 1))))
        is_mp4 = save_path.lower().endswith(".mp4")

        if is_mp4:
            if animation.writers.is_available("ffmpeg"):
                writer = animation.FFMpegWriter(fps=fps)
                self._ani.save(save_path, writer=writer)
                return
            fallback_path = save_path.rsplit(".", 1)[0] + ".gif"
            print(
                f"ffmpeg not available; cannot write {save_path!r}. "
                f"Falling back to GIF at {fallback_path!r}."
            )
            save_path = fallback_path

        writer = animation.PillowWriter(fps=fps)
        self._ani.save(save_path, writer=writer)
