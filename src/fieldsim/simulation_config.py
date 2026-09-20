import math


class SimulationConfig:
    """Declarative description of a simulation.

    The timestep is *not* part of the configuration: it is derived from the
    physics by :mod:`fieldsim.stability` once the initial conditions exist (see
    :class:`~fieldsim.simulation_runner.SimulationRunner`).  A config therefore
    specifies how long to integrate for, not how many steps to take.
    """

    def __init__(self, name, field_defs, lagrangian_terms, flux_terms, sources,
                 total_time, safety=0.8, adaptive=False, max_dt=0.1,
                 derived_fields=None):
        """
        Args:
            name: str — name of the simulation.
            field_defs: dict of field_name → kwargs for
                :class:`~fieldsim.field.Field` (shape, dx, init_fn, is_dynamic,
                bc_type, units).
            lagrangian_terms: list of LagrangianTerm instances.
            flux_terms: list of FluxTerm instances.
            sources: list of SourceTerm instances.
            total_time: float — physical duration to integrate (simulation time
                units; years for the bundled configs).
            safety: float in (0, 1) — CFL safety factor used when deriving dt.
        """
        if not math.isfinite(total_time) or not total_time > 0:
            raise ValueError(f"total_time must be positive, got {total_time!r}.")
        if not math.isfinite(safety) or not 0.0 < safety < 1.0:
            raise ValueError(f"safety must lie in (0, 1), got {safety!r}.")
        if not math.isfinite(max_dt) or not max_dt > 0.0:
            raise ValueError(f"max_dt must be positive, got {max_dt!r}.")

        self.name = name
        self.field_defs = field_defs
        self.lagrangian_terms = lagrangian_terms
        self.flux_terms = flux_terms
        self.sources = sources
        self.total_time = float(total_time)
        self.safety = float(safety)
        self.adaptive = bool(adaptive)
        self.max_dt = float(max_dt)
        self.derived_fields = dict(derived_fields or {})
