class SimulationConfig:
    def __init__(self, name, field_defs, lagrangian_terms, sources, dt, steps):
        """
        Args:
            name: str — name of the simulation
            field_defs: dict of field_name → dict {shape, dx, init_fn, is_dynamic}
            lagrangian_terms: list of LagrangianTerm instances
            sources: list of SourceTerm instances
            dt: float — timestep size
            steps: int — number of simulation steps
        """
        self.name = name
        self.field_defs = field_defs
        self.lagrangian_terms = lagrangian_terms
        self.sources = sources
        self.dt = dt
        self.steps = steps
