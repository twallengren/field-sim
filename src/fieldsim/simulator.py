class Simulator:
    def __init__(self, fields: dict, lagrangian, sources: list, flux_terms: list, dt: float):
        """
        Initialize the simulation engine.

        Args:
            fields: dict of {field_name: Field} instances.
            lagrangian: Lagrangian object containing all Lagrangian terms.
            sources: list of SourceTerm objects.
            flux_terms: list of FluxTerm objects.
            dt: timestep size.
        """
        self.fields = fields
        self.lagrangian = lagrangian
        self.sources = sources
        self.flux_terms = flux_terms
        self.dt = dt
        self.time = 0.0

    def step(self):
        """
        Advance the system by one time step using:
        - gradient flow from Lagrangian
        - source/sink terms
        - flux divergence from continuity equation
        """
        new_values = {}

        for name, field in self.fields.items():
            if not field.is_dynamic:
                continue

            # Lagrangian term: gradient flow
            dF_df = self.lagrangian.functional_derivative(self.fields, name)

            # Source/sink terms
            source_sum = sum(
                s.evaluate(self.fields) for s in self.sources if s.target == name
            )

            # Flux divergence terms (∇ · J)
            flux_sum = sum(
                flux.divergence(self.fields, field.dx)
                for flux in self.flux_terms
                if flux.target_field_name == name
            )

            # Time step update
            updated = (
                field.get_values()
                - self.dt * dF_df
                - self.dt * flux_sum
                + self.dt * source_sum
            )

            new_values[name] = updated

        # Apply updates and boundary conditions
        for name, field in self.fields.items():
            if name in new_values:
                field.set_values(new_values[name])
                field.apply_bc()

        self.time += self.dt

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.step()

    def get_state(self):
        return {name: field.get_values() for name, field in self.fields.items()}

