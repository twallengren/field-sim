class Simulator:
    def __init__(self, fields: dict, lagrangian, sources: list, dt: float):
        """
        Initialize the simulation engine.

        Args:
            fields: dict of {field_name: Field} instances.
            lagrangian: Lagrangian object containing all Lagrangian terms.
            sources: list of SourceTerm objects.
            dt: timestep size.
        """
        self.fields = fields
        self.lagrangian = lagrangian
        self.sources = sources
        self.dt = dt
        self.time = 0.0

    def step(self):
        """
        Advance the system by one time step using gradient descent and source terms.
        """
        new_values = {}

        for name, field in self.fields.items():
            if not field.is_dynamic:
                continue

            # Functional derivative (gradient flow)
            dF_df = self.lagrangian.functional_derivative(self.fields, name)

            # Sum of source/sink terms targeting this field
            source_sum = sum(
                s.evaluate(self.fields) for s in self.sources if s.target == name
            )

            # Time step update
            updated = field.get_values() - self.dt * dF_df + self.dt * source_sum
            new_values[name] = updated

        # Apply updates and boundary conditions
        for name, field in self.fields.items():
            if name in new_values:
                field.set_values(new_values[name])
                field.apply_bc()

        self.time += self.dt

    def run(self, n_steps: int):
        """
        Run the simulation forward for n_steps.
        """
        for _ in range(n_steps):
            self.step()

    def get_state(self):
        """
        Return current state of all fields as a dict of arrays.
        Useful for visualization or exporting data.
        """
        return {name: field.get_values() for name, field in self.fields.items()}
