import jax.numpy as jnp

from fieldsim.stability import rate_sum

#: Absolute mass floor added to the per-step truncation tolerance so that a
#: field whose total mass is (legitimately) zero does not have a zero tolerance.
_ABS_MASS_FLOOR = 1e-30


class Simulator:
    """Forward-Euler integrator for ``dphi/dt = -deltaF/delta phi - div(J) + S``.

    Boundary conditions are built into the operators (zero-flux), so the step
    loop no longer mutates boundary rows/columns.  The old ghost-copy Neumann
    update (``f[0] = f[1]`` ...) destroyed mass conservation and double-wrote
    the corners; it is gone.

    Positivity is a property of the discretisation (see
    :mod:`fieldsim.stability`), but round-off can still produce values of order
    ``-eps``.  Rather than silently clamping, each step measures the mass it
    would remove, ``neg = -sum(min(phi, 0)) * dx**2``, applies the floor, and
    records the result in :attr:`diagnostics`.  If a single step truncates more
    than ``truncation_tolerance`` times that field's mass the run stops with a
    ``RuntimeError``: round-off is recorded, a real bug is caught, and nothing
    is hidden.
    """

    def __init__(self, fields: dict, lagrangian, sources: list, flux_terms: list,
                 dt: float, check_every: int = 25, truncation_tolerance: float = 1e-8):
        """
        Args:
            fields: dict of {field_name: Field}. All fields must share one shape
                and one grid spacing.
            lagrangian: Lagrangian holding the free-energy terms.
            sources: list of SourceTerm objects.
            flux_terms: list of FluxTerm objects.
            dt: timestep size.
            check_every: run the finiteness/stability guard every N steps
                (``None`` or 0 disables it).
            truncation_tolerance: maximum fraction of a field's mass that the
                positivity floor may remove in a single step.
        """
        if not fields:
            raise ValueError("Simulator requires at least one field.")
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt!r}.")

        shapes = {name: tuple(f.shape) for name, f in fields.items()}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ValueError(
                f"All fields must share one grid shape; got {shapes!r}."
            )
        spacings = {name: float(f.dx) for name, f in fields.items()}
        if len(set(spacings.values())) != 1:
            raise ValueError(
                f"All fields must share one grid spacing dx; got {spacings!r}."
            )

        self.fields = fields
        self.shape = unique_shapes.pop()
        self.dx = next(iter(spacings.values()))
        self.lagrangian = lagrangian
        self.sources = sources
        self.flux_terms = flux_terms
        self.dt = float(dt)
        self.check_every = check_every
        self.truncation_tolerance = float(truncation_tolerance)
        self.time = 0.0
        self.step_count = 0
        self.diagnostics = {
            name: {"cumulative_truncated_mass": 0.0, "last_truncated_mass": 0.0}
            for name, field in fields.items()
            if field.is_dynamic
        }

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self):
        """Advance the system by one forward-Euler step."""
        dx = self.dx
        values = {name: field.get_values() for name, field in self.fields.items()}
        new_values = {}

        for name, field in self.fields.items():
            if not field.is_dynamic:
                continue

            # Gradient flow from the free energy: -delta F / delta phi.
            dF_dphi = self.lagrangian.functional_derivative(values, dx, name)

            # Conservative transport: -div(J).
            flux_sum = sum(
                flux.divergence(values, dx)
                for flux in self.flux_terms
                if flux.target == name
            )

            # Local sources/sinks.
            source_sum = sum(
                s.evaluate(values) for s in self.sources if s.target == name
            )

            updated = values[name] + self.dt * (-dF_dphi - flux_sum + source_sum)
            new_values[name] = self._apply_monitored_floor(name, updated, dx)

        for name, updated in new_values.items():
            self.fields[name].set_values(updated)

        self.time += self.dt
        self.step_count += 1

        if self.check_every and self.step_count % self.check_every == 0:
            self.check_state()

        return self.step_count

    def _apply_monitored_floor(self, name, updated, dx):
        """Clamp negatives to zero, measuring and policing the mass removed."""
        # ``+ 0.0`` normalises the -0.0 that arises when nothing was truncated.
        truncated = -float(jnp.sum(jnp.minimum(updated, 0.0))) * dx ** 2 + 0.0
        floored = jnp.maximum(updated, 0.0)

        diag = self.diagnostics[name]
        diag["last_truncated_mass"] = truncated
        diag["cumulative_truncated_mass"] += truncated

        if truncated > 0.0:
            mass = float(jnp.sum(floored)) * dx ** 2
            budget = self.truncation_tolerance * mass + _ABS_MASS_FLOOR
            if truncated > budget:
                raise RuntimeError(
                    f"Positivity floor truncated {truncated:.6g} of mass from field "
                    f"{name!r} at step {self.step_count + 1} (t={self.time:.6g}), "
                    f"exceeding the allowed {budget:.6g} "
                    f"({self.truncation_tolerance:g} x mass {mass:.6g}). "
                    "This indicates a genuine scheme failure, not round-off."
                )
        return floored

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def check_state(self):
        """Assert finiteness and that ``dt`` still satisfies the stability bound.

        Advective and consumption rates are state dependent, so a timestep that
        was safe for the initial condition can become unsafe as the solution
        evolves.  This re-derives the bound from the *current* fields.
        """
        values = {name: field.get_values() for name, field in self.fields.items()}

        for name, array in values.items():
            if not bool(jnp.all(jnp.isfinite(array))):
                raise RuntimeError(
                    f"Field {name!r} contains non-finite values at step "
                    f"{self.step_count} (t={self.time:.6g})."
                )

        total = rate_sum(
            values, self.lagrangian.terms, self.flux_terms, self.sources, self.dx
        )
        if self.dt * total > 1.0:
            raise RuntimeError(
                f"Timestep dt={self.dt:.6g} violates the stability bound at step "
                f"{self.step_count} (t={self.time:.6g}): dt * rate_sum = "
                f"{self.dt * total:.6g} > 1 (rate_sum={total:.6g}). "
                "Reduce dt or the transport/reaction coefficients."
            )
        return total

    def get_state(self):
        return {name: field.get_values() for name, field in self.fields.items()}
