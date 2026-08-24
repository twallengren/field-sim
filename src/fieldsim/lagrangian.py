import jax
import jax.numpy as jnp


class Lagrangian:
    """Collection of :class:`~fieldsim.lagrangian_term.LagrangianTerm` objects.

    The total free energy is ``F = sum(term.energy(values, dx))`` and the
    dynamics are gradient flow ``dphi/dt = -delta F / delta phi``.
    """

    def __init__(self):
        self.terms = []

    def add_term(self, term):
        """Add a LagrangianTerm object to this Lagrangian."""
        self.terms.append(term)

    def total_energy(self, values: dict, dx: float) -> jnp.ndarray:
        """Total scalar free energy F[values].

        Args:
            values: {field_name: jnp.ndarray of field values}
            dx: uniform grid spacing.
        """
        if not self.terms:
            return jnp.zeros(())
        total = self.terms[0].energy(values, dx)
        for term in self.terms[1:]:
            total = total + term.energy(values, dx)
        return total

    def functional_derivative(self, values: dict, dx: float, wrt_name: str) -> jnp.ndarray:
        """Variational derivative ``delta F / delta phi`` of the energy density.

        ``jax.grad`` of the *total* energy gives ``dF/dphi_ij``, the derivative
        with respect to the value in a cell.  The variational derivative of the
        corresponding density (the object that appears in the PDE) is

            delta F / delta phi (x_ij)  =  (dF/dphi_ij) / dx^2

        because ``F = sum_ij f(phi_ij) * dx^2``.  Making the ``/dx**2`` explicit
        here removes the hidden dx^2 cancellation the old code relied on.
        """
        current = values[wrt_name]

        if not self.terms:
            return jnp.zeros(current.shape, dtype=current.dtype)

        def energy_of(var_values):
            return self.total_energy({**values, wrt_name: var_values}, dx)

        return jax.grad(energy_of)(current) / dx ** 2
