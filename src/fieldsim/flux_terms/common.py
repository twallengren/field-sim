import jax.numpy as jnp

from fieldsim.flux_term import FluxTerm


def _face_velocities(gradient_values, kappa, dx):
    """Signed face velocities ``kappa * grad(F)`` evaluated on interior faces."""
    u = kappa * jnp.diff(gradient_values, axis=1) / dx   # (ny, nx-1)
    v = kappa * jnp.diff(gradient_values, axis=0) / dx   # (ny-1, nx)
    return u, v


class AdvectionAlongGradientFlux(FluxTerm):
    r"""Density-dependent chemotaxis-style transport ``J = kappa * P * grad(F)``.

    ``P`` is ``target_field`` (the transported density) and ``F`` is
    ``gradient_field`` (the attractant).  The simulator applies ``-div(J)``, so
    with ``kappa > 0`` the density moves **up** the gradient of ``F``
    (attraction); ``kappa < 0`` gives repulsion.

    Discretisation (donor-cell / first-order upwind)::

        u  = kappa * diff(F, axis=1) / dx          # (ny, nx-1)
        v  = kappa * diff(F, axis=0) / dx          # (ny-1, nx)
        Jx = max(u, 0) * P[:, :-1] + min(u, 0) * P[:, 1:]
        Jy = max(v, 0) * P[:-1, :] + min(v, 0) * P[1:, :]

    i.e. the face flux always carries the density of the *upwind* (donor) cell.

    Two properties this buys, both of which the previous implementation lacked:

    * **Density dependence.**  The old flux was ``J = kappa * grad(F)``, which
      is independent of ``P`` and therefore transports mass out of cells that
      contain none, producing spurious negative densities that were then hidden
      by the softplus clamp.  Here ``J = 0`` wherever the donor cell is empty.
    * **Positivity.**  A cell can only lose what it holds: the outgoing flux is
      proportional to its own value, and under the advective CFL condition
      ``dt * (|u| + |v|)/dx <= 1`` the total fractional outflow in one step is
      at most 1, so a cell can never be drained below zero.

    Mass conservation is inherited from :meth:`FluxTerm.divergence` (telescoping
    face differences with zero boundary faces).
    """

    def __init__(self, target_field, gradient_field, kappa=1.0):
        self.target_field = target_field
        self.gradient_field = gradient_field
        self.kappa = kappa

        def flux_fn(values, dx):
            P = values[target_field]
            u, v = _face_velocities(values[gradient_field], kappa, dx)
            Jx = jnp.maximum(u, 0.0) * P[:, :-1] + jnp.minimum(u, 0.0) * P[:, 1:]
            Jy = jnp.maximum(v, 0.0) * P[:-1, :] + jnp.minimum(v, 0.0) * P[1:, :]
            return Jx, Jy

        def max_rate_fn(values, dx):
            u, v = _face_velocities(values[gradient_field], kappa, dx)
            umax = float(jnp.max(jnp.abs(u))) if u.size else 0.0
            vmax = float(jnp.max(jnp.abs(v))) if v.size else 0.0
            return (umax + vmax) / dx

        super().__init__(
            name=f"Advection({target_field} <- grad {gradient_field})",
            target=target_field,
            flux_fn=flux_fn,
            max_rate_fn=max_rate_fn,
        )

    def face_velocities(self, values: dict, dx: float):
        """Signed interior-face velocities ``(u, v)`` for the current state."""
        return _face_velocities(values[self.gradient_field], self.kappa, dx)
