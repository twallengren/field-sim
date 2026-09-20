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
      proportional to its own value.  A cell drains through *every* face whose
      velocity points out of it, so its fractional loss in one step is
      ``dt * (u_R^+ + u_L^- + v_U^+ + v_D^-)/dx``, where ``x^+ = max(x, 0)``,
      ``x^- = max(-x, 0)`` and the four terms are its right/left/upper/lower
      faces (boundary faces are zero for Neumann and wrap for periodic). The
      condition that no cell can be
      drained below zero is therefore

          ``dt * max_cell[(u_R^+ + u_L^- + v_U^+ + v_D^-)/dx] <= 1``

      and :meth:`max_rate` returns exactly that per-cell two-sided outflow
      maximum.  The looser-looking ``(max|u| + max|v|)/dx`` is *not* an upper
      bound for it: where ``F`` has an interior local minimum both faces of an
      axis drain the same cell, and the one-sided form under-counts by up to a
      factor of 2, permitting a dt that drives that cell negative in a single
      step.  Conversely, on smooth fields where each cell has one outflowing
      face per axis the per-cell maximum is usually *smaller*
      than ``(max|u| + max|v|)/dx`` (the two maxima need not occur in the same
      cell), so the tight rate also buys a larger timestep.

    Mass conservation is inherited from :meth:`FluxTerm.divergence`: both the
    zero-boundary and wrap-around face differences telescope.
    """

    def __init__(self, target_field, gradient_field, kappa=1.0,
                 bc_type="neumann"):
        self.target_field = target_field
        self.gradient_field = gradient_field
        self.kappa = kappa
        self.bc_type = bc_type
        if bc_type not in ("neumann", "periodic"):
            raise ValueError(f"Unsupported boundary condition {bc_type!r}.")

        def flux_fn(values, dx):
            P = values[target_field]
            u, v = _face_velocities(values[gradient_field], kappa, dx)
            Jx = jnp.maximum(u, 0.0) * P[:, :-1] + jnp.minimum(u, 0.0) * P[:, 1:]
            Jy = jnp.maximum(v, 0.0) * P[:-1, :] + jnp.minimum(v, 0.0) * P[1:, :]
            return Jx, Jy

        def periodic_flux_fn(values, dx):
            P = values[target_field]
            F = values[gradient_field]
            u = kappa * (jnp.roll(F, -1, axis=1) - F) / dx
            v = kappa * (jnp.roll(F, -1, axis=0) - F) / dx
            Jx = jnp.maximum(u, 0.0) * P + jnp.minimum(u, 0.0) * jnp.roll(P, -1, axis=1)
            Jy = jnp.maximum(v, 0.0) * P + jnp.minimum(v, 0.0) * jnp.roll(P, -1, axis=0)
            return Jx, Jy

        def max_rate_fn(values, dx):
            # Per-cell *two-sided* outflow coefficient.  A cell loses mass
            # through every face whose velocity points out of it, so its
            # fractional loss in one step is dt/dx times the sum of the
            # outflowing parts of all four of its faces -- not the one-sided
            # (max|u| + max|v|)/dx, which under-counts by up to 4x wherever the
            # attractant has an interior local minimum (or maximum, for
            # kappa < 0) and both faces of an axis drain the same cell.
            # Zero-padding to the domain boundary encodes the zero-flux BC, so
            # boundary faces contribute no outflow, exactly as in flux_fn.
            if bc_type == "periodic":
                F = values[gradient_field]
                u = kappa * (jnp.roll(F, -1, axis=1) - F) / dx
                v = kappa * (jnp.roll(F, -1, axis=0) - F) / dx
                outflow = (
                    jnp.maximum(u, 0.0)
                    + jnp.maximum(-jnp.roll(u, 1, axis=1), 0.0)
                    + jnp.maximum(v, 0.0)
                    + jnp.maximum(-jnp.roll(v, 1, axis=0), 0.0)
                ) / dx
            else:
                u, v = _face_velocities(values[gradient_field], kappa, dx)
                up = jnp.pad(u, ((0, 0), (1, 1)))
                vp = jnp.pad(v, ((1, 1), (0, 0)))
                outflow = (
                    jnp.maximum(up[:, 1:], 0.0)
                    + jnp.maximum(-up[:, :-1], 0.0)
                    + jnp.maximum(vp[1:, :], 0.0)
                    + jnp.maximum(-vp[:-1, :], 0.0)
                ) / dx
            if not outflow.size:
                return 0.0
            return float(jnp.max(outflow))

        super().__init__(
            name=f"Advection({target_field} <- grad {gradient_field})",
            target=target_field,
            flux_fn=flux_fn,
            max_rate_fn=max_rate_fn,
            bc_type=bc_type,
            periodic_flux_fn=periodic_flux_fn if bc_type == "periodic" else None,
        )

    def face_velocities(self, values: dict, dx: float):
        """Signed interior-face velocities ``(u, v)`` for the current state."""
        if self.bc_type == "periodic":
            F = values[self.gradient_field]
            return (
                self.kappa * (jnp.roll(F, -1, axis=1) - F) / dx,
                self.kappa * (jnp.roll(F, -1, axis=0) - F) / dx,
            )
        return _face_velocities(values[self.gradient_field], self.kappa, dx)
