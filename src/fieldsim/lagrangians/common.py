import jax.numpy as jnp

from fieldsim.lagrangian_term import LagrangianTerm


class Diffusion(LagrangianTerm):
    r"""Dirichlet energy giving Fickian diffusion ``dphi/dt = alpha * lap(phi)``.

    Discretisation
    --------------
    The energy is a sum over *interior faces only*::

        E = (alpha/2) * [ sum (diff(phi, axis=1))**2 + sum (diff(phi, axis=0))**2 ]

    Why there is no ``dx`` in that expression: the continuum energy is
    ``F = (alpha/2) * int |grad phi|^2 dA``.  A face difference approximates the
    gradient as ``(phi_{i+1} - phi_i)/dx`` and the cell area measure is ``dx^2``,
    so ``(1/dx)^2 * dx^2 = 1`` and the two cancel exactly.  The cancellation is
    written down here rather than being an accident of the code.

    Resulting stencil.  In 1D, ``E = (alpha/2) sum_i (phi_{i+1} - phi_i)^2`` and

        dE/dphi_j = alpha*(phi_j - phi_{j+1}) + alpha*(phi_j - phi_{j-1})
                  = -alpha * (phi_{j+1} - 2 phi_j + phi_{j-1})
                  = -alpha * dx^2 * lap_5(phi)_j

    so, after ``Lagrangian.functional_derivative`` divides by ``dx^2``,

        delta F / delta phi = -alpha * lap_5(phi)

    with ``lap_5`` the **compact 5-point Laplacian**
    ``(phi_E + phi_W + phi_N + phi_S - 4 phi_C)/dx^2``.  This is the fix for the
    old ``jnp.gradient`` (central-difference) energy, which produced the wide
    ``[1, 0, -2, 0, 1]/(4 dx^2)`` stencil: that operator has a *zero* eigenvalue
    on the checkerboard mode (odd/even decoupling), so grid-scale noise was
    completely undamped.  The compact stencil has checkerboard eigenvalue
    bounded by ``-8 alpha / dx^2`` (approached, not attained, on a finite
    grid: on the cell-centred Neumann grid the highest mode is ``m = n-1``
    with eigenvalue ``-alpha * (8/dx^2) * sin^2((n-1) pi / 2n)``, e.g.
    ``-8172`` vs. the ``-8192`` bound for ``alpha=1, n=32, dx=1/32``), the
    most strongly damped mode in the spectrum.

    Boundary conditions. For Neumann, omitting boundary faces from the sum is
    the variational statement of the homogeneous zero-flux condition: a
    boundary cell simply has fewer faces, giving the one-sided update
    ``phi_0 <- (1-c) phi_0 + c phi_1``.  No ghost cells are needed anywhere, and
    the resulting operator is exactly mass-conserving. Periodic mode includes
    right/top wrap faces through ``roll``; those contributions telescope too.

    Positivity.  With ``c = dt*alpha/dx^2 <= 1/4`` every update is a convex
    combination of non-negative values, so non-negative data stays non-negative.
    """

    def __init__(self, target, alpha, bc_type="neumann"):
        self.alpha = alpha
        self.bc_type = bc_type
        if bc_type not in ("neumann", "periodic"):
            raise ValueError(f"Unsupported boundary condition {bc_type!r}.")

        def energy_fn(values, dx):
            phi = values[target]
            if bc_type == "periodic":
                dphi_dx = jnp.roll(phi, -1, axis=1) - phi
                dphi_dy = jnp.roll(phi, -1, axis=0) - phi
            else:
                dphi_dx = jnp.diff(phi, axis=1)   # (ny, nx-1) x-interior faces
                dphi_dy = jnp.diff(phi, axis=0)   # (ny-1, nx) y-interior faces
            return 0.5 * alpha * (jnp.sum(dphi_dx ** 2) + jnp.sum(dphi_dy ** 2))

        def max_rate_fn(values, dx):
            # Largest eigenvalue of alpha*lap_5 is 8*alpha/dx^2, but the
            # classic (and sufficient) explicit-Euler bound for 2D diffusion is
            # dt <= dx^2/(4 alpha); use rate = 4 alpha / dx^2.
            return 4.0 * abs(alpha) / dx ** 2

        super().__init__(
            name=f"{target} diffusion",
            target=target,
            energy_fn=energy_fn,
            max_rate_fn=max_rate_fn,
            bc_type=bc_type,
        )
