import jax.numpy as jnp


class FluxTerm:
    r"""A conservative transport term entering the PDE as ``-div(J)``.

    Contract (Phase 2a)
    -------------------
    ``flux_fn(values, dx) -> (Jx, Jy)`` where ``values`` maps field names to raw
    ``jnp.ndarray`` value arrays and

        Jx has shape (ny, nx-1)  -- the x-normal *interior* faces
        Jy has shape (ny-1, nx)  -- the y-normal *interior* faces

    i.e. the flux is defined **on faces**, not at cell centres.  Only interior
    faces exist; the boundary faces are implicitly zero, which is exactly the
    zero-flux (homogeneous Neumann) boundary condition.

    ``flux_fn`` receives ``dx`` because face velocities are built from face
    differences (``diff(F)/dx``) and therefore need the grid spacing.

    Coefficient convention
    ----------------------
    ``FluxTerm`` deliberately has **no** ``coefficient`` multiplier: ``flux_fn``
    must return the full physical flux, coefficient included.  Rationale: an
    upwind (donor-cell) flux has to know the *signed* velocity in order to pick
    the correct donor cell, so the transport coefficient cannot be factored out
    of ``flux_fn`` without breaking the upwinding for negative coefficients.
    One multiplication site, no ambiguity.
    """

    def __init__(self, name, target, flux_fn, max_rate_fn=None):
        """
        Args:
            name: descriptive name of the flux.
            target: name of the field whose continuity equation this affects.
            flux_fn: Callable[[dict, float], tuple[Array, Array]] returning
                interior-face fluxes (Jx, Jy) of shape (ny, nx-1), (ny-1, nx).
            max_rate_fn: optional Callable[[dict, float], float] bounding the
                transport rate (1/time) for timestep selection.
        """
        self.name = name
        self.target = target
        self.flux_fn = flux_fn
        self.max_rate_fn = max_rate_fn

    def divergence(self, values: dict, dx: float) -> jnp.ndarray:
        r"""Cell-centred ``div(J)``, shape (ny, nx).

        The interior-face fluxes are zero-padded to the domain boundary (zero
        flux through the boundary) and differenced::

            Jx_p = pad(Jx, ((0,0),(1,1)))
            Jy_p = pad(Jy, ((1,1),(0,0)))
            div  = (diff(Jx_p, axis=1) + diff(Jy_p, axis=0)) / dx

        Every interior face appears exactly twice in the result, with opposite
        signs, so ``sum(div) == 0`` by telescoping: any term built this way
        conserves total mass to machine precision, whatever ``flux_fn`` returns.
        """
        Jx, Jy = self.flux_fn(values, dx)
        Jx_p = jnp.pad(Jx, ((0, 0), (1, 1)))
        Jy_p = jnp.pad(Jy, ((1, 1), (0, 0)))
        return (jnp.diff(Jx_p, axis=1) + jnp.diff(Jy_p, axis=0)) / dx

    def max_rate(self, values: dict, dx: float) -> float:
        """Upper bound on the advective rate (1/time) imposed on ``target``."""
        if self.max_rate_fn is None:
            raise NotImplementedError(
                f"FluxTerm {self.name!r} does not declare a max_rate; provide "
                "max_rate_fn so a stable timestep can be derived."
            )
        return self.max_rate_fn(values, dx)
