import jax.numpy as jnp


SUPPORTED_BC_TYPES = ("neumann", "periodic")


class FluxTerm:
    r"""A conservative transport term entering the PDE as ``-div(J)``.

    Contract (Phase 2a)
    -------------------
    ``flux_fn(values, dx) -> (Jx, Jy)`` where ``values`` maps field names to raw
    ``jnp.ndarray`` value arrays and

        Jx has shape (ny, nx-1)  -- the x-normal *interior* faces
        Jy has shape (ny-1, nx)  -- the y-normal *interior* faces

    The legacy/default Neumann flux is defined on interior faces, with boundary
    faces implicitly zero. Periodic terms additionally provide a wrap-aware
    callback returning one right/top face per cell; the original
    ``flux_fn(values, dx)`` interface remains unchanged.

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

    def __init__(self, name, target, flux_fn, max_rate_fn=None,
                 bc_type="neumann", periodic_flux_fn=None):
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
        if bc_type not in SUPPORTED_BC_TYPES:
            raise ValueError(f"Unsupported boundary condition {bc_type!r}.")
        if bc_type == "periodic" and periodic_flux_fn is None:
            raise ValueError(
                "Periodic FluxTerm requires periodic_flux_fn returning one "
                "wrap-aware face flux per cell in each direction."
            )
        self.bc_type = bc_type
        self.periodic_flux_fn = periodic_flux_fn

    def divergence(self, values: dict, dx: float) -> jnp.ndarray:
        r"""Cell-centred ``div(J)``, shape (ny, nx).

        Neumann interior-face fluxes are zero-padded and differenced. Periodic
        face fluxes use a rolled backward difference so wrap faces telescope::

            Jx_p = pad(Jx, ((0,0),(1,1)))
            Jy_p = pad(Jy, ((1,1),(0,0)))
            div  = (diff(Jx_p, axis=1) + diff(Jy_p, axis=0)) / dx

        Every interior face appears exactly twice in the result, with opposite
        signs, so ``sum(div) == 0`` by telescoping: any term built this way
        conserves total mass to machine precision, whatever ``flux_fn`` returns.
        """
        if self.bc_type == "periodic":
            Jx, Jy = self.periodic_flux_fn(values, dx)
            shape = next(iter(values.values())).shape
            if Jx.shape != shape or Jy.shape != shape:
                raise ValueError(
                    f"Periodic fluxes must both have cell-grid shape {shape}; "
                    f"got {Jx.shape} and {Jy.shape}."
                )
            return (
                Jx - jnp.roll(Jx, 1, axis=1)
                + Jy - jnp.roll(Jy, 1, axis=0)
            ) / dx

        Jx, Jy = self.flux_fn(values, dx)
        ny, nx = next(iter(values.values())).shape
        if Jx.shape != (ny, nx - 1) or Jy.shape != (ny - 1, nx):
            raise ValueError(
                "Neumann flux_fn must return interior-face arrays with shapes "
                f"{(ny, nx - 1)} and {(ny - 1, nx)}; got {Jx.shape} and {Jy.shape}."
            )
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
