import jax
import jax.numpy as jnp


def canonicalize(values):
    """Return ``values`` as a device array with a concrete (non-weak) dtype.

    JAX marks arrays that were built from Python scalars -- e.g. the cell-centre
    grid ``(arange(n) + 0.5) * dx``, and hence any initial condition derived
    from it -- as *weakly typed*.  A weakly typed argument is a **different jit
    signature** from the strongly typed array the same expression produces once
    it has been through one arithmetic step, so leaving the weak flag in place
    costs the simulator's compiled step an extra retrace on its second call.
    Canonicalising every value a :class:`Field` stores keeps that cache to a
    single entry.  Nothing about the numbers changes: the dtype is unchanged,
    only the "this could be promoted" annotation is dropped.
    """
    array = jnp.asarray(values)
    if getattr(array, "weak_type", False):
        array = jax.lax.convert_element_type(array, array.dtype)
    return array


#: Boundary conditions implemented by the operator layer. ``Field`` stores and
#: validates the metadata; it does not modify boundary cells itself.
SUPPORTED_BC_TYPES = ("neumann", "periodic")


class Field:
    """A scalar field sampled at cell centres on a uniform square grid.

    The domain is ``[0, nx*dx] x [0, ny*dx]`` and sample points are the **cell
    centres** ``x_i = (i + 1/2) dx``.  (The previous ``x_i = i*dx`` put samples
    on the left/bottom edges and covered only ``[0, L - dx]``, an off-by-one
    domain; cell centring is also what makes the finite-volume flux and the
    face-difference energy consistent, and gives the DCT-II cosine modes as
    exact eigenvectors of the discrete Neumann Laplacian.) Both homogeneous
    Neumann and wrap-around periodic operators use this grid.

    A ``Field`` is now a thin container: values plus metadata.  It applies no
    clipping and no boundary condition of its own.  In particular the old
    ``_clip`` (``vmin + softplus(v - vmin)``) is gone: softplus is *strictly
    increasing*, so it fabricated ``>= ln 2`` of mass in every empty cell every
    time it ran (init, every ``set_values``, every ``apply_bc``), and that
    ratchet was masking both a division by zero in the logistic source and an
    over-large timestep.  Positivity is now a property of the discretisation,
    backstopped by a *monitored* floor in :class:`~fieldsim.simulator.Simulator`.
    """

    def __init__(self, name, shape, dx=1.0, units=None, is_dynamic=True,
                 init_fn=None, bc_type="neumann"):
        if bc_type not in SUPPORTED_BC_TYPES:
            raise ValueError(
                f"Field {name!r}: unsupported boundary condition type "
                f"{bc_type!r}; supported types are {list(SUPPORTED_BC_TYPES)}."
            )
        shape = tuple(shape)
        try:
            normalized_shape = tuple(int(s) for s in shape)
        except (TypeError, ValueError, OverflowError):
            normalized_shape = ()
        if (
            len(shape) != 2
            or len(normalized_shape) != 2
            or any(n != s or n < 2 for n, s in zip(normalized_shape, shape))
        ):
            raise ValueError(
                f"Field {name!r}: shape must be a 2-tuple of ints >= 2, got {shape!r}."
            )
        shape = normalized_shape
        if not dx > 0:
            raise ValueError(f"Field {name!r}: dx must be positive, got {dx!r}.")

        self.name = name
        self.shape = shape
        self.dx = float(dx)
        self.units = units
        self.is_dynamic = is_dynamic
        self.bc_type = bc_type
        self.values = self._initialize(init_fn)

    def _initialize(self, fn):
        ny, nx = self.shape
        if fn is None:
            return jnp.zeros((ny, nx))

        # Cell-centred coordinates: x_i = (i + 1/2) dx.
        x = (jnp.arange(nx) + 0.5) * self.dx
        y = (jnp.arange(ny) + 0.5) * self.dx
        X, Y = jnp.meshgrid(x, y, indexing="xy")
        return canonicalize(fn(X, Y))

    def get_values(self):
        return self.values

    def set_values(self, new_values):
        self.values = canonicalize(new_values)
