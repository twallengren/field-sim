"""Base class for the additive terms of the free-energy functional F[phi].

Contract (Phase 2a)
-------------------
``energy(values, dx) -> scalar``

``values`` maps field names to raw ``jnp.ndarray`` value arrays -- never
``Field`` objects.  Operating on plain arrays keeps the whole right-hand side
``jax.grad`` / ``jax.jit`` friendly and means autodiff never has to rebuild a
``Field`` (the previous implementation did, silently dropping field metadata).

The physical coefficient of a term is applied *inside* ``energy_fn``; there is
no separate ``coefficient`` multiplier on the base class, so a coefficient
enters in exactly one place and can never be applied twice.

Each term also declares:

* ``target``   -- the name of the field whose evolution (and hence whose
                  explicit-Euler stability limit) this term governs.
* ``max_rate`` -- an upper bound on the per-unit-time rate this term imposes on
                  ``target``, used by :mod:`fieldsim.stability` to pick dt.
"""


class LagrangianTerm:
    def __init__(self, name, target, energy_fn, max_rate_fn=None, bc_type=None):
        """
        Args:
            name: descriptive name, for debugging/reporting.
            target: name of the field this term drives.
            energy_fn: Callable[[dict[str, jnp.ndarray], float], scalar].
                Must already include the term's coefficient.
            max_rate_fn: optional Callable[[dict, float], float] returning an
                upper bound on the rate (1/time) this term imposes.
        """
        self.name = name
        self.target = target
        self.energy_fn = energy_fn
        self.max_rate_fn = max_rate_fn
        self.bc_type = bc_type

    def energy(self, values: dict, dx: float):
        """Total (already coefficient-weighted) energy contribution, a scalar."""
        return self.energy_fn(values, dx)

    def max_rate(self, values: dict, dx: float) -> float:
        """Upper bound on the explicit-Euler rate (1/time) imposed on ``target``."""
        if self.max_rate_fn is None:
            raise NotImplementedError(
                f"LagrangianTerm {self.name!r} does not declare a max_rate; "
                "provide max_rate_fn so a stable timestep can be derived."
            )
        return self.max_rate_fn(values, dx)
