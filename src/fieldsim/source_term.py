import jax.numpy as jnp


class SourceTerm:
    """A local (pointwise) source/sink added to the right-hand side.

    Contract (Phase 2a)
    -------------------
    ``evaluate(values) -> Array`` of shape (ny, nx), where ``values`` maps field
    names to raw ``jnp.ndarray`` value arrays (never ``Field`` objects).  Source
    terms are pointwise, so they do not need ``dx``.

    ``coefficient`` is a plain multiplier applied to ``expression_fn``; unlike a
    flux coefficient it has no effect on any upwind/branch decision, so keeping
    it on the base class is unambiguous.
    """

    def __init__(self, name, target_field_name, expression_fn, coefficient=1.0,
                 max_rate_fn=None):
        """
        Args:
            name: descriptive name, for debugging.
            target_field_name: name of the field this term updates.
            expression_fn: Callable[[dict[str, jnp.ndarray]], jnp.ndarray].
            coefficient: scalar multiplier.
            max_rate_fn: optional Callable[[dict], float] bounding the per-unit
                -time rate this term imposes, for timestep selection.
        """
        self.name = name
        self.target = target_field_name
        self.expression_fn = expression_fn
        self.coefficient = coefficient
        self.max_rate_fn = max_rate_fn

    def evaluate(self, values: dict) -> jnp.ndarray:
        """Evaluate the source/sink on the grid for the current field values."""
        return self.coefficient * self.expression_fn(values)

    def max_rate(self, values: dict, dx: float = None) -> float:
        """Upper bound on the reaction rate (1/time) imposed on ``target``.

        ``dx`` is accepted (and ignored) so that all term types share one
        signature for :mod:`fieldsim.stability`.
        """
        if self.max_rate_fn is None:
            raise NotImplementedError(
                f"SourceTerm {self.name!r} does not declare a max_rate; provide "
                "max_rate_fn so a stable timestep can be derived."
            )
        return self.max_rate_fn(values)

    def __repr__(self):
        return f"SourceTerm(name={self.name}, target={self.target}, coeff={self.coefficient})"
