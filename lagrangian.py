import jax
import jax.numpy as jnp


class Lagrangian:
    def __init__(self):
        self.terms = []

    def add_term(self, term):
        """
        Add a LagrangianTerm object to this Lagrangian.
        Each term contributes to the total energy.
        """
        self.terms.append(term)

    def total_energy(self, fields: dict) -> jnp.ndarray:
        """
        Compute the total scalar energy functional F = ∫ L dx dy.
        - fields: a dict of {field_name: Field instance}
        - Each LagrangianTerm returns a 2D energy density array
        - We sum over all space to compute the total scalar energy
        """
        energy_density = sum(term.evaluate(fields) for term in self.terms)
        return jnp.sum(energy_density)  # This gives scalar ∫ L dx dy

    def functional_derivative(self, fields: dict, wrt_field_name: str) -> jnp.ndarray:
        """
        Compute δF/δφ, the variational derivative of the energy functional
        with respect to the field φ (named wrt_field_name).

        JAX will treat the values array of the target field as a differentiable input,
        and return the gradient of the scalar total_energy with respect to it.

        - fields: dictionary of {name: Field instance}
        - wrt_field_name: the name of the field we're taking δF/δφ for
        """

        # Step 1: Define a function where the target field is the input
        def energy_wrt_var_field(var_values):
            # Copy fields and replace the target with var_values
            temp_fields = fields.copy()
            original_field = fields[wrt_field_name]

            # Make a new temporary field with the same metadata but replaced values
            new_field = original_field.__class__(
                name=original_field.name,
                shape=original_field.shape,
                dx=original_field.dx,
                units=original_field.units,
                is_dynamic=True
            )
            new_field.set_values(var_values)
            temp_fields[wrt_field_name] = new_field

            return self.total_energy(temp_fields)  # Scalar energy

        # Step 2: Use JAX to differentiate the energy w.r.t. the field array
        return jax.grad(energy_wrt_var_field)(fields[wrt_field_name].get_values())
