from flux_term import FluxTerm


class AdvectionAlongGradientFlux(FluxTerm):
    def __init__(self, target_field, gradient_field, kappa=1.0, scale_by_target=True):
        """
        Models advection of `target_field` along the gradient of `gradient_field`.

        J = ±kappa * [target] * ∇[gradient_field] if scale_by_target
            ±kappa * ∇[gradient_field] otherwise

        Args:
            target_field (str): The field being transported.
            gradient_field (str): The field whose gradient defines the direction.
            kappa (float): Strength and direction of transport.
            scale_by_target (bool): Whether to multiply flux by target field value.
        """

        def flux_fn(fields):
            grad_x, grad_y = fields[gradient_field].gradient()
            if scale_by_target:
                target_vals = fields[target_field].get_values()
                grad_x *= target_vals
                grad_y *= target_vals
            return grad_x, grad_y

        super().__init__(
            name=f"Advection({target_field} ← ∇{gradient_field})",
            target_field_name=target_field,
            flux_fn=flux_fn,
            coefficient=kappa
        )
