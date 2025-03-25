from lagrangian_term import LagrangianTerm


class Diffusion(LagrangianTerm):
    def __init__(self, target, alpha):

        def expression_fn(fields):
            df_dx, df_dy = fields[target].gradient()
            return 0.5 * (df_dx ** 2 + df_dy ** 2)

        super().__init__(name=f'{target} diffusion', expression_fn=expression_fn, coefficient=alpha)

class GradientAlignmentTerm(LagrangianTerm):
    def __init__(self, field_a, field_b, kappa=1.0):
        """
        Adds a term: -kappa * ∇A · ∇B
        Encourages alignment of gradients between two fields.

        Args:
            field_a (str): First field name (e.g. "food")
            field_b (str): Second field name (e.g. "infra")
            kappa (float): Strength of interaction
        """

        def expression_fn(fields):
            dA_dx, dA_dy = fields[field_a].gradient()
            dB_dx, dB_dy = fields[field_b].gradient()
            return -(dA_dx * dB_dx + dA_dy * dB_dy)

        super().__init__(
            name=f"GradientAlignment({field_a}, {field_b})",
            expression_fn=expression_fn,
            coefficient=kappa
        )