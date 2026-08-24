from fieldsim.lagrangian_term import LagrangianTerm


class Diffusion(LagrangianTerm):
    def __init__(self, target, alpha):

        def expression_fn(fields):
            df_dx, df_dy = fields[target].gradient()
            return 0.5 * (df_dx ** 2 + df_dy ** 2)

        super().__init__(name=f'{target} diffusion', expression_fn=expression_fn, coefficient=alpha)
