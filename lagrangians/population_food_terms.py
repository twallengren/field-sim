class PopulationDiffusion:
    def __init__(self, alpha):
        self.alpha = alpha

    def evaluate(self, fields):
        df_dx, df_dy = fields["pop"].gradient()
        return 0.5 * self.alpha * (df_dx**2 + df_dy**2)

class FoodDiffusion:
    def __init__(self, alpha):
        self.alpha = alpha

    def evaluate(self, fields):
        df_dx, df_dy = fields["food"].gradient()
        return 0.5 * self.alpha * (df_dx**2 + df_dy**2)

class FoodTransportToInfraTerm:
    def __init__(self, kappa=1.0):
        """
        Encourages food to flow toward infrastructure-rich regions:
        L = -kappa * ∇F ⋅ ∇I
        """
        self.kappa = kappa

    def evaluate(self, fields):
        dF_dx, dF_dy = fields["food"].gradient()
        dI_dx, dI_dy = fields["infra"].gradient()
        return -self.kappa * (dF_dx * dI_dx + dF_dy * dI_dy)
