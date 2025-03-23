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
