class GradientEnergyTerm:
    def evaluate(self, fields):
        df_dx, df_dy = fields["rho"].gradient()
        return 0.5 * (df_dx**2 + df_dy**2)
