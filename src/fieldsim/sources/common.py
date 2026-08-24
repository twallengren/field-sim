from fieldsim.source_term import SourceTerm


class LogisticGrowthSource(SourceTerm):
    def __init__(self, target, upper_limit, gamma=1.0):

        def expression_fn(fields):
            T = fields[target].get_values()
            U = fields[upper_limit].get_values()
            return T * (1 - T / U)

        super().__init__(name=f"{target} growth", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)
