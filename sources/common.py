import jax.numpy as jnp

from source_term import SourceTerm


class ExponentialGrowthSource(SourceTerm):
    def __init__(self, target, gamma=1.0):

        def expression_fn(fields):
            return fields[target].get_values()

        super().__init__(name=f"{target} exponential growth", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)

class ExponentialDecaySource(SourceTerm):
    def __init__(self, target, gamma=1.0, min_threshold=None):

        def expression_fn_no_threshold(fields):
            return -fields[target].get_values()

        def expression_fn_min_threshold(fields):
            T = fields[target].get_values()
            M = fields[min_threshold].get_values()
            sigmoid = 1 / (1 + jnp.exp(-10 * (T - M)))
            return -T * sigmoid

        if min_threshold is None:
            expression_fn = expression_fn_no_threshold
        else:
            expression_fn = expression_fn_min_threshold

        super().__init__(name=f"{target} decay", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)

class LogisticGrowthSource(SourceTerm):
    def __init__(self, target, upper_limit, gamma=1.0):

        def expression_fn(fields):
            T = fields[target].get_values()
            U = fields[upper_limit].get_values()
            return T * (1 - T / U)

        super().__init__(name=f"{target} growth", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)

class BoostedLogisticGrowthSource(SourceTerm):
    def __init__(self, target, upper_limit, boost_field, gamma_0=1.0, alpha=1.0):
        def expression_fn(fields):
            F = fields[target].get_values()
            K = fields[upper_limit].get_values()
            B = fields[boost_field].get_values()
            gamma = gamma_0 * (1 + alpha * B)
            return gamma * F * (1 - F / K)  # avoid division by zero

        name = f"Logistic({target}→{upper_limit}) × Industry"
        super().__init__(name=name, target_field_name=target, expression_fn=expression_fn)


class ResourceLimitedDecaySource(SourceTerm):
    def __init__(self, target, resource, gamma=1.0):

        def expression_fn(fields):
            T = fields[target].get_values()
            R = fields[resource].get_values()
            sigmoid = 1 / (1 + jnp.exp(-10 * (T - R)))
            return -T*(T - R) * sigmoid

        super().__init__(name=f"{resource} limited {target} decay", target_field_name=target, expression_fn=expression_fn, coefficient=gamma)

class ResourceConsumptionSource(SourceTerm):
    def __init__(self, target, consumer, rho=1.0):

        def expression_fn(fields):
            C = fields[consumer].get_values()
            return -C

        super().__init__(name=f"{target} consumption by {consumer}", target_field_name=target, expression_fn=expression_fn, coefficient=rho)

class ConstantSource(SourceTerm):
    def __init__(self, target, value=1.0, mask_fn=None):

        def expression_fn(fields):
            F = fields[self.target].get_values()
            shape = F.shape

            # Default: uniform source everywhere
            source = jnp.ones(shape) * value

            # If masked, apply spatial mask
            if mask_fn is not None:
                dx = fields[self.target].dx
                x = jnp.arange(shape[1]) * dx
                y = jnp.arange(shape[0]) * dx
                X, Y = jnp.meshgrid(x, y, indexing="xy")
                mask = mask_fn(X, Y)
                source *= mask

            return source

        super().__init__(name=f"constant {target} source", target_field_name=target, expression_fn=expression_fn)

class MultiplicativeGrowthSource(SourceTerm):
    def __init__(self, target_field, input_fields, gamma=1.0):
        """
        Growth of `target_field` is proportional to the product of values from `input_fields`.

        Args:
            target_field (str): Field to which the source applies (e.g., "population").
            input_fields (list of str): Fields to multiply together.
            gamma (float): Coefficient for growth scaling.
        """
        def expression_fn(fields):
            result = jnp.ones_like(fields[target_field].get_values())
            for name in input_fields:
                result *= fields[name].get_values()
            return result

        super().__init__(
            name=f"MultiplicativeGrowth({target_field} ← {' * '.join(input_fields)})",
            target_field_name=target_field,
            expression_fn=expression_fn,
            coefficient=gamma
        )

class LimitedMultiplicativeGrowthSource(SourceTerm):
    def __init__(self, target_field, input_fields, gamma=1.0, upper_limit=None, tau=10.0):
        """
        Growth of `target_field` is proportional to the product of values from `input_fields`,
        optionally capped by a smooth minimum with an `upper_limit` field.

        Args:
            target_field (str): Field to which the source applies (e.g., "population").
            input_fields (list of str): Fields to multiply together.
            gamma (float): Coefficient for growth scaling.
            upper_limit (str or None): If set, soft-min the growth rate with this field.
            softness (float): Controls smoothness of softmin transition.
        """
        def expression_fn(fields):
            result = jnp.ones_like(fields[target_field].get_values())
            for name in input_fields:
                result *= fields[name].get_values()

            if upper_limit is not None:
                cap = fields[upper_limit].get_values()
                result = - (1.0 / tau) * jnp.log(jnp.exp(-tau * result) + jnp.exp(-tau * cap))

            return result

        super().__init__(
            name=f"LimitedMultiplicativeGrowth({target_field} ← {' * '.join(input_fields)})",
            target_field_name=target_field,
            expression_fn=expression_fn,
            coefficient=gamma
        )

