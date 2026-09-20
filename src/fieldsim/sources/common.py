import jax.numpy as jnp

from fieldsim.source_term import SourceTerm


class LogisticGrowthSource(SourceTerm):
    r"""Bounded logistic growth ``gamma * T * (U - T) / (U + T + eps)``.

    ``T`` is the growing field (``target``) and ``U`` is the field that plays
    the role of a carrying capacity (``upper_limit``).

    Why this form rather than the textbook ``gamma * T * (1 - T/U)``:

    * **Division safety.**  ``U`` is a dynamic field that can legitimately reach
      zero, and the old expression divided by it directly.  The blow-up was
      being masked by the softplus clamp on the fields; removing the clamp makes
      it fatal.  The denominator here is ``U + T + eps >= eps > 0`` for all
      non-negative ``T, U``.
    * **Bounded stiffness.**  The per-capita rate ``gamma (U - T)/(U + T + eps)``
      lies in ``[-gamma, +gamma]`` for all ``T, U >= 0``, so the explicit-Euler
      stability contribution of this term is just ``gamma`` -- independent of
      the state.  (The naive regularisation ``1 - T/max(U, eps)`` instead has an
      unbounded decay rate ``gamma*T/eps``.)
    * **Same qualitative behaviour.**  Equilibria are ``T = 0`` and ``T = U``;
      growth for ``T < U``, decay for ``T > U``; near ``T << U`` the per-capita
      rate tends to ``+gamma``.
    * **Graceful starvation.**  At ``U = 0`` it reduces to ``-gamma T^2/(T+eps)
      ~= -gamma T``, i.e. exponential decay, instead of ``-inf``.
    * **Positivity.**  The whole expression carries a factor ``T``, so it
      vanishes at ``T = 0``, and ``|dT/dt| <= gamma T`` means a step with
      ``dt*gamma <= 1`` cannot drive ``T`` negative.
    """

    def __init__(self, target, upper_limit, gamma=1.0, eps=1e-6):
        self.target_field = target
        self.upper_limit = upper_limit
        self.gamma = gamma
        self.eps = eps

        def expression_fn(values):
            T = values[target]
            U = values[upper_limit]
            return T * (U - T) / (U + T + eps)

        def max_rate_fn(values):
            return abs(gamma)

        super().__init__(
            name=f"{target} bounded logistic growth (capacity {upper_limit})",
            target_field_name=target,
            expression_fn=expression_fn,
            coefficient=gamma,
            max_rate_fn=max_rate_fn,
        )


class RelaxationSource(SourceTerm):
    r"""Linear relaxation toward a capacity field: ``rate * (K - F)``.

    ``F`` is ``target`` and ``K`` is ``capacity_field`` (typically a static
    fertility map).  Unlike a logistic regrowth term this does **not** vanish at
    ``F = 0``, so a fully depleted cell can recover.

    Positivity: at ``F = 0`` the term equals ``rate * K >= 0`` for ``K >= 0``,
    and the decay branch satisfies ``|dF/dt| <= rate * F`` whenever ``F > K``,
    so ``dt * rate <= 1`` keeps ``F`` non-negative.  Rate bound: ``rate``.
    """

    def __init__(self, target, capacity_field, rate=1.0):
        self.target_field = target
        self.capacity_field = capacity_field
        self.rate = rate

        def expression_fn(values):
            return values[capacity_field] - values[target]

        def max_rate_fn(values):
            return abs(rate)

        super().__init__(
            name=f"{target} relaxation toward {capacity_field}",
            target_field_name=target,
            expression_fn=expression_fn,
            coefficient=rate,
            max_rate_fn=max_rate_fn,
        )


class ConsumptionSource(SourceTerm):
    r"""Bilinear consumption of a resource by a consumer: ``-beta * C * T``.

    ``T`` is ``target`` (the resource being eaten) and ``C`` is ``consumer``.

    Positivity: the term carries a factor ``T`` and so vanishes at ``T = 0``;
    the local decay rate is ``beta * C``, hence the reported rate bound
    ``beta * max(C)``, which is state dependent and therefore re-evaluated by
    the runtime stability guard.
    """

    def __init__(self, target, consumer, beta=1.0):
        self.target_field = target
        self.consumer = consumer
        self.beta = beta

        def expression_fn(values):
            return -values[consumer] * values[target]

        def max_rate_fn(values):
            return abs(beta) * float(jnp.max(jnp.abs(values[consumer])))

        super().__init__(
            name=f"{target} consumption by {consumer}",
            target_field_name=target,
            expression_fn=expression_fn,
            coefficient=beta,
            max_rate_fn=max_rate_fn,
        )


class CivilizationFoodSource(SourceTerm):
    """Food renewal, consumption, and construction cost for civilization."""

    def __init__(self, target, population, fertility, soil, infrastructure,
                 regrowth, consumption, investment, infra_boost, food_cost):
        def build(values):
            P, F = values[population], values[target]
            return investment * P * F / (1.0 + F)

        def expression_fn(values):
            P = values[population]
            F = values[target]
            K = values[fertility]
            S = values[soil]
            I = values[infrastructure]
            capacity = K * S * (1.0 + infra_boost * I / (1.0 + I))
            return (
                regrowth * (capacity - F)
                - consumption * P * F
                - food_cost * build(values)
            )

        def max_rate_fn(values):
            max_population = float(jnp.max(values[population]))
            return (
                abs(regrowth)
                + abs(consumption) * max_population
                + abs(food_cost * investment) * max_population
            )

        super().__init__(
            name=f"{target} civilization balance",
            target_field_name=target,
            expression_fn=expression_fn,
            max_rate_fn=max_rate_fn,
        )


class InfrastructureSource(SourceTerm):
    """Construction ``investment*P*F/(1+F)`` minus infrastructure decay."""

    def __init__(self, target, population, food, investment, decay):
        def expression_fn(values):
            P, F, I = values[population], values[food], values[target]
            return investment * P * F / (1.0 + F) - decay * I

        super().__init__(
            name=f"{target} construction and decay",
            target_field_name=target,
            expression_fn=expression_fn,
            max_rate_fn=lambda values: abs(decay),
        )


class SoilSource(SourceTerm):
    """Bounded soil recovery and population-driven erosion."""

    def __init__(self, target, population, recovery, erosion):
        def expression_fn(values):
            S = values[target]
            return recovery * (1.0 - S) - erosion * values[population] * S

        def max_rate_fn(values):
            return abs(recovery) + abs(erosion) * float(jnp.max(values[population]))

        super().__init__(
            name=f"{target} recovery and erosion",
            target_field_name=target,
            expression_fn=expression_fn,
            max_rate_fn=max_rate_fn,
        )
