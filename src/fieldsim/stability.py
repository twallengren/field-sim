r"""Explicit-Euler timestep selection.

Every operator in the library declares an upper bound on the rate (units of
1/time) it imposes on the field it drives:

======================  ==========================================
term                    ``max_rate``
======================  ==========================================
``Diffusion(alpha)``    ``4*alpha/dx**2``
advective flux          ``(max|u| + max|v|)/dx``
bounded logistic        ``gamma``
relaxation              ``r``
consumption             ``beta*max(C)``
======================  ==========================================

For one field, the forward-Euler update can be written as

    phi_new = (1 - dt * R) * phi + (non-negative inflow / production terms)

where ``R`` is the *sum* of the rates of the terms acting on that field.  So
the condition for both stability and positivity is ``dt * R <= 1``; we take
``dt = safety / max_over_fields(R)`` with ``safety = 0.8``.

Rates are therefore summed **per target field** and the maximum is taken across
fields.  In the standard one-diffusion-term-per-field case this reduces to the
familiar ``4*alpha_max/dx**2 + (max|u| + max|v|)/dx + s_max``; grouping by
target also gets the food field's ``r + beta*max(P)`` right automatically.

Advective and consumption rates depend on the current state, so the returned dt
is only valid for the state it was computed from.  :class:`Simulator` therefore
re-evaluates :func:`rate_sum` every ``check_every`` steps and raises if the
bound has been violated.
"""

import math


def _as_values(fields_or_values):
    """Accept either ``{name: Field}`` or ``{name: array}`` and return arrays."""
    return {
        name: (obj.get_values() if hasattr(obj, "get_values") else obj)
        for name, obj in fields_or_values.items()
    }


def rate_contributions(fields_or_values, lagrangian_terms, flux_terms, sources, dx):
    """Return ``{field_name: summed rate}`` for the current state."""
    values = _as_values(fields_or_values)
    per_field = {}
    for term in list(lagrangian_terms) + list(flux_terms) + list(sources):
        rate = float(term.max_rate(values, dx))
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError(
                f"Term {term.name!r} reported a non-finite/negative max_rate {rate!r}."
            )
        per_field[term.target] = per_field.get(term.target, 0.0) + rate
    return per_field


def rate_sum(fields_or_values, lagrangian_terms, flux_terms, sources, dx):
    """Largest per-field summed rate; ``dt * rate_sum <= 1`` is the bound."""
    per_field = rate_contributions(
        fields_or_values, lagrangian_terms, flux_terms, sources, dx
    )
    if not per_field:
        return 0.0
    return max(per_field.values())


def stable_dt(fields_or_values, lagrangian_terms, flux_terms, sources, dx,
              safety=0.8):
    """Timestep satisfying ``dt * rate_sum == safety`` for the given state."""
    if not 0.0 < safety < 1.0:
        raise ValueError(f"safety must lie in (0, 1), got {safety!r}.")
    total = rate_sum(fields_or_values, lagrangian_terms, flux_terms, sources, dx)
    if total <= 0.0:
        raise ValueError(
            "Cannot derive a timestep: no term imposes a positive rate. "
            "Supply dt explicitly for a trivial (no-dynamics) configuration."
        )
    return safety / total


def n_steps(total_time, dt):
    """Integer-safe ``ceil(total_time / dt)``, validated to be >= 1."""
    if not total_time > 0:
        raise ValueError(f"total_time must be positive, got {total_time!r}.")
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt!r}.")
    steps = math.ceil(total_time / dt)
    if steps < 1:
        raise ValueError(
            f"total_time={total_time!r} and dt={dt!r} give steps={steps}, which is < 1."
        )
    return steps
