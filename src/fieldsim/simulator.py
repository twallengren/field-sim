import math

import jax
import jax.numpy as jnp
import numpy as np

from fieldsim.stability import rate_sum

#: Absolute mass floor added to the per-step truncation tolerance so that a
#: field whose total mass is (legitimately) zero does not have a zero tolerance.
_ABS_MASS_FLOOR = 1e-30

#: Upper bound on the number of un-synchronised per-step floor measurements
#: held on device.  Only reached when ``check_every`` is disabled: it keeps the
#: truncation guard alive (and the pending list bounded) in that case too.
_MAX_PENDING = 1000


class Simulator:
    """Forward-Euler integrator for ``dphi/dt = -deltaF/delta phi - div(J) + S``.

    Boundary conditions are built into the operators (zero-flux Neumann or
    wrap-around periodic), so the step loop never mutates boundary rows/columns.
    The old ghost-copy Neumann
    update (``f[0] = f[1]`` ...) destroyed mass conservation and double-wrote
    the corners; it is gone.

    Positivity is a property of the discretisation (see
    :mod:`fieldsim.stability`), but round-off can still produce values of order
    ``-eps``.  Rather than silently clamping, each step measures the mass it
    would remove, ``neg = -sum(min(phi, 0)) * dx**2``, applies the floor, and
    records the result in :attr:`diagnostics`.  If a single step truncates more
    than ``truncation_tolerance`` times that field's mass the run stops with a
    ``RuntimeError``: round-off is recorded, a real bug is caught, and nothing
    is hidden.

    The stability bound is checked **at construction** (``__init__`` ends with
    :meth:`check_state`) as well as every ``check_every`` steps, so an oversized
    hand-picked ``dt`` raises before a single bad step is taken rather than
    after up to ``check_every - 1`` of them -- or never, when ``check_every`` is
    disabled.  Because that check runs at construction, every
    ``LagrangianTerm``/``FluxTerm``/``SourceTerm`` passed in must implement a
    real ``max_rate`` (a term that raises ``NotImplementedError`` there can no
    longer be used with ``check_every=None`` to dodge the requirement -- the
    constructor call to :meth:`check_state` triggers it regardless).

    Performance / host-device split
    -------------------------------
    The numerics live in :meth:`_step_values`, a **pure** function of the
    ``{name: array}`` state dict plus scalar timestep, wrapped in ``jax.jit``
    once in ``__init__``. Operators, ``dx`` and dynamic field names are closed
    over; ``dt`` remains a dynamic scalar so adaptive changes reuse the same
    executable. The trace happens exactly once (``Lagrangian.functional_derivative`` in
    particular no longer re-differentiates the energy every step).

    Static (non-dynamic) fields are **passed through the argument dict**, not
    closed over.  Closing over them would bake their values into the compiled
    code, so a later ``set_values`` on e.g. the fertility map would be silently
    ignored; passing them keys the jit cache only on their shape/dtype, which
    never changes during a run.  They are returned unchanged and are not
    written back.

    The per-step positivity bookkeeping is done **on device**: the step returns
    0-d arrays and no ``float()`` is taken inside it.  Those arrays are
    accumulated in a small host-side list and synchronised once every
    ``check_every`` steps (and whenever :attr:`diagnostics` is read, or
    :meth:`sync_diagnostics` is called).  This removes a per-step
    device-to-host sync -- the single biggest cost in the old eager loop -- and
    lets JAX's asynchronous dispatch keep the device queue full.

    The one semantic consequence: the per-step truncation guard is *evaluated*
    per step but *reported* at the end of the check window it falls in, i.e. up
    to ``check_every - 1`` steps late.  The exception names the offending step
    and the window it was detected in.
    """

    def __init__(self, fields: dict, lagrangian, sources: list, flux_terms: list,
                 dt: float = None, check_every: int = 25,
                 truncation_tolerance: float = 1e-8, adaptive: bool = False,
                 safety: float = 0.8, max_dt: float = 0.1,
                 derived_fields=None):
        """
        Args:
            fields: dict of {field_name: Field}. All fields must share one shape
                and one grid spacing.
            lagrangian: Lagrangian holding the free-energy terms.
            sources: list of SourceTerm objects.
            flux_terms: list of FluxTerm objects.
            dt: timestep size.
            check_every: run the finiteness/stability guard, and flush the
                positivity diagnostics, every N steps (``None`` or 0 disables
                the periodic check; diagnostics are still flushed on demand).
            truncation_tolerance: maximum fraction of a field's mass that the
                positivity floor may remove in a single step.
            adaptive: derive a fresh stable timestep before every step.
            safety: fraction of the rate bound used by adaptive stepping.
            max_dt: upper bound on an adaptive timestep.
        """
        if not fields:
            raise ValueError("Simulator requires at least one field.")
        if not adaptive and (dt is None or not math.isfinite(dt) or not dt > 0):
            raise ValueError(f"dt must be positive, got {dt!r}.")
        if not 0.0 < safety < 1.0:
            raise ValueError(f"safety must lie in (0, 1), got {safety!r}.")
        if not math.isfinite(max_dt) or not max_dt > 0.0:
            raise ValueError(f"max_dt must be positive, got {max_dt!r}.")
        if (
            not math.isfinite(truncation_tolerance)
            or truncation_tolerance < 0.0
        ):
            raise ValueError(
                "truncation_tolerance must be finite and non-negative, got "
                f"{truncation_tolerance!r}."
            )

        shapes = {name: tuple(f.shape) for name, f in fields.items()}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ValueError(
                f"All fields must share one grid shape; got {shapes!r}."
            )
        spacings = {name: float(f.dx) for name, f in fields.items()}
        if len(set(spacings.values())) != 1:
            raise ValueError(
                f"All fields must share one grid spacing dx; got {spacings!r}."
            )
        boundaries = {name: f.bc_type for name, f in fields.items()}
        if len(set(boundaries.values())) != 1:
            raise ValueError(
                f"All fields must share one boundary condition; got {boundaries!r}."
            )

        self.fields = fields
        self.shape = unique_shapes.pop()
        self.dx = next(iter(spacings.values()))
        self.bc_type = next(iter(boundaries.values()))
        self.lagrangian = lagrangian
        self.sources = sources
        self.flux_terms = flux_terms
        self.adaptive = bool(adaptive)
        self.safety = float(safety)
        self.max_dt = float(max_dt)
        self.derived_fields = dict(derived_fields or {})
        overlap = set(self.derived_fields) & set(fields)
        if overlap:
            raise ValueError(f"Derived fields duplicate stored fields: {sorted(overlap)}.")
        self.dt = float(dt) if dt is not None else self.max_dt
        self.check_every = check_every
        self.truncation_tolerance = float(truncation_tolerance)
        self.time = 0.0
        self.step_count = 0

        self.dynamic_names = tuple(
            name for name, field in fields.items() if field.is_dynamic
        )
        for term in list(lagrangian.terms) + list(flux_terms):
            term_bc = getattr(term, "bc_type", None)
            if term_bc is not None and term_bc != self.bc_type:
                raise ValueError(
                    f"Term {term.name!r} uses boundary condition {term_bc!r}, "
                    f"but fields use {self.bc_type!r}."
                )
        self._diagnostics = {
            name: {"cumulative_truncated_mass": 0.0, "last_truncated_mass": 0.0}
            for name in self.dynamic_names
        }
        #: Un-synchronised (device-resident) per-step ``(neg_mass, mass)`` pairs,
        #: one entry per step taken since the last flush.
        self._pending = []

        # Compile once.  ``self._step_values`` is a bound method, so the same
        # callable object -- and therefore the same jit cache -- is reused for
        # every step; nothing here is rebuilt inside the loop.
        self._jitted_step = jax.jit(self._step_values)
        # Validate the timestep against the *initial* state before anyone can
        # step.  Without this, a hand-constructed Simulator with an oversized
        # dt takes up to ``check_every - 1`` garbage steps before the periodic
        # guard fires -- and never fires at all when ``check_every`` is falsy.
        # ``stable_dt`` callers pass a dt that satisfies this by construction,
        # so only a hand-picked dt can trip it.  (A caller that deliberately
        # wants an unstable run can still assign to ``simulator.dt`` after
        # construction; the periodic guard catches it without retracing.)
        if self.adaptive:
            self.dt = self.safe_timestep()
        self.check_state()

    # ------------------------------------------------------------------
    # Diagnostics (host side, lazily synchronised)
    # ------------------------------------------------------------------

    @property
    def diagnostics(self):
        """Positivity-floor diagnostics, synchronised on read.

        Same keys and shape as before jit-compilation
        (``{field: {"cumulative_truncated_mass", "last_truncated_mass"}}``);
        reading it flushes any steps taken since the last check so the numbers
        are always up to date.  Because the flush also applies the per-step
        truncation guard, reading this attribute can raise the same
        ``RuntimeError`` a periodic check would have raised.
        """
        return self.sync_diagnostics()

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def _step_values(self, values: dict, dt) -> tuple:
        """One forward-Euler step + positivity floor, as a pure function.

        Args:
            values: ``{field_name: array}`` for **all** fields, dynamic and
                static alike.

        Returns:
            ``(new_values, neg_mass, mass)`` where ``new_values`` holds the
            updated, floored values of the *dynamic* fields only (static fields
            are unchanged, so they are not copied back), ``neg_mass[name]`` is
            the 0-d mass ``-sum(min(phi, 0)) * dx**2`` removed by the floor, and
            ``mass[name]`` is the 0-d post-floor total mass ``sum(phi) * dx**2``
            that the truncation tolerance is measured against.

        No ``float()``/``bool()`` is taken here and there is no Python-level
        branching on a traced value: the guards are pure host-side logic and
        live in :meth:`sync_diagnostics` / :meth:`check_state`.
        """
        dx = self.dx
        new_values = {}
        neg_mass = {}
        mass = {}

        for name in self.dynamic_names:
            # Gradient flow from the free energy: -delta F / delta phi.
            dF_dphi = self.lagrangian.functional_derivative(values, dx, name)

            # Conservative transport: -div(J).
            flux_sum = sum(
                flux.divergence(values, dx)
                for flux in self.flux_terms
                if flux.target == name
            )

            # Local sources/sinks.
            source_sum = sum(
                s.evaluate(values) for s in self.sources if s.target == name
            )

            updated = values[name] + dt * (-dF_dphi - flux_sum + source_sum)

            # Monitored positivity floor, entirely on device.
            neg_mass[name] = -jnp.sum(jnp.minimum(updated, 0.0)) * dx ** 2
            floored = jnp.maximum(updated, 0.0)
            new_values[name] = floored
            mass[name] = jnp.sum(floored) * dx ** 2

        return new_values, neg_mass, mass

    def safe_timestep(self, remaining=None):
        """Return the current adaptive Euler step without mutating the state."""
        total = self._current_rate()
        dt = self.max_dt if total <= 0.0 else min(self.max_dt, self.safety / total)
        if remaining is not None:
            if not math.isfinite(remaining) or remaining <= 0.0:
                raise ValueError(f"remaining must be positive, got {remaining!r}.")
            dt = min(dt, float(remaining))
        return dt

    def _current_rate(self):
        values = {name: field.get_values() for name, field in self.fields.items()}
        return rate_sum(
            values, self.lagrangian.terms, self.flux_terms, self.sources, self.dx
        )

    def step(self, dt=None, remaining=None):
        """Advance by one step and return the step count.

        Adaptive runs recompute a safe step from the current state. ``remaining``
        caps the step so an :meth:`advance` call lands exactly on its target.
        The timestep is a dynamic JAX argument, so changing it does not retrace.
        """
        if remaining is not None and (
            not math.isfinite(remaining) or not remaining > 0.0
        ):
            raise ValueError(f"remaining must be positive, got {remaining!r}.")
        if dt is None:
            if self.adaptive:
                step_dt = self.safe_timestep(remaining)
            else:
                step_dt = min(self.dt, float(remaining)) if remaining is not None else self.dt
        else:
            if not math.isfinite(dt) or not dt > 0.0:
                raise ValueError(f"dt must be positive, got {dt!r}.")
            step_dt = min(float(dt), float(remaining)) if remaining is not None else float(dt)
        total = self._current_rate() if dt is not None else None
        limit = self.safety if self.adaptive else 1.0
        if total is not None and step_dt * total > limit * (1.0 + 1e-12):
            raise RuntimeError(
                f"Requested timestep dt={step_dt:.6g} is unsafe at t={self.time:.6g}: "
                f"dt * rate_sum = {step_dt * total:.6g} > {limit:g}."
            )
        self.dt = float(step_dt)

        values = {name: field.get_values() for name, field in self.fields.items()}
        dtype = next(iter(values.values())).dtype
        new_values, neg_mass, mass = self._jitted_step(
            values, jnp.asarray(step_dt, dtype=dtype)
        )

        for name, updated in new_values.items():
            self.fields[name].set_values(updated)

        self.time += step_dt
        self.step_count += 1
        self._pending.append((neg_mass, mass, self.time, step_dt))

        if self.check_every and self.step_count % self.check_every == 0:
            self.sync_diagnostics()
            self.check_state()
        elif len(self._pending) >= _MAX_PENDING:
            self.sync_diagnostics()

        return self.step_count

    def advance(self, duration):
        """Advance by ``duration`` and land on the requested time exactly."""
        if not math.isfinite(duration) or not duration > 0.0:
            raise ValueError(f"duration must be positive, got {duration!r}.")
        target = self.time + float(duration)
        start_step = self.step_count
        while self.time < target:
            remaining = target - self.time
            if (
                self.step_count > start_step
                and remaining <= 1e-12 * max(1.0, abs(target))
            ):
                self.time = target
                break
            self.step(remaining=remaining)
            if target - self.time <= 1e-12 * max(1.0, abs(target)):
                self.time = target
                break
        return self.step_count

    def sync_diagnostics(self):
        """Pull the pending per-step floor measurements to the host.

        This is the *only* device-to-host synchronisation in the step loop.  It
        folds every pending step into :attr:`diagnostics` and then applies the
        per-step truncation guard to each of them, raising if any single step in
        the window removed more than ``truncation_tolerance`` times that field's
        mass.

        Returns the diagnostics dict (the same object every time, so callers
        that stashed a reference keep seeing updates).
        """
        pending, self._pending = self._pending, []
        if not pending:
            return self._diagnostics

        first_step = self.step_count - len(pending) + 1
        last_step = self.step_count
        neg_pending, mass_pending, times_pending, _ = zip(*pending)
        # A single blocking transfer for the whole window (a handful of scalars).
        neg_host, mass_host = jax.device_get((neg_pending, mass_pending))

        violation = None
        for name in self.dynamic_names:
            negs = np.array([float(entry[name]) for entry in neg_host])
            masses = np.array([float(entry[name]) for entry in mass_host])

            diag = self._diagnostics[name]
            # ``+ 0.0`` normalises the -0.0 that arises when nothing was cut.
            diag["cumulative_truncated_mass"] += float(negs.sum()) + 0.0
            diag["last_truncated_mass"] = float(negs[-1]) + 0.0

            budgets = self.truncation_tolerance * masses + _ABS_MASS_FLOOR
            # A NaN state makes both measurements NaN, and *every* comparison
            # with a NaN is False -- so a plain ``negs > budgets`` would let a
            # blown-up run slip through the guard entirely (until the next
            # ``check_state``, which ``check_every=None`` disables).  Treat a
            # non-finite measurement as a violation in its own right.
            offending = np.flatnonzero(
                (negs > budgets) | ~np.isfinite(negs) | ~np.isfinite(masses)
            )
            if offending.size and violation is None:
                index = int(offending[0])
                violation = (
                    name,
                    first_step + index,
                    float(times_pending[index]),
                    float(negs[index]),
                    float(masses[index]),
                    float(budgets[index]),
                )

        if violation is not None:
            name, step, step_time, truncated, mass, budget = violation
            if not (math.isfinite(truncated) and math.isfinite(mass)):
                raise RuntimeError(
                    f"Field {name!r} produced non-finite positivity diagnostics "
                    f"at step {step} (t={step_time:.6g}): truncated mass "
                    f"{truncated!r}, post-floor mass {mass!r}. The state contains "
                    "NaN or inf, so the run has already diverged; the violation "
                    f"is reported at the end of its check window (steps "
                    f"{first_step}-{last_step})."
                )
            raise RuntimeError(
                f"Positivity floor truncated {truncated:.6g} of mass from field "
                f"{name!r} at step {step} (t={step_time:.6g}), exceeding the "
                f"allowed {budget:.6g} ({self.truncation_tolerance:g} x mass "
                f"{mass:.6g}). The violation occurred within the last check "
                f"window (steps {first_step}-{last_step}, checked every "
                f"{self.check_every} steps), so it is reported here rather than "
                "on the offending step itself. This indicates a genuine scheme "
                "failure, not round-off."
            )
        return self._diagnostics

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def check_state(self):
        """Assert finiteness and that ``dt`` still satisfies the stability bound.

        Advective and consumption rates are state dependent, so a timestep that
        was safe for the initial condition can become unsafe as the solution
        evolves.  This re-derives the bound from the *current* fields.
        """
        values = {name: field.get_values() for name, field in self.fields.items()}

        for name, array in values.items():
            if not bool(jnp.all(jnp.isfinite(array))):
                raise RuntimeError(
                    f"Field {name!r} contains non-finite values at step "
                    f"{self.step_count} (t={self.time:.6g})."
                )

        total = rate_sum(
            values, self.lagrangian.terms, self.flux_terms, self.sources, self.dx
        )
        limit = self.safety if self.adaptive else 1.0
        if not self.adaptive and self.dt * total > limit * (1.0 + 1e-12):
            raise RuntimeError(
                f"Timestep dt={self.dt:.6g} violates the stability bound at step "
                f"{self.step_count} (t={self.time:.6g}): dt * rate_sum = "
                f"{self.dt * total:.6g} > {limit:g} (rate_sum={total:.6g}). "
                "Reduce dt or the transport/reaction coefficients."
            )
        return total

    def get_state(self):
        state = {name: field.get_values() for name, field in self.fields.items()}
        state.update({name: derive(state) for name, derive in self.derived_fields.items()})
        return state
