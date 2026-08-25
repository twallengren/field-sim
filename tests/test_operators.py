"""Phase 2a operator-level tests.

These tests exercise the operators directly on plain ``jnp`` arrays -- they
never construct a ``Field`` and never apply any floor/clip, so a positivity
assertion here is a genuine statement about the *discretisation*, not about a
clamp hiding the result.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fieldsim.flux_term import FluxTerm
from fieldsim.flux_terms.common import AdvectionAlongGradientFlux
from fieldsim.lagrangian import Lagrangian
from fieldsim.lagrangians.common import Diffusion
from fieldsim.sources.common import (
    ConsumptionSource,
    LogisticGrowthSource,
    RelaxationSource,
)

from conftest import cell_center_grid, cell_centers, gaussian, x_centroid

PHI = "phi"
POP = "pop"
FOOD = "food"


def _lagrangian(*terms):
    lag = Lagrangian()
    for term in terms:
        lag.add_term(term)
    return lag


# --------------------------------------------------------------------------
# Diffusion
# --------------------------------------------------------------------------


def test_diffusion_conserves_mass(x64):
    """Face-difference diffusion is exactly conservative and positivity preserving."""
    n = 32
    L = 1.0
    dx = L / n
    alpha = 0.05
    dt = 0.8 * dx ** 2 / (4.0 * alpha)

    X, Y = cell_center_grid(n, dx)
    phi = jnp.asarray(gaussian(X, Y, 0.5 * L, 0.5 * L, sigma=0.12 * L, amp=1.0))

    lag = _lagrangian(Diffusion(target=PHI, alpha=alpha))

    m0 = float(jnp.sum(phi)) * dx ** 2
    assert m0 > 0.0

    for _ in range(500):
        dF = lag.functional_derivative({PHI: phi}, dx, PHI)
        phi = phi - dt * dF

    mass = float(jnp.sum(phi)) * dx ** 2
    assert abs(mass - m0) / m0 < 1e-12

    # Under dt <= dx^2/(4 alpha) every update is a convex combination of
    # non-negative numbers, so this holds exactly -- no floor is applied.
    assert float(jnp.min(phi)) >= 0.0

    # And the bump has actually spread (this is not a no-op).
    assert float(jnp.max(phi)) < 0.9


def test_discrete_decay_exact(x64):
    """A discrete Neumann eigenmode decays at exactly the predicted rate.

    On a cell-centred grid the eigenfunctions of the zero-flux 5-point
    Laplacian are the DCT-II modes ``cos(m pi x_c / L)``.  Taking m = 2 in both
    directions, the discrete eigenvalue of ``-lap_5`` is

        lambda_h = (4/dx^2) * (sin^2(k dx / 2) + sin^2(k dx / 2)),  k = 2 pi / L

    so explicit Euler on ``dphi/dt = -delta F/delta phi = alpha lap_5 phi`` must
    reproduce ``A_N = (1 - dt alpha lambda_h)^N A_0`` to round-off.  This
    simultaneously validates the stencil *and* the variational treatment of the
    boundary: a wrong boundary handling changes the eigenvector and the test
    fails immediately.

    The mode is sign-changing; that is deliberate.  No floor is applied here.
    """
    n = 32
    L = 1.0
    dx = L / n
    alpha = 0.05
    dt = 0.8 * dx ** 2 / (4.0 * alpha)
    steps = 100

    xc = cell_centers(n, dx)
    k = 2.0 * np.pi / L
    mode = np.outer(np.cos(k * xc), np.cos(k * xc))  # cos(k y_c) cos(k x_c)
    amp0 = 1.0
    phi = jnp.asarray(amp0 * mode)

    lam_h = (4.0 / dx ** 2) * (np.sin(k * dx / 2.0) ** 2 + np.sin(k * dx / 2.0) ** 2)

    lag = _lagrangian(Diffusion(target=PHI, alpha=alpha))
    for _ in range(steps):
        phi = phi - dt * lag.functional_derivative({PHI: phi}, dx, PHI)

    expected = (1.0 - dt * alpha * lam_h) ** steps * amp0
    assert 0.0 < expected < 1.0

    # Project onto the mode (robust to any residual component).
    phi_np = np.asarray(phi, dtype=np.float64)
    measured = float(np.sum(phi_np * mode) / np.sum(mode * mode))
    assert measured == pytest.approx(expected, rel=1e-10)

    # The state is still (numerically) a pure mode: no other component grew.
    residual = phi_np - measured * mode
    assert np.max(np.abs(residual)) < 1e-10 * expected


# --------------------------------------------------------------------------
# Advection / chemotaxis
# --------------------------------------------------------------------------


def _advective_dt(flux, values, dx, alpha=0.0, safety=0.8):
    rate = flux.max_rate(values, dx) + 4.0 * alpha / dx ** 2
    return safety / rate


def test_chemotaxis_sign_and_conservation(x64):
    """Population climbs the food gradient, and total population is conserved."""
    n = 48
    L = 10.0
    dx = L / n
    chi = 0.2
    alpha = 0.01
    steps = 200

    X, Y = cell_center_grid(n, dx)
    food = jnp.asarray(gaussian(X, Y, 0.75 * L, 0.5 * L, sigma=0.25 * L, amp=10.0))
    pop = jnp.asarray(gaussian(X, Y, 0.25 * L, 0.5 * L, sigma=0.10 * L, amp=1.0))

    flux = AdvectionAlongGradientFlux(target_field=POP, gradient_field=FOOD, kappa=chi)
    lag = _lagrangian(Diffusion(target=POP, alpha=alpha))

    values = {POP: pop, FOOD: food}
    dt = _advective_dt(flux, values, dx, alpha=alpha)

    m0 = float(jnp.sum(pop)) * dx ** 2
    c0 = x_centroid(pop, dx)

    for _ in range(steps):
        values = {POP: pop, FOOD: food}
        dF = lag.functional_derivative(values, dx, POP)
        div = flux.divergence(values, dx)
        pop = pop - dt * dF - dt * div

    mass = float(jnp.sum(pop)) * dx ** 2
    assert mass == pytest.approx(m0, rel=1e-12)

    c1 = x_centroid(pop, dx)
    assert c1 - c0 > 0.05 * L, f"centroid moved {c1 - c0}, expected > {0.05 * L}"

    # Donor-cell + sub-CFL diffusion: still non-negative with no floor.
    assert float(jnp.min(pop)) >= 0.0


def test_upwind_positivity_no_floor():
    """A sharp density step advected hard downgradient never goes negative.

    Runs in the default float32 precision on purpose: the donor-cell positivity
    guarantee is structural, not a precision artefact.
    """
    n = 48
    L = 10.0
    dx = L / n
    chi = 0.5
    steps = 200

    X, _ = cell_center_grid(n, dx)
    # Steep, monotone attractant: a tanh ramp centred in the domain.
    food = jnp.asarray(20.0 * np.tanh((X - 0.5 * L) / (0.05 * L)))
    # Sharp step: full on the left half, exactly zero on the right half.
    pop = jnp.asarray(np.where(X < 0.5 * L, 1.0, 0.0))

    flux = AdvectionAlongGradientFlux(target_field=POP, gradient_field=FOOD, kappa=chi)
    values = {POP: pop, FOOD: food}
    dt = _advective_dt(flux, values, dx)

    m0 = float(jnp.sum(pop)) * dx ** 2

    minimum = 0.0
    for _ in range(steps):
        pop = pop - dt * flux.divergence({POP: pop, FOOD: food}, dx)
        minimum = min(minimum, float(jnp.min(pop)))

    assert minimum >= 0.0, f"donor-cell scheme produced {minimum} < 0"
    assert float(jnp.sum(pop)) * dx ** 2 == pytest.approx(m0, rel=1e-5)
    # Mass really did pile up on the right of the step.
    assert x_centroid(pop, dx) > 0.25 * L


@pytest.mark.parametrize("two_dimensional", [False, True], ids=["x-valley", "pyramid"])
def test_upwind_positivity_divergent_flow(two_dimensional):
    """Positivity holds where a cell drains through *both* faces of an axis.

    ``test_upwind_positivity_no_floor`` uses a monotone attractant, so every
    cell has exactly one outflowing face per axis and the loosest possible CFL
    rate (``max|u|/dx``) already suffices.  That is structurally blind to the
    worst case: an attractant with an interior local **minimum**.  Here

        F = A |x - x_c|                     (and optionally + A |y - y_c|)

    with the valley sitting exactly on a cell centre.  With ``kappa > 0`` the
    density climbs the gradient, so the valley cell pushes mass out through its
    left *and* right faces simultaneously (and, in the pyramid case, through all
    four faces).  Its true fractional outflow per step is therefore

        dt * (u_R^+ + u_L^- + v_U^+ + v_D^-) / dx,

    up to 2x (4x in 2D) the one-sided ``(max|u| + max|v|)/dx`` surrogate.  A
    ``max_rate`` that reports the surrogate hands back a dt that drives the
    valley cell negative on the very first step.

    Run at the library's default float32 and with no floor of any kind applied
    (plain arrays, no ``Field``/``Simulator``), so ``min(P) >= 0`` here is a
    statement about the discretisation and the declared rate, not about a clamp.
    """
    n = 32
    L = 10.0
    dx = L / n
    chi = 1.0
    amp = 2.0
    steps = 200

    X, Y = cell_center_grid(n, dx)
    centres = cell_centers(n, dx)
    # Valley on a cell centre (index n//2), so the minimum cell really does have
    # an outflowing face on both sides; a valley on a face would give u = 0
    # there and quietly reduce to the one-sided case.
    x_c = centres[n // 2]
    y_c = centres[n // 2]

    food = amp * np.abs(X - x_c)
    if two_dimensional:
        food = food + amp * np.abs(Y - y_c)
    food = jnp.asarray(food)

    # All the mass in the minimum cell(s): the fastest-draining location.
    pop = np.zeros((n, n))
    if two_dimensional:
        pop[n // 2, n // 2] = 1.0
    else:
        pop[:, n // 2] = 1.0
    pop = jnp.asarray(pop)

    flux = AdvectionAlongGradientFlux(target_field=POP, gradient_field=FOOD, kappa=chi)
    values = {POP: pop, FOOD: food}
    rate = flux.max_rate(values, dx)

    # The declared rate must bound the true two-sided outflow of the worst cell.
    faces = 4.0 if two_dimensional else 2.0
    assert rate == pytest.approx(faces * chi * amp / dx, rel=1e-5)

    dt = 0.8 / rate
    m0 = float(jnp.sum(pop)) * dx ** 2

    minimum = 0.0
    for _ in range(steps):
        pop = pop - dt * flux.divergence({POP: pop, FOOD: food}, dx)
        minimum = min(minimum, float(jnp.min(pop)))

    assert minimum >= 0.0, f"donor-cell scheme produced {minimum} < 0"
    assert float(jnp.sum(pop)) * dx ** 2 == pytest.approx(m0, rel=1e-5)
    # The flow is genuinely divergent: mass left the valley in both directions.
    pop_np = np.asarray(pop, dtype=np.float64)
    left = pop_np[:, : n // 2].sum()
    right = pop_np[:, n // 2 + 1:].sum()
    assert left > 0.1 * pop_np.sum() and right > 0.1 * pop_np.sum()


def test_flux_divergence_sums_to_zero(x64):
    """Any FluxTerm conserves mass: the telescoping property, tested directly."""
    rng = np.random.default_rng(0)
    ny, nx = 17, 23
    dx = 0.37

    def flux_fn(values, dx):
        return (
            jnp.asarray(rng.normal(size=(ny, nx - 1))),
            jnp.asarray(rng.normal(size=(ny - 1, nx))),
        )

    term = FluxTerm(name="random", target=PHI, flux_fn=flux_fn)
    div = term.divergence({PHI: jnp.zeros((ny, nx))}, dx)
    assert div.shape == (ny, nx)
    assert abs(float(jnp.sum(div))) < 1e-10


def test_flux_term_without_max_rate_raises():
    term = FluxTerm(
        name="no-rate",
        target=PHI,
        flux_fn=lambda values, dx: (jnp.zeros((3, 2)), jnp.zeros((2, 3))),
    )
    with pytest.raises(NotImplementedError):
        term.max_rate({PHI: jnp.zeros((3, 3))}, 1.0)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def test_logistic_safety():
    """Bounded logistic growth is finite and sane where the capacity is zero."""
    n = 32
    gamma = 0.5
    dt = 0.5  # dt * gamma = 0.25 < 1
    steps = 100

    capacity = np.ones((n, n))
    capacity[:, : n // 2] = 0.0          # left half: no food at all
    capacity[:, n // 2:] = 2.0           # right half: capacity 2
    food = jnp.asarray(capacity)
    pop = jnp.ones((n, n))

    source = LogisticGrowthSource(target=POP, upper_limit=FOOD, gamma=gamma)

    for _ in range(steps):
        pop = pop + dt * source.evaluate({POP: pop, FOOD: food})
        assert bool(jnp.all(jnp.isfinite(pop)))

    pop_np = np.asarray(pop, dtype=np.float64)
    starved = pop_np[:, : n // 2]
    fed = pop_np[:, n // 2:]

    assert np.all(np.isfinite(pop_np))
    assert np.all(starved < 1.0)             # strictly decayed
    assert np.all(starved >= 0.0)            # but never negative
    assert np.max(starved) < 1e-3            # essentially wiped out
    assert np.allclose(fed, 2.0, atol=1e-5)  # moved to the capacity


def test_logistic_per_capita_rate_is_bounded():
    """|rate| <= gamma for every non-negative (T, U) pair, including U = 0."""
    gamma = 0.5
    source = LogisticGrowthSource(target=POP, upper_limit=FOOD, gamma=gamma)

    grid = np.array([0.0, 1e-12, 1e-6, 1e-3, 1.0, 10.0, 1e6])
    T, U = np.meshgrid(grid, grid, indexing="ij")
    out = np.asarray(source.evaluate({POP: jnp.asarray(T), FOOD: jnp.asarray(U)}))

    assert np.all(np.isfinite(out))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_capita = np.where(T > 0, out / np.where(T > 0, T, 1.0), 0.0)
    assert np.max(np.abs(per_capita)) <= gamma * (1.0 + 1e-6)
    assert np.all(out[T == 0.0] == 0.0)
    assert source.max_rate({POP: jnp.asarray(T), FOOD: jnp.asarray(U)}) == gamma


def test_relaxation_source():
    """r*(K-F) pushes F toward K from both sides and revives F = 0."""
    rate = 0.2
    source = RelaxationSource(target=FOOD, capacity_field="fertility", rate=rate)

    K = jnp.asarray(np.array([[1.0, 3.0], [0.0, 2.0]]))
    F = jnp.asarray(np.array([[0.0, 5.0], [0.0, 2.0]]))
    out = np.asarray(source.evaluate({FOOD: F, "fertility": K}))

    np.testing.assert_allclose(out, rate * (np.asarray(K) - np.asarray(F)), rtol=1e-6)
    assert out[0, 0] > 0.0    # F = 0 with K > 0 regrows
    assert out[0, 1] < 0.0    # F > K decays
    assert out[1, 1] == 0.0   # F == K is an equilibrium
    assert source.max_rate({FOOD: F, "fertility": K}) == rate


def test_consumption_source():
    """-beta*C*T vanishes at T = 0 and reports rate beta*max(C)."""
    beta = 0.02
    source = ConsumptionSource(target=FOOD, consumer=POP, beta=beta)

    C = jnp.asarray(np.array([[0.0, 4.0], [3.0, 1.0]]))
    T = jnp.asarray(np.array([[5.0, 0.0], [2.0, 1.0]]))
    out = np.asarray(source.evaluate({FOOD: T, POP: C}))

    np.testing.assert_allclose(out, -beta * np.asarray(C) * np.asarray(T), rtol=1e-6)
    assert out[0, 1] == 0.0   # nothing to eat
    assert out[0, 0] == 0.0   # nobody eating
    assert source.max_rate({FOOD: T, POP: C}) == pytest.approx(beta * 4.0)
