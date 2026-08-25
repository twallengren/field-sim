# field-sim

A small [JAX](https://github.com/jax-ml/jax)-based 2D reaction-advection-diffusion
field simulator. Diffusion is derived from a variational (Lagrangian
gradient-flow) formulation — `jax.grad` of a discrete energy functional yields
the diffusion stencil, rather than a hand-written finite-difference Laplacian —
and advection is a conservative finite-volume (donor-cell upwind) flux. Two
deterministic, seeded example configurations are bundled: an `agriculture`
model (population/food/fertility) and a `chemotaxis_demo` (pure transport,
used as the conservation reference case).

**Model scope, stated honestly:**

* Time integration is **explicit (forward) Euler only** — no implicit or
  adaptive stepping.
* All fields share **one isotropic scalar grid spacing `dx`** (a single
  uniform square grid per simulation; no per-axis or per-field resolution).
* Boundary conditions are **zero-flux (homogeneous Neumann) only**, built
  into the discrete operators themselves (no ghost cells, no other BC type
  is implemented).
* The reference implementation is **CPU-oriented**: it has been developed and
  tested against JAX's CPU backend; nothing prevents running under a GPU/TPU
  JAX build, but that configuration is untested here.

## Governing equations

### Agriculture (`fieldsim.simulations.agriculture`)

The flagship example. On the square domain `[0, L]²`, `L = 10` km, time in
years, with zero-flux (homogeneous Neumann) boundaries on every field:

```
∂P/∂t = D_P ∇²P − ∇·(χ P ∇F) + γ P (F − P)/(F + P + ε)
∂F/∂t = D_F ∇²F + r (K(x) − F) − β P F
```

Fields:

| symbol | code name    | meaning                        | units        | dynamic |
|--------|--------------|---------------------------------|--------------|---------|
| `P`    | `population` | population density              | people/km²   | yes     |
| `F`    | `food`       | food density                     | food/km²     | yes     |
| `K`    | `fertility`  | static soil/fertility capacity   | food/km²     | no      |

Parameters (defaults, from `src/fieldsim/simulations/agriculture.py`):

| symbol | code name          | default | units              | meaning |
|--------|--------------------|--------:|--------------------|---------|
| `D_P`  | `D_P`              | `0.1`   | km²/yr             | population dispersal (random-walk diffusivity) |
| `D_F`  | `D_F`              | `0.1`   | km²/yr             | food spreading diffusivity |
| `χ`    | `CHI`              | `0.1`   | km²·food⁻¹·yr⁻¹    | chemotactic mobility |
| `γ`    | `GAMMA`            | `0.5`   | 1/yr               | max per-capita growth rate |
| `r`    | `R`                | `0.2`   | 1/yr               | soil regrowth rate |
| `β`    | `BETA`             | `0.02`  | km²·person⁻¹·yr⁻¹  | per-capita consumption rate |
| `ε`    | (source `eps`)     | `1e-6`  | food/km²           | logistic division-safety regularisation (default of `LogisticGrowthSource`, not overridden) |
| —      | `FERTILITY_FLOOR`  | `0.25`  | food/km²           | baseline added to the random fertility bumps so no cell is permanently dead |
| `L`    | `DOMAIN_KM`        | `10.0`  | km                 | domain edge length |
| `n`    | `DEFAULT_N`        | `128`   | cells/edge         | grid resolution (`dx = L/n = 0.078125` km) |
| `T`    | `DEFAULT_TOTAL_TIME`| `10.0` | yr                 | physical duration integrated |

Signs and interpretation:

* `−∇·(χ P ∇F)`, `χ > 0`: population moves **up** the food gradient
  (attraction). The flux is proportional to `P`, so it vanishes wherever
  nobody lives.
* The growth term has per-capita rate `γ(F−P)/(F+P+ε) ∈ [−γ, +γ]`.
  Equilibria are `P = 0` and `P = F`: population grows toward locally
  available food and decays (rate `≈ −γ`) where there is none. It is
  division-safe at `F = 0` (denominator `≥ ε > 0`).
* `r(K − F)` is linear relaxation toward the local fertility; unlike a
  logistic term it does **not** vanish at `F = 0`, so a fully depleted cell
  recovers.
* `−βPF` is consumption; it vanishes at `F = 0` and at `P = 0`.

Conservation: both transport operators (diffusion, chemotaxis flux) are
exactly conservative, so with sources present

```
d/dt Σ P dx² = Σ (growth source) dx²
d/dt Σ F dx² = Σ (regrowth − consumption) dx²
```

with no boundary leakage — the budgets equal the volume integral of the
sources exactly, to machine precision.

### Chemotaxis demo (`fieldsim.simulations.chemotaxis_demo`)

The sources-off subset of the same model, kept as the conservation
integration example:

```
∂P/∂t = D_P ∇²P − ∇·(χ P ∇F)
∂F/∂t = D_F ∇²F
```

on `[0, L]²` with the same zero-flux boundaries. Both right-hand sides are
pure flux divergences that vanish on the domain boundary, so `Σ P dx²` and
`Σ F dx²` are each invariant to round-off — any drift is a bug (this is what
`tests/test_configs.py::test_chemotaxis_demo_conserves_both_totals` checks).

Parameters (`src/fieldsim/simulations/chemotaxis_demo.py`):

| symbol | code name | default | units             | meaning |
|--------|-----------|--------:|-------------------|---------|
| `D_P`  | `D_P`     | `0.05`  | km²/yr            | population dispersal |
| `D_F`  | `D_F`     | `0.02`  | km²/yr            | food spreading (slower than `D_P`, so the gradient survives) |
| `χ`    | `CHI`     | `0.3`   | km²·food⁻¹·yr⁻¹   | chemotactic mobility |
| `n`    | `DEFAULT_N` | `96`  | cells/edge        | grid resolution (`dx = 10/96 ≈ 0.104167` km) |
| `T`    | `DEFAULT_TOTAL_TIME` | `5.0` | yr        | physical duration integrated |

There are no sources in this configuration (`sources=[]`); `P` still climbs
the gradient of `F` while `F` spreads and flattens, so the animation is
visually interesting despite conserving both totals exactly.

## Numerics

**Grid.** Cell-centred: sample points are `x_i = (i + 1/2) dx`, `dx = L/n`,
covering `[0, L]` exactly (not `[0, L−dx]`). This is also what makes the
finite-volume flux and the face-difference energy consistent, and gives the
DCT-II cosine modes as exact eigenvectors of the discrete Neumann Laplacian.

**Variational diffusion** (`fieldsim/lagrangians/common.py`). The diffusion
energy is a sum over *interior faces only*:

```
E = (α/2) [ Σ (Δ_x φ)² + Σ (Δ_y φ)² ]
```

(no explicit `dx` factor: the continuum energy `(α/2) ∫|∇φ|² dA` approximates
the gradient as `(φ_{i+1} − φ_i)/dx` and the cell area is `dx²`, so
`(1/dx)² · dx² = 1` cancels exactly). `Lagrangian.functional_derivative` takes
`jax.grad` of the *total* energy with respect to the cell values and divides
by `dx²` (the explicit measure that makes the cancellation non-hidden). In 1D
this reproduces

```
δF/δφ_j = −α(φ_{j+1} − 2φ_j + φ_{j−1})/dx²
```

i.e. autodiff of the face-difference energy yields exactly the compact
5-point Laplacian `(φ_E + φ_W + φ_N + φ_S − 4φ_C)/dx²`. Its checkerboard
eigenvalue is `−8α/dx²` — the most strongly damped mode in the spectrum, so
grid-scale noise is damped rather than left undamped (a wide
central-difference stencil built from `jnp.gradient` has a *zero* eigenvalue
on the checkerboard mode — odd/even decoupling — and is not used here).

*Boundary treatment.* Omitting the boundary faces from the energy sum **is**
the variational statement of the zero-flux (homogeneous Neumann) condition: a
boundary cell simply has fewer face terms, giving the one-sided update
`φ_0 ← (1−c)φ_0 + c φ_1` with `c = dt·α/dx²`. No ghost cells are needed.

*Conservation.* Every interior face contributes `+d` to one cell and `−d` to
its neighbour, so `Σ φ dx²` is exactly invariant under diffusion alone (with
sources off).

**Donor-cell upwind chemotaxis flux** (`fieldsim/flux_terms/common.py`). Face
velocities are built from face differences of the attractant field `F`:

```
u = χ · diff(F, axis=1)/dx      # x-normal interior faces, shape (ny, nx-1)
v = χ · diff(F, axis=0)/dx      # y-normal interior faces, shape (ny-1, nx)
```

and the flux on each face carries the density of the *upwind* (donor) cell:

```
J_x = max(u, 0)·P_left + min(u, 0)·P_right
J_y = max(v, 0)·P_below + min(v, 0)·P_above
```

Divergence (`FluxTerm.divergence`) zero-pads the interior-face arrays to the
domain boundary (implicit zero flux through the boundary) and differences
them: `div = (diff(pad(J_x))_x + diff(pad(J_y))_y)/dx`. Every interior face
therefore appears exactly twice with opposite signs in the result, so
`Σ div(J) dx² = 0` by telescoping — this is the discrete conservation
statement `Σ φ dx² = const` under transport alone, exact to machine
precision for any well-formed flux, and it holds regardless of the sign or
smoothness of `flux_fn`.

*Positivity.* A cell can only lose what it holds (the outgoing face flux is
proportional to the donor cell's own value). A cell drains through *every*
face whose velocity points out of it, so the CFL condition under which no cell
can be driven negative is
`dt·max_cell[(u_R⁺ + u_L⁻ + v_U⁺ + v_D⁻)/dx] ≤ 1`, where `x⁺ = max(x,0)`,
`x⁻ = max(−x,0)`, the four terms are the cell's right/left/upper/lower faces
and boundary faces contribute zero. `AdvectionAlongGradientFlux.max_rate`
returns exactly this per-cell two-sided outflow maximum. The one-sided
`(max|u| + max|v|)/dx` is *not* an upper bound for it: wherever the attractant
has an interior local minimum, both faces of an axis drain the same cell and
the one-sided form under-counts by up to 2× per axis.

**Bounded logistic growth** (`fieldsim/sources/common.py`). The population
growth source uses `γ P (F−P)/(F+P+ε)` rather than the textbook
`γP(1 − P/F)`: the denominator `F+P+ε ≥ ε > 0` for all non-negative `P, F`,
so it is division-safe where the naive form (dividing by `F` directly) is
not. The per-capita rate `γ(F−P)/(F+P+ε)` lies in `[−γ, +γ]` for all
`P, F ≥ 0`, so the term's contribution to the explicit-Euler stability bound
is the state-independent constant `γ` (the naive regularisation
`1 − P/max(F, ε)` instead has an unbounded decay rate `γP/ε`). At `F = 0` the
term reduces to `≈ −γP` (graceful exponential starvation decay rather than
`−∞`).

**Timestep** (`src/fieldsim/stability.py`). Every operator declares an upper
bound on the rate (units 1/time) it imposes on the field it drives:

| term | `max_rate` |
|------|-----------|
| `Diffusion(alpha)` | `4·alpha/dx²` |
| `AdvectionAlongGradientFlux` | `max_cell[(u_R⁺ + u_L⁻ + v_U⁺ + v_D⁻)/dx]` |
| `LogisticGrowthSource` | `gamma` |
| `RelaxationSource` | `rate` |
| `ConsumptionSource` | `beta·max(consumer)` |

Rates are **summed per target field** (e.g. the food field's diffusion +
relaxation + consumption rates add up) and then **maximized across fields**;
call this `rate_sum`. The timestep is `dt = safety / rate_sum`, evaluated
once from the initial condition, with `safety = 0.8` by default (the `safety`
field of `SimulationConfig`). `n_steps = ceil(total_time / dt)` (integer-safe:
never truncates the requested duration short).

*Runtime guards.* Advective and consumption rates are state-dependent, so a
`dt` safe for the initial condition can become unsafe as the solution
evolves. `Simulator.check_state()` re-derives `rate_sum` from the *current*
fields and raises `RuntimeError` if `dt·rate_sum > 1` or if any field
contains a non-finite value; by default (`check_every=25`) this runs every 25
steps, and it also runs once at the end of `Simulator.__init__`, so a
hand-picked oversized `dt` is rejected before a single step is taken (rather
than after up to `check_every − 1` bad ones, or never when `check_every` is
disabled).

*Monitored positivity floor.* Positivity is a property of the discretisation
by construction (convex-combination diffusion under the CFL bound, donor-cell
upwind advection, sources that vanish at zero) — there is no unconditional
clamp. Round-off can still leave values of order `−eps`, so after each step
`Simulator._step_values` measures `neg_mass = −Σ min(φ, 0)·dx²`, applies
`φ ← max(φ, 0)`, and accumulates the removed mass in `Simulator.diagnostics`.
If a single step's `neg_mass` exceeds `truncation_tolerance × mass` (default
`truncation_tolerance = 1e-8`, plus an absolute floor `1e-30` so a
legitimately-zero field doesn't get a zero tolerance), the run raises
`RuntimeError`: round-off is recorded, a real scheme failure is caught, and
nothing is silently hidden. A non-finite measurement is itself treated as a
violation (every comparison with `NaN` is `False`, so `neg_mass > budget`
alone would let a blown-up state slip through the mask).

**Reproducibility.** `get_config(seed=...)` is a pure function of its
arguments: `np.random.default_rng(seed)` is constructed inside `get_config`
and threaded explicitly into `generate_random_bump_specs(rng, ...)` for every
field's initial condition; no module-level RNG state is consumed on import
(`tests/test_import_safety.py`). Identical seeds reproduce bit-identical
initial conditions and, since the step function is deterministic,
bit-identical trajectories (`tests/test_reproducibility.py`).

*jit determinism caveat.* `Simulator._step_values` is wrapped once in
`jax.jit`. In the library's default float32 precision, the jitted and eager
(`jax.disable_jit()`) code paths can in principle differ at the unit-in-last-
place level because the compiler may reorder floating-point operations
differently than the eager interpreter. `tests/test_jit.py::test_jit_consistency`
therefore runs under the `x64` fixture (float64) and asserts `atol=1e-12,
rtol=0.0` agreement between jitted and eager execution over 50 steps — a
tolerance tight enough to catch a real divergence but not so tight that it
would be sensitive to legitimate float64 operation-reordering.

## Install & run

Requires Python `>=3.10` (per `pyproject.toml`; developed against 3.11).

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
fieldsim-run --sim agriculture --seed 0 --no-anim
fieldsim-run --sim chemotaxis_demo --save out.gif
pytest
```

(The commands above were re-run from a clean shell against the repo's
existing `.venv` — `pip install -e ".[dev]"` is idempotent against an
already-installed editable checkout — to confirm they work as documented.)

CLI flags (`fieldsim-run` / `python -m fieldsim`, from `src/fieldsim/main.py`):

| flag | default | meaning |
|------|---------|---------|
| `--sim {agriculture,chemotaxis_demo}` | `agriculture` | which bundled simulation to run |
| `--seed SEED` | `0` | RNG seed for initial-condition generation |
| `--years YEARS` | config's own default | override `total_time` (years) |
| `--no-anim` | off | run only; print a text summary instead of animating |
| `--save PATH` | none | save the animation to `PATH` (`.gif` via `PillowWriter`; `.mp4` via `ffmpeg` if available, else falls back to `.gif`) instead of showing it interactively |
| `--max-frames MAX_FRAMES` | `150` | maximum number of history snapshots recorded |

A headless (`Agg`) matplotlib backend is selected automatically whenever
`--save` or `--no-anim` is used, before `matplotlib.pyplot` is imported
anywhere in the process.

Approximate runtimes measured on this machine (CPU, single process):

* `fieldsim-run --sim agriculture --seed 0 --no-anim` (default `n=128`,
  875 steps, `dt≈0.0114`): **≈2.5 s CPU** (2.22 s user + 0.27 s system;
  ≈2.2 s wall).
* `fieldsim-run --sim chemotaxis_demo --seed 0 --save out.gif` (default
  `n=96`): most of the ≈16 s wall time is GIF encoding (`PillowWriter`), not
  the simulation itself.
* `pytest` (48 tests): ≈38 s.

## Testing

Run with `pytest` (or `.venv/bin/pytest -q`); `tests/conftest.py` provides an
`x64` fixture (enables/restores JAX float64 for tests that assert
machine-precision numerical claims) and small grid/mass helpers shared across
the suite. Per test file:

* `tests/test_operators.py` — operator-level guarantees on plain arrays (no
  `Field`, no floor): diffusion conserves mass to `1e-12` and stays
  non-negative; a discrete Neumann eigenmode decays at exactly the predicted
  rate `(1 − dt·α·λ_h)^N`; population climbs a static food gradient while its
  total is conserved to `1e-12`; a sharp density step advected downgradient
  never goes negative (donor-cell positivity, zero floor activation), and
  neither does a density sitting in a V-shaped (or pyramidal) attractant
  valley, where the cell drains through both/all four faces at once and the
  declared `max_rate` must be the two-sided one; any
  `FluxTerm` telescopes to zero net divergence; the bounded logistic term
  stays finite and non-negative even where the capacity field is zero.
* `tests/test_integration.py` — `Field`/`Simulator` integration: all-zero
  initial conditions are an exact fixed point of the full agriculture system;
  full dynamics from random bumps stay non-negative with negligible
  cumulative floor truncation; field/simulator construction validates shape,
  `dx`, `bc_type`, and positive `dt`; an oversized `dt` is rejected at
  construction, and a `dt` made unsafe afterwards still trips the runtime
  guard, as do injected non-finite values; the derived `dt` matches
  the rate bound exactly; the floor diagnostics correctly report an injected
  negative excursion.
* `tests/test_configs.py` — the two bundled configurations: forward Euler's
  first-order convergence via Richardson extrapolation (halving `dt` should
  roughly halve the error; ratio asserted in `[1.6, 2.4]`); both configs run
  end-to-end headlessly, finite and non-negative, with the static fertility
  field verified to stay static and the chemotaxis-only config's population
  and food totals conserved.
* `tests/test_reproducibility.py` — `get_config(seed=...)` produces
  bit-identical initial conditions and bit-identical short trajectories for
  the same seed.
* `tests/test_jit.py` — the jitted and eager (`jax.disable_jit()`) step paths
  agree to `atol=1e-12` in float64 over 50 steps; the runtime guards and
  diagnostics flush still work under jit (including a `NaN` injected into a
  field, which trips the truncation guard at the next sync); consecutive steps
  reuse one compiled executable (no retracing).
* `tests/test_runner.py` — `SimulationRunner`'s bounded/strided history
  (float32 snapshots, correct stride, streaming min/max matching a direct
  computation over the full history), the single-frame and many-field
  animation edge cases, and end-to-end headless CLI subprocess runs
  (`--no-anim` and `--save`).
* `tests/test_import_safety.py` — importing every `fieldsim` submodule
  consumes no global `numpy` RNG state and opens no matplotlib figures.
* `tests/test_smoke.py` — the core modules import cleanly under the
  src-layout package.

## Package layout

```
src/fieldsim/
    main.py                       CLI entry point (argparse), `fieldsim-run`
    __main__.py                   `python -m fieldsim` entry point
    field.py                      Field: cell-centred array + metadata container
    lagrangian_term.py            LagrangianTerm base class (energy contract)
    lagrangian.py                 Lagrangian: total energy + functional_derivative
    lagrangians/common.py         Diffusion (face-difference energy)
    flux_term.py                  FluxTerm base class (face-flux divergence contract)
    flux_terms/common.py          AdvectionAlongGradientFlux (donor-cell chemotaxis)
    source_term.py                SourceTerm base class
    sources/common.py             LogisticGrowthSource, RelaxationSource, ConsumptionSource
    simulator.py                  Simulator: jit-compiled forward-Euler step + guards
    simulation_config.py          SimulationConfig: declarative simulation description
    simulation_runner.py          SimulationRunner: builds fields/dt, runs, animates
    stability.py                  timestep / rate-bound derivation
    simulations/agriculture.py    flagship population/food/fertility config
    simulations/chemotaxis_demo.py transport-only conservation reference config
    utils/constants.py            field-name string constants
    utils/generators.py           seeded random Gaussian-bump initial-condition generator
tests/                            pytest suite (see Testing above)
pyproject.toml
LICENSE
```

## Limitations / future work

* **Forward Euler only.** No implicit, semi-implicit, or adaptive
  (variable-`dt`) time stepping; the diffusion timestep bound
  (`dt ≤ dx²/(4α)` per field) is the classic explicit-scheme constraint, so
  fine grids are diffusion-limited.
* **Neumann-only boundaries.** `bc_type="neumann"` is the only supported
  value (validated at `Field` construction); there is no Dirichlet, periodic,
  or mixed boundary support.
* **Single isotropic scalar `dx`.** One grid spacing for the whole domain and
  every field in a simulation; no adaptive mesh refinement or per-axis
  resolution.
* **Historical note on the checkerboard mode.** An earlier implementation
  built the diffusion energy from `jnp.gradient` (a central-difference
  approximation), which produced a wide `[1, 0, −2, 0, 1]/(4dx²)` stencil
  with a *zero* eigenvalue on the checkerboard mode — grid-scale noise was
  completely undamped. The face-difference energy described above fixes
  this (checkerboard eigenvalue `−8α/dx²`, the most strongly damped mode);
  it is noted here only because the failure mode is a common pitfall for
  this style of variational discretisation.
