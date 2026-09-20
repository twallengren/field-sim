# Numerical methods and guarantees

## Grid and boundaries

All fields share a uniform two-dimensional grid. The bundled domain has side length `L = 10`; cell centers are `(i + 1/2) dx`, with `dx = L/n`, and arrays are row-major. Python also accepts rectangular arrays in lower-level APIs; the browser and bundled setups use square grids.

Neumann means homogeneous zero flux across exterior faces. It does not copy boundary rows, zero a field, or create internal obstacles. Periodic connects opposite edges for diffusion, upwind transport, and brushes. Every operator in a run uses the selected boundary mode.

## Diffusion and conservative transport

The Python diffusion term is the gradient of a discrete face energy:

```text
E = alpha/2 * sum_faces (phi_right - phi_left)^2
diffusion RHS = -grad(E)/dx^2
```

The browser evaluates the equivalent five-point stencil. Neumann sums interior faces; periodic includes the seam. Diffusion conserves the field integral to round-off.

For population density `P` moving up an attractant `A`, each oriented face uses donor-cell flux:

```text
u = chi * (A_right - A_left)/dx
J = max(u, 0)*P_left + min(u, 0)*P_right
```

The two adjacent cells receive opposite flux updates. Ecology adds separate population transports up food and water. An exterior Neumann face has zero flux; a periodic seam is an ordinary face. An empty donor cannot export mass.

## Explicit Euler and adaptive step bounds

The browser and civilization/ecology Python configurations use forward Euler with safety factor `0.8` and maximum timestep `0.1`. Before each step, the runtime derives a fractional loss-rate bound `R` from the current state and takes `dt = min(0.1, 0.8/R, remaining duration)`. If `R` is zero, it uses `0.1`. This is a stability bound, not an error estimate; convergence still requires changing grid and timestep scales.

The ecology bound includes:

```text
population: 4*dp/dx^2 + food-attraction outflow + water-attraction outflow + growth
food:       4*df/dx^2 + consumption*max(P) + spoilage
water:      4*dw/dx^2 + maximum local recharge/withdrawal loss rate
soil:       maximum local (recovery coefficient + depletion coefficient)
```

The transport contribution is the maximum sum of all outgoing face rates from one cell. A largest single-axis speed is insufficient because a cell can drain through several faces. Water, food, and soil source terms are evaluated from the pre-step state, and the same rates feed the water ledger. Changing a parameter or painting a field recomputes the next bound before evolution continues.

Legacy agriculture and chemotaxis Python demos retain their fixed-step behavior. JAX receives changing ecology timesteps as data rather than compiling one function per timestep. Browser advances are capped by a requested step count, so a slow device advances less simulated time rather than violating the bound.

## Positivity, bounds, and diagnostics

After each Euler update, dynamic fields are checked for finite values. Small negative round-off is floored to zero; soil, water-source quality, and derived cultivation are limited to `[0, 1]` where applicable. A correction is allowed only up to `1e-8` times the post-floor field mass plus a tiny absolute floor. A larger correction or any non-finite value stops the step. The cumulative correction is exposed as `truncatedMass`/diagnostics and should normally be zero.

Population, food, and water are stocks and are not generally conserved in ecology: harvest, consumption, spoilage, recharge, and withdrawals change their totals. Transport and diffusion alone conserve their integrals. The water snapshot reports initial stock, cumulative recharge, household withdrawal, irrigation withdrawal, water interventions, current stock, and the residual

```text
initial + recharge + water interventions
  - household withdrawal - irrigation withdrawal - current
```

up to floating-point round-off. This ledger distinguishes water painted directly into the stock from source-map painting. Painting `water` changes the stock and is counted as an intervention. Painting `waterSources` changes only future recharge quality and does not inject current water; its separate field intervention diagnostic still records the source-map edit. Painting `cultivation` is rejected because cultivation is derived from population.

## Reproducibility

Ecology uses the same uint32 Mulberry32 stream and Gaussian bump initializer in Python and the browser. The legacy population, food, and fertility bump draws come first; ecology then consumes five source-map bumps with amplitudes `[0.55, 1]`, widths `[0.45, 1]`, and a zero floor. Periodic initialization uses shortest wrapped distance. Initial water is `sourceCapacity * waterSources`, and `water_overuse` scales initial population by `0.35`; `soil_recovery` starts soil at `0.25`.

The browser uses `Float64Array`; Python fixture generation enables JAX float64. Parity tests use identical arrays and timesteps with `atol=1e-9`, `rtol=1e-9`. Long nonlinear runs can diverge in their final low bits across runtimes or hardware even when the local equations and safeguards agree.

Python is CPU-oriented. GPU/TPU performance, anisotropic grids, mixed edge conditions, internal walls, implicit solvers, and arbitrary equation editing are outside the bundled contract.
