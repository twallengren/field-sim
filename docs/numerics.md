# Numerical methods and guarantees

## Grid and boundaries

Fields share one uniform 2D grid and scalar spacing `dx`. Centers are `(i+1/2)dx`. Bundled domains have side length 10. Python supports rectangular arrays; bundled configurations and the browser use square grids.

**Neumann** means homogeneous zero flux across exterior faces. It does not copy boundary rows, zero the field, or create internal obstacles. **Periodic** connects opposite edges, including diffusion and upwind seam fluxes. Fields and transport operators must agree on boundary mode.

## Variational diffusion

Python differentiates a discrete energy with JAX:

```text
E = alpha/2 * sum_faces (phi_right - phi_left)^2
diffusion RHS = -grad(E)/dx^2
```

Neumann sums interior faces; periodic includes wraparound faces. Cell area cancels the squared gradient's spacing factors in the 2D energy. The resulting compact five-point Laplacian damps checkerboard modes and conserves the field integral. The browser evaluates the equivalent stencil directly, without autodiff.

## Conservative upwind transport

For density `P` and attractant `A`, an oriented face carries:

```text
u = chi * (A_right - A_left)/dx
J = max(u,0)*P_left + min(u,0)*P_right
```

Each face contributes equal and opposite changes to adjacent cells. Neumann exterior faces carry zero flux; periodic faces connect opposite edges. An empty donor cannot export mass.

The original `FluxTerm.flux_fn(values, dx)` returns interior arrays `(ny,nx-1)` and `(ny-1,nx)`. A periodic custom flux supplies `periodic_flux_fn`, returning two `(ny,nx)` arrays for right and upper face fluxes. Divergence subtracts rolled incoming faces.

## Stability and time

Integration is forward Euler. Each term declares a bound on the fractional loss rate of its target field. Sum contributions per target, then take the largest field sum `R`.

| Term | Rate bound |
| --- | --- |
| Diffusion | `4 alpha/dx²` |
| Attraction | Maximum per-cell sum of all outgoing face speeds divided by `dx` |
| Population growth | `growth` |
| Agriculture food loss | `regrowth + consumption*max(P)` |
| Civilization food loss | Above, plus `foodCost*investment*max(P)` |
| Infrastructure loss | `infraDecay` |
| Soil | `soilRecovery + erosion*max(P)` |

Food and infrastructure attraction contribute separately. The largest speed per axis alone is unsafe: a cell can drain through opposing faces simultaneously.

Adaptive mode chooses `dt=min(0.1,0.8/R,remaining_duration)` before each step. Zero rates use the maximum timestep. The soil bound preserves both 0 and 1. Painting and coefficient changes precede the next rate calculation. Playback never relaxes the numerical bound.

Legacy Python demos default to fixed steps selected from the initial state, with runtime guards detecting later violations. Civilization and the browser use adaptive stepping. This is stability-based adjustment, **not error-controlled integration**; convergence studies still require grid and timestep refinement.

JAX receives changing timesteps as data, avoiding repeated compilation. Histories record actual simulation time. Browser batches are bounded, so slow hardware advances less simulated time rather than compromising stability.

## Conservation and diagnostics

Transport conserves `sum(phi)dx²` to round-off. With reactions, total changes equal integrated sources and sinks. Population and food are not conserved in civilization runs.

A monitored positivity floor records round-off correction. A material correction beyond `1e-8` times field mass plus a tiny absolute floor stops the run. Non-finite states are errors. Browser soil upper-bound correction is also monitored. Brushes have separate signed integrated mass accounting.

## Reproducibility

Civilization uses the same uint32 Mulberry32 stream and Gaussian initializer in both runtimes. Periodic maps use shortest wrapped distances. Legacy Python demos retain their NumPy initializer; equal seeds do not imply equal browser baseline maps.

The browser uses `Float64Array`. Python normally uses installed JAX precision; fixture generation explicitly enables float64. Parity checks use identical arrays and timesteps, `atol=1e-9`, `rtol=1e-9`, and one-step/short trajectories under both boundaries, including soil depletion. Long nonlinear runs need not remain bit-identical across runtimes or hardware.

Python is CPU-oriented. GPU/TPU performance, anisotropic grids, mixed edge conditions, internal walls, implicit solvers, and arbitrary browser equation editing are not implemented.
