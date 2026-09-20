# Shared implementation contract

This file describes the Python/browser agreement. The catalog, TypeScript contracts, and worker are executable parts of the same contract; update them together when an interface changes.

## Domain, arrays, and initialization

The domain is a square of side `L = 10`. Cell centers are `(i + 0.5)L/n`; flattened arrays are row-major with y increasing upward in the field model. Browser ecology resolutions are 32, 64, and 128. Ecology values and time are dimensionless.

Both runtimes use the uint32 Mulberry32 stream. Each bump consumes five draws: x, y, amplitude, sigma, then one reserved draw. Population, food, and fertility use the legacy sequences and ranges. Ecology then draws five source bumps with amplitude `[0.55, 1]`, sigma `[0.45, 1]`, and zero floor, clips `waterSources` to `[0, 1]`, and initializes `water = sourceCapacity * waterSources`. `water_overuse` scales initial population by `0.35`; `soil_recovery` initializes soil to `0.25`; all other ecology soil starts at `1`. Periodic maps use shortest wrapped distances. `cultivation = population/(population + cultivationScale)` is derived after initialization and after every population or parameter change.

Ecology fields are `population`, `food`, `water`, and `soil` (dynamic), `fertility` and `waterSources` (static but paintable), and `cultivation` (derived and read-only). Civilization fields remain population, food, infrastructure, soil, and fertility. The browser field descriptors expose kind, editability, palette, and optional bounds.

## Equations and stepping

Ecology uses population diffusion and donor-cell attraction up food and water; food and water also diffuse. Its local rates are:

```text
C = P/(P + cultivationScale), h = W/(1 + W)
H = yield*C*K*S*h
R = replenishmentRate*Q*max(1 - W/sourceCapacity, 0)
D = waterConsumption*P*h, A = harvestWaterCost*H
U = consumption*P*F/(1 + F), L = spoilage*F
T = min(F/foodSupport, W/waterSupport)
G = growth*P*(T-P)/(T+P+1e-6)
```

The source equations are `F' = H-U-L`, `W' = R-D-A`, and `S' = soilRecovery*h*(1-C)*(1-P/(1+P))*(1-S) - (erosion*C + settlementErosion*P/(1+P))*S`; population adds `G` to its diffusion and transport RHS. `K` and `Q` are static. `sourceCapacity` is the recharge target rather than a hard upper bound on water stock. All RHS terms in one Euler step read the old state.

Explicit Euler uses safety `0.8`, maximum `dt = 0.1`, and `dt = min(0.1, 0.8/R, remaining)` for the current combined rate bound, with `0.1` when `R = 0`. Ecology bounds population diffusion plus both outgoing transport rates and growth; food diffusion plus consumption and spoilage; water diffusion plus its maximum local recharge/withdrawal loss rate; and soil by local recovery plus depletion coefficients. A requested timestep above the safe bound is rejected.

After each step, dynamic fields must be finite, non-negative, and soil must remain in `[0, 1]`. Corrections are permitted only below `1e-8 * postFloorMass + 1e-30`; larger corrections abort the step. Snapshots copy arrays and expose cumulative correction and intervention diagnostics.

## Water accounting and interventions

The browser ecology snapshot reports `initial`, `recharged`, `domesticUse`, `agriculturalUse`, `interventions`, `current`, and `residual`. The residual is `initial + recharged + interventions - domesticUse - agriculturalUse - current` and should be round-off. Direct water painting changes the stock and increments `interventions`; source-map painting changes only future recharge and does not inject water. Cultivation painting is rejected.

## Browser engine and worker

`Simulation` in `web/src/engine/simulation.ts` exposes `constructor(setup, initialFields?)`, `step(dtOverride?)`, `advance(duration, maxSteps)`, `setParameters(parameters)`, `paint(brush)`, `snapshot()`, and `rateBound(...)`. A supplied timestep is accepted only when finite, positive, and within the current safe bound. Snapshots contain detached field arrays. `advance` may stop at `maxSteps`; callers use returned simulation time rather than assuming the requested duration was reached. The worker serializes `init`, `advance`, `step`, `parameters`, and `paint` commands and ignores stale generations.

## Tiles, rendering, and sharing

A `TileConfig` has a stable ID, ordered unique `layers`, and one `paintField` selected from those layers. A layer stores `field`, `opacity`, and `visible`. The UI permits any number of tiles; a one-layer tile selects that layer automatically. Derived fields can be rendered and selected in a tile, but a paint attempt is rejected because the field is read-only. The renderer keeps one color scale per field key, so repeated fields share a scale, and it records one history series per unique visible field. Layer order controls compositing order; visibility controls both rendering and history selection.

Version 2 setup serialization contains preset, seed, resolution, boundary, the complete model parameter map, and tile layout. It excludes evolved arrays, time, brush interventions, water ledger history, speed, and chart history. Version 1 setup links remain valid for the original non-ecology presets; ecology setup links require version 2. Malformed links fall back to safe defaults with an explanatory status.

## Ownership and parity

The Python model and reference fixtures own the numerical reference. The browser engine owns the registered TypeScript implementation and worker protocol. UI owns controls and tile editing; rendering owns raster maps, shared scales, and history. Integration owns generation changes, worker messaging, pointer painting, and setup-link actions. Parity fixtures are generated from Python, never from browser output, and compare identical arrays/timesteps under both boundaries at `atol=1e-9`, `rtol=1e-9`.
