# Models and equations

The laboratory represents a landscape on a uniform two-dimensional grid. A field is a non-negative density or condition at each cell; it is not a collection of agents. The bundled models are qualitative experiments in spatial feedback. Their variables and time are dimensionless unless a legacy model says otherwise, and none is calibrated to a historical or ecological system.

## Ecology: food, water, cultivation, and soil

Ecology uses dynamic population `P`, food stock `F`, local water stock `W`, and soil condition `S`. Fertility `K` and the water-source map `Q` are static maps. Cultivation `C` is derived from population at every snapshot and cannot be painted:

```text
C = P / (P + a)
h(W) = W / (1 + W)
```

Here `a` is `cultivationScale`. The source map is clipped to `[0, 1]`; initial water is `W = sourceCapacity * Q`. The half-saturation constants in the denominators are one, except for the configurable food and water support scales below.

Define the local rates:

```text
H  = yield * C * K * S * h(W)                         harvest
R  = replenishmentRate * Q * max(1 - W/sourceCapacity, 0) source recharge
D  = waterConsumption * P * h(W)                     household withdrawal
A  = harvestWaterCost * H                             irrigation withdrawal
U  = consumption * P * F/(1 + F)                      food consumption
L  = spoilage * F                                      food spoilage
T  = min(F/foodSupport, W/waterSupport)                support level
G  = growth * P * (T - P)/(T + P + epsilon)            population source
```

The equations are:

```text
dP/dt = dp lap(P) - div(chiFood P grad(F))
                 - div(chiWater P grad(W)) + G
dF/dt = df lap(F) + H - U - L
dW/dt = dw lap(W) + R - D - A
dS/dt = soilRecovery * h(W) * (1-C)
                 * (1 - P/(1+P)) * (1-S)
         - [erosion*C + settlementErosion*P/(1+P)] * S
dK/dt = 0
dQ/dt = 0
C    = P/(P + cultivationScale)
```

Harvest therefore needs cultivation, fertility, soil, and water. Water use has two reported components: household withdrawal `D` and irrigation withdrawal `A`. Recharge slows linearly as a source approaches its `sourceCapacity` target and is zero at or above that target. `sourceCapacity` sets the level at which recharge stops; diffusion or painting can put a cell above it. `replenishmentRate` limits the incoming rate. The source map specifies replenishment locations and strengths, while the separate water stock diffuses and attracts population.

The bounded growth form remains safe when food or water is zero. `epsilon = 1e-6` prevents a zero denominator and the per-capita rate is bounded by `growth`; at a cell with no incoming transport, zero population remains zero until a population brush adds mass. Soil recovery is strongest on watered, lightly cultivated, lightly settled land. Cultivation and settlement pressure remove soil in proportion to the current soil stock, so the source respects the `[0, 1]` bound.

The browser groups ecology controls as Settlement, Water supply and demand, Food and cultivation, and Soil depletion and recovery. Grouping changes presentation only; every coefficient remains part of the reproducible setup.

### Ecology presets

All three presets use the same equations and deterministic initializer. They differ in parameters and starting conditions:

| Preset | Starting condition | Main contrast |
| --- | --- | --- |
| `water_settlement` | Normal initial population and full soil | Baseline settlement with moderate recharge, household use, irrigation cost, and soil pressure. |
| `water_overuse` | Population is scaled to `0.35` of the initializer | Household use is `0.25`, irrigation cost `0.45`, recharge `0.12`, and cultivation erosion `0.16`, making drawdown easier to see. |
| `soil_recovery` | Soil starts at `0.25` everywhere | Recovery is `0.18`, cultivation erosion `0.06`, and settlement erosion `0.015`; watered fallow land can rebuild soil. |

These settings are designed to expose feedbacks, not to predict thresholds or guarantee a particular outcome at every seed, grid, boundary, or parameter override.

## Civilization

Civilization uses population `P`, food `F`, infrastructure `I`, soil `S`, and static fertility `K`:

```text
B = investment * P * F/(1 + F)

dP/dt = dp lap(P) - div(chiFood P grad(F))
                 - div(chiInfra P grad(I))
                 + growth * P * (F-P)/(F+P+epsilon)
dF/dt = df lap(F)
         + regrowth * [K*S*(1 + infraBoost*I/(1+I)) - F]
         - consumption*P*F - foodCost*B
dI/dt = B - infraDecay*I
dS/dt = soilRecovery*(1-S) - erosion*P*S
dK/dt = 0
```

The `settlement`, `collective_investment`, and `overshoot` presets turn construction, infrastructure attraction, and erosion on or off to make different feedbacks legible. The growth, consumption, and infrastructure terms are phenomenological couplings, not conversions between physical units.

## Agriculture and chemotaxis baselines

The legacy agriculture model retains population, food, and static fertility:

```text
dP/dt = dp lap(P) - div(chiFood P grad(F))
                 + growth * P * (F-P)/(F+P+epsilon)
dF/dt = df lap(F) + regrowth*(K-F) - consumption*P*F
```

Its Python conventions use a 10 km domain and years. Chemotaxis removes all sources and sinks and keeps only diffusion and attraction. Under either boundary mode, transport and diffusion conserve each transported field to round-off until a brush changes it.

## Interpretation limits

There is no explicit trade, specialization, institution, culture, warfare, weather, groundwater basin, or individual decision-making. Diffusion and attraction stand for aggregate movement. A pattern that resembles a settlement is evidence about these local rules only.
