# Models: civilization through fields

The experiment asks what spatial patterns arise when local concentrations affect migration, production, investment, and regeneration. A settlement is a concentration of population density. Infrastructure is accumulated collective work. Neither requires individual agents, city objects, or predetermined collapse events.

## Civilization equations

For the civilization model, all variables, the domain length `L=10`, and time are dimensionless. Its dynamic fields are population `P`, food `F`, infrastructure `I`, and soil health `S` in `[0,1]`; fertility `K` is static productive potential.

Let construction `B = a P F/(1+F)`:

```text
dP/dt = Dp lap(P) - div(chiF P grad(F)) - div(chiI P grad(I))
        + g P (F-P)/(F+P+epsilon)
dF/dt = Df lap(F) + r [K S (1 + q I/(1+I)) - F] - beta P F - c B
dI/dt = B - delta I
dS/dt = rho (1-S) - e P S
dK/dt = 0
```

`epsilon=1e-6`. Food and infrastructure attract population. Infrastructure improves production with diminishing returns, costs food to build, and decays after its builders leave. Population pressure degrades soil; soil recovers independently. Zero population stays zero unless painted in.

| Symbol | Catalog key | Meaning |
| --- | --- | --- |
| Dp, Df | `dp`, `df` | Population dispersal and food spreading |
| chiF, chiI | `chiFood`, `chiInfra` | Attraction to food and infrastructure |
| g | `growth` | Bounded growth/starvation rate |
| r | `regrowth` | Food renewal toward productive capacity |
| beta | `consumption` | Population consumption |
| a | `investment` | Construction rate |
| delta | `infraDecay` | Infrastructure loss |
| q | `infraBoost` | Saturating productivity benefit |
| c | `foodCost` | Construction food cost |
| rho | `soilRecovery` | Soil recovery |
| e | `erosion` | Population-driven soil loss |

The shared catalog is the source of defaults and browser ranges. Python accepts finite nonnegative overrides outside those interface ranges; extreme rates can make runs expensive.

## Three experiments, one model

**Settlement** disables construction, infrastructure attraction, and erosion. Food availability organizes the population toward a resource-limited pattern.

**Collective investment** enables construction and infrastructure attraction, keeping erosion disabled. Built capacity persists and influences future growth. Food costs and crowding can offset productivity benefits; more investment does not guarantee more population everywhere. The preset uses moderate attraction to avoid rapid grid-scale concentration.

**Overshoot and recovery** combines investment, strong erosion, and slow regeneration. Population initially expands, then depleted soil reduces production. Recovery is a process to investigate through interventions, not a guaranteed repeating cycle.

For illustration, the default overshoot run at `n=32`, seed `0`, and closed boundaries peaks near total population `62.9` at time `18`, then falls to about `32.4` at time `100`, with mean soil health `0.243`. These figures describe that setup. Seeds 7 and 42 and periodic boundaries also produced overshoot in the checked runs.

## Agriculture and chemotaxis baselines

The original agriculture model is:

```text
dP/dt = Dp lap(P) - div(chiF P grad(F)) + g P (F-P)/(F+P+epsilon)
dF/dt = Df lap(F) + r(K-F) - beta P F
```

Its Python conventions use a 10 km domain and time in years. The growth law numerically interprets food as a population carrying capacity: this is a phenomenological coupling, not a calibrated conversion between physical food units and people.

Chemotaxis retains only `Dp lap(P) - div(chiF P grad(F))` and `Df lap(F)`. Both totals are conserved to round-off under either boundary mode. Painting deliberately changes totals and is accounted for separately.

## Limits of interpretation

There is no explicit trade, specialization, institution, culture, warfare, or individual decision-making. Diffusion and attraction represent aggregate movement. These models demonstrate consequences of stated local rules; resemblance to a city does not establish historical realism.
