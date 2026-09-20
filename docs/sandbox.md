# Using the field laboratory

1. Choose an experiment. It loads a reproducible world and starts paused.
2. Select the primary field and optionally a second field. Move over the map to inspect cell values.
3. Run, pause, or take one numerical step. Time is simulated time, not wall-clock time.
4. Change coefficients to alter the current world. Changing seed, resolution, boundaries, or experiment starts a fresh world.
5. Choose a brush field, add/remove mode, radius, and strength, then paint. Painting pauses evolution; press Run to resume.

## Reading the display

Maps have numeric color scales. Ranges can expand when higher values appear but do not contract during a run, so decline remains visible. Reset restores ranges. Soil is bounded between 0 and 1.

The population metric integrates density across the domain. Charts show selected field totals over actual simulation time; different fields have different meanings, so their totals should not be added. History storage is bounded.

Closed Neumann edges stop transport; periodic edges connect opposite sides and brushes wrap. Changing boundaries also rebuilds the initial landscape for that topology.

A brush applies a uniform density change to cell centers inside its circle. Removal stops at zero; soil restoration stops at one. Very small brushes may miss centers on coarse grids. Painting intentionally changes mass.

## Suggested interventions

- **Settlement:** add population far from fertile patches and compare food with population.
- **Collective investment:** reduce construction after infrastructure appears and observe whether settlements persist.
- **Overshoot:** compare soil and population, then increase recovery or restore land after decline.
- **Chemotaxis:** observe conserved totals before painting.

## Sharing and reset

Copy setup link includes experiment, seed, resolution, boundaries, and coefficients. It excludes evolved fields, interventions, time, playback speed, and chart history. Opening it starts a fresh world.

Reset uses current parameters and the selected seed. Select a gallery card again to restore its original defaults. Invalid links load safe defaults with an explanation.

The default 64×64 grid balances detail and speed. Choose 32×32 for faster experiments or 128×128 for finer structure. Requested playback speed may be limited by hardware. Numerical failures pause the run with an explanation; reset or reduce coefficients to continue.
