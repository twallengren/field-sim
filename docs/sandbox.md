# Using the field laboratory

Choose an experiment from the gallery. It loads a deterministic world and starts paused. Run, pause, or take one stable numerical step; simulation time is independent of wall-clock time. Changing the seed, resolution, boundary, preset, or a model coefficient starts or rebuilds the current setup as appropriate. A parameter change keeps the current fields and changes the next right-hand side.

## Compose field views

Ecology opens with several tiles. Add as many tiles as are useful for the question, then choose the layers in each tile. Layers render in their listed order and each layer has independent visibility and opacity. A tile's paint target is selected separately; with one layer it is selected automatically. A target must be one of that tile's layers.

The same field uses one color scale everywhere it appears, so a population map in two tiles remains comparable. Scales may expand when a higher value appears and do not contract during a run; Reset restores them. The chart records totals once per visible field, even when that field appears in multiple tiles or layers. Hidden fields do not add a history series. Static fertility and water sources can be viewed and painted; derived cultivation is displayed read-only and follows population immediately.

Move over a map to inspect all fields at a cell. A brush changes the selected tile's paint target, pauses evolution, and must be followed by Run to continue. Removal stops at zero; soil and source quality stop at one. A small brush can miss cell centers on a coarse grid. Periodic brushes wrap across the edge; Neumann brushes must remain inside the domain.

## Ecology walkthroughs

For `water_settlement`, create one tile with population and water layers, then another with water sources and cultivation. To paint water, add a water layer to the second tile and select it as that tile's paint target, or paint the water layer in the first tile. Add water to see an immediate stock intervention; the water budget labels it as painted water. Compare the same tile after running: recharge is gradual and capped by the local source capacity target.

For `water_overuse`, keep water and population visible together and watch the budget. Household withdrawal and irrigation withdrawal reduce the stock, while the source map only controls how quickly recharge returns. To test source quality, paint `waterSources` instead; current water and the water budget ledger do not change, although the separate `interventionMass.waterSources` diagnostic records the source-map change. Future recharge does change. This is the intended distinction between water paint and source paint.

For `soil_recovery`, show soil, cultivation, and food in separate tiles. Start with the depleted soil map, run until population pressure retreats, and compare low-cultivation cells with watered cells. Painting cultivation is unavailable because the map is the derived function `P/(P + cultivationScale)`; paint population or soil when you need an intervention.

The civilization suggestions remain useful: add population far from food, reduce construction after infrastructure appears, and compare soil with population during overshoot. In chemotaxis, observe conserved totals before painting.

## Sharing and reset

Copy setup link writes a version 2 payload containing the preset, seed, resolution, boundary, complete model parameter set, and tile layout (tile IDs, ordered layers, visibility, opacity, and paint target). It excludes evolved fields, time, brush interventions, water-budget history, playback speed, and chart history. Opening a link initializes a fresh world from those reproducible inputs. Older version 1 links for the original non-ecology presets remain loadable with their original setup meaning; ecology links use version 2 because tile layout is part of the ecology view contract.

Reset uses the current setup and seed. Selecting a gallery card restores that preset's parameters while retaining the seed, resolution, and boundary. Invalid or out-of-date links load safe defaults and explain the fallback. Use 32×32 for speed, 64×64 for the default balance, or 128×128 for finer structures. Hardware can limit how fast requested playback advances; stability is preserved.
