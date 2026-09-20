# Development and extension

## Verification

After installing dependencies, run from the repository root:

```sh
.venv/bin/pytest -q
.venv/bin/python -m pyflakes src tests scripts
.venv/bin/python scripts/generate_reference.py
cd web
npm ci
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

Reference generation enables JAX float64. Generate expectations from Python, never from the browser being tested. Browser parity covers ecology transport, coupled sources, derived cultivation, water accounting, and both boundary modes. Browser acceptance uses the production build under `/field-sim/`.

## Add a model or preset

`src/fieldsim/catalog.json` is shared by Python and Vite. It defines preset order, model identity, descriptions, fields, defaults, parameter definitions, and browser ranges; the TypeScript catalog adapter places ecology presets first and assigns the ecology control groups used by the UI. Keep stable preset and parameter identifiers because they appear in setup links and fixtures.

A new ecology coefficient requires coordinated changes to the catalog, Python source terms and rate bound, browser equations and rate bound, initialization or parity fixtures when relevant, tests, and model/numerics documentation. New Python operators must declare `max_rate`; conservative transport uses face fluxes. Keep water-stock accounting beside the equations whenever a source or withdrawal changes the ledger.

Python retains general composition through `Field`, `LagrangianTerm`, `FluxTerm`, `SourceTerm`, and `SimulationConfig`. The browser supports registered models with explicit equations. `shared/IMPLEMENTATION.md` records the cross-runtime array, initializer, worker, tile, sharing, and ownership contracts.

## Browser views and setup links

Tile layout is view state, not simulation state. A tile has a stable ID, an ordered list of unique field layers, visibility and opacity per layer, and one paint target from those layers. There is no fixed two-view limit. A model's field descriptor marks fields as dynamic, static, or derived; derived fields may be rendered and selected as a tile target, but a paint attempt is rejected because they are read-only.

The renderer owns one scale per field key and one history series per unique visible field. A tile change must preserve those semantics when it adds, removes, reorders, or overlays layers. Setup serialization includes model parameters and tile layout in version 2 and excludes arrays, evolved time, interventions, and history. Version 1 links for legacy presets remain valid; do not reinterpret their parameter payloads.

## GitHub Pages

Target URL: `https://twallengren.github.io/field-sim/`. Vite uses `/field-sim/`, including worker assets. Setup links use fragments and require no server routing.

The Actions workflow verifies Python, browser numerics, the production build, and browser interactions before publishing. Pull requests only run checks. Pushes and manual runs on `main` deploy after successful verification.

In **Settings → Pages → Build and deployment**, select **GitHub Actions**. Any required `github-pages` environment approval is handled in GitHub. The deployment job requires Pages write and OIDC permissions, without application secrets.

For another repository name or custom domain, update Vite's base and the browser-test path. The runtime needs no API keys, server, database, or environment variables.
