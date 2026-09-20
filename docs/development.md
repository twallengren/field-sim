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

Reference generation enables JAX float64. Generate expectations from Python, never from the browser being tested. CI regenerates fixtures before browser parity tests. Browser acceptance uses the production build under `/field-sim/`.

## Add a demo or model

`src/fieldsim/catalog.json` controls gallery ordering, descriptions, model identity, defaults, highlighted fields, and ranges. It is packaged with Python and imported by Vite. Keep stable identifiers for shared links.

A new parameter requires coordinated changes to the catalog, TypeScript contract, both implementations, stability bounds, docs, and parity cases. New Python operators must declare `max_rate`. Conservative transport uses face fluxes.

Python retains general composition through `Field`, `LagrangianTerm`, `FluxTerm`, `SourceTerm`, and `SimulationConfig`. The browser supports registered models with explicit equations. `shared/IMPLEMENTATION.md` records array layout, initialization, worker messages, and ownership contracts.

## Specialist agent workflow

Use at most three working subagents plus a primary technical lead. Every task specifies its specialist role, allowed files, read-only dependencies, frozen interfaces, validation, and handoff criteria.

| Assignment | Model | Role |
| --- | --- | --- |
| Python engine | Sol / high | Senior scientific Python developer; expert in JAX, conservative PDEs, and stability |
| Browser engine | Sol / high | Senior TypeScript systems developer; expert in browser numerics and Web Workers |
| UI components | Luna / medium | Senior frontend developer specializing in accessible scientific interfaces |
| App integration | Sol / high | Senior frontend engineer; expert in Canvas and asynchronous state |
| Documentation/release | Luna / medium | Senior developer specializing in documentation and reproducible CI/CD |
| Browser acceptance | Luna / medium | Senior QA engineer specializing in browser interaction and accessibility |

The primary is the technical lead and scientific-software architect. It owns shared contracts, manifests, lockfiles, preset approval, scientific review, and release acceptance. If session capacity prevents an assignment, the primary completes it and reports the deviation rather than attributing it to a specialist.

Build engines and presentational UI in parallel, then transfer UI ownership to integration. Documentation and acceptance follow the actual interfaces. No recursive delegation or concurrent edits to another agent's files. Each handoff reports changes, test results, unresolved issues, and requested contract changes.

## GitHub Pages

Target URL: `https://twallengren.github.io/field-sim/`. Vite uses `/field-sim/`, including worker assets. Setup links use fragments and require no server routing.

The Actions workflow verifies Python, browser numerics, the production build, and browser interactions before publishing. Pull requests only run checks. Pushes and manual runs on `main` deploy after successful verification.

In **Settings → Pages → Build and deployment**, select **GitHub Actions**. Any required `github-pages` environment approval is handled in GitHub. The deployment job requires Pages write and OIDC permissions, without application secrets.

For another repository name or custom domain, update Vite's base and the browser-test path. The runtime needs no API keys, server, database, or environment variables.
