#!/usr/bin/env python3
"""Generate browser parity fixtures from the float64 Python implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from fieldsim.catalog import get_preset
from fieldsim.field import Field
from fieldsim.initialization import initialize_civilization
from fieldsim.lagrangian import Lagrangian
from fieldsim.simulator import Simulator
from fieldsim.simulations import agriculture, chemotaxis_demo, civilization, ecology
from fieldsim.stability import rate_sum, stable_dt
from fieldsim.utils.constants import FERTILITY

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "web" / "tests" / "fixtures" / "reference.json"
DEFAULT_ECOLOGY_OUTPUT = ROOT / "web" / "tests" / "fixtures" / "ecology-reference.json"
N = 8
SEED = 7


def _config(model, boundary, preset_id=None):
    if model == "civilization":
        preset_id = preset_id or "collective_investment"
        return civilization.get_config(
            seed=SEED, n=N, total_time=1.0, bc_type=boundary,
            preset=preset_id, adaptive=False,
        ), preset_id
    if model == "agriculture":
        return agriculture.get_config(
            seed=SEED, n=N, total_time=1.0, bc_type=boundary
        ), "agriculture"
    return chemotaxis_demo.get_config(
        seed=SEED, n=N, total_time=1.0, bc_type=boundary
    ), "chemotaxis_demo"


def _build(model, boundary, preset_id=None):
    config, preset_id = _config(model, boundary, preset_id)
    initial = initialize_civilization(SEED, N, bc_type=boundary)
    fields = {}
    for name, definition in config.field_defs.items():
        field = Field(name=name, **definition)
        if name in initial:
            field.set_values(initial[name])
        fields[name] = field
    # Baseline fixtures carry K even for source-free chemotaxis, matching the
    # browser's always-present field set while keeping expected data relevant.
    if FERTILITY not in fields:
        fields[FERTILITY] = Field(
            FERTILITY, (N, N), dx=10.0 / N, bc_type=boundary,
            init_fn=lambda X, Y: initial[FERTILITY], is_dynamic=False,
        )

    lagrangian = Lagrangian()
    for term in config.lagrangian_terms:
        lagrangian.add_term(term)
    safe = stable_dt(
        fields, config.lagrangian_terms, config.flux_terms, config.sources,
        10.0 / N, safety=0.5,
    )
    current_rate = rate_sum(
        fields, config.lagrangian_terms, config.flux_terms, config.sources,
        10.0 / N,
    )
    adaptive_dt = min(0.1, 0.8 / current_rate) if current_rate > 0.0 else 0.1
    dt = min(0.05, safe)
    simulator = Simulator(
        fields, lagrangian, config.sources, config.flux_terms, dt=dt,
        check_every=None,
    )
    relevant = list(config.field_defs)
    if model != "civilization" and FERTILITY not in relevant:
        relevant.append(FERTILITY)
    setup = {
        "version": 1,
        "preset": preset_id,
        "seed": SEED,
        "n": N,
        "boundary": boundary,
        "parameters": get_preset(preset_id)["parameters"],
    }
    return simulator, relevant, setup, dt, current_rate, adaptive_dt


def _flatten(values):
    return np.asarray(values, dtype=np.float64).reshape(-1).tolist()


def generate():
    cases = []
    model_presets = (
        ("civilization", "collective_investment"),
        ("civilization", "overshoot"),
        ("agriculture", "agriculture"),
        ("chemotaxis", "chemotaxis_demo"),
    )
    for model, preset_id in model_presets:
        for boundary in ("neumann", "periodic"):
            for steps in (1, 20):
                simulator, relevant, setup, dt, current_rate, adaptive_dt = _build(
                    model, boundary, preset_id
                )
                initial = {
                    name: _flatten(simulator.get_state()[name]) for name in relevant
                }
                for _ in range(steps):
                    simulator.step()
                diagnostics = simulator.sync_diagnostics()
                assert all(
                    entry["cumulative_truncated_mass"] == 0.0
                    for entry in diagnostics.values()
                ), f"positivity floor activated in {preset_id}/{boundary}/{steps}"
                expected = {
                    name: _flatten(simulator.get_state()[name]) for name in relevant
                }
                cases.append({
                    "name": f"{preset_id}-{boundary}-{steps}",
                    "model": model,
                    "setup": setup,
                    "fields": initial,
                    "dt": dt,
                    "rate": current_rate,
                    "safe_dt": adaptive_dt,
                    "steps": steps,
                    "expected": expected,
                })
    return {"version": 1, "cases": cases}


def _build_ecology(boundary, preset_id, seed=SEED):
    config = ecology.get_config(
        seed=seed, n=N, total_time=1.0, bc_type=boundary,
        preset=preset_id, adaptive=False,
    )
    fields = {
        name: Field(name=name, **definition)
        for name, definition in config.field_defs.items()
    }
    lagrangian = Lagrangian()
    for term in config.lagrangian_terms:
        lagrangian.add_term(term)
    safe = stable_dt(
        fields, config.lagrangian_terms, config.flux_terms, config.sources,
        10.0 / N, safety=0.5,
    )
    current_rate = rate_sum(
        fields, config.lagrangian_terms, config.flux_terms, config.sources,
        10.0 / N,
    )
    adaptive_dt = min(0.1, 0.8 / current_rate) if current_rate > 0.0 else 0.1
    dt = min(0.05, safe)
    simulator = Simulator(
        fields, lagrangian, config.sources, config.flux_terms, dt=dt,
        check_every=None, derived_fields=config.derived_fields,
    )
    setup = {
        "version": 2,
        "preset": preset_id,
        "seed": seed,
        "n": N,
        "boundary": boundary,
        "parameters": get_preset(preset_id)["parameters"],
    }
    return simulator, setup, dt, current_rate, adaptive_dt


def generate_ecology():
    cases = []
    for seed in (0, SEED):
        for preset_id in ("water_settlement", "water_overuse", "soil_recovery"):
            for boundary in ("neumann", "periodic"):
                for steps in (1, 20):
                    simulator, setup, dt, current_rate, adaptive_dt = _build_ecology(
                        boundary, preset_id, seed
                    )
                    initial = {
                        name: _flatten(values)
                        for name, values in simulator.get_state().items()
                    }
                    for _ in range(steps):
                        simulator.step()
                    diagnostics = simulator.sync_diagnostics()
                    assert all(
                        entry["cumulative_truncated_mass"] == 0.0
                        for entry in diagnostics.values()
                    ), f"positivity floor activated in {preset_id}/{boundary}/{steps}"
                    expected = {
                        name: _flatten(values)
                        for name, values in simulator.get_state().items()
                    }
                    cases.append({
                        "name": f"{preset_id}-seed{seed}-{boundary}-{steps}",
                        "model": "ecology",
                        "setup": setup,
                        "fields": initial,
                        "dt": dt,
                        "rate": current_rate,
                        "safe_dt": adaptive_dt,
                        "steps": steps,
                        "expected": expected,
                    })
    return {"version": 1, "cases": cases}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ecology-output", type=Path, default=DEFAULT_ECOLOGY_OUTPUT)
    args = parser.parse_args(argv)
    payload = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {args.output}")
    ecology_payload = generate_ecology()
    args.ecology_output.parent.mkdir(parents=True, exist_ok=True)
    args.ecology_output.write_text(
        json.dumps(ecology_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(ecology_payload['cases'])} cases to {args.ecology_output}")


if __name__ == "__main__":
    main()
