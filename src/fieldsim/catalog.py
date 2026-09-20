"""Access to the shared experiment and parameter catalog."""

from __future__ import annotations

import json
from importlib.resources import files


def load_catalog():
    return json.loads(files("fieldsim").joinpath("catalog.json").read_text(encoding="utf-8"))


def get_preset(preset_id):
    for preset in load_catalog()["presets"]:
        if preset["id"] == preset_id:
            return preset
    raise ValueError(f"Unknown preset {preset_id!r}.")


def parameter_defaults():
    return {entry["key"]: float(entry["default"]) for entry in load_catalog()["parameters"]}

