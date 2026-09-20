"""Deterministic cross-runtime initial conditions for civilization models."""

from __future__ import annotations

import numpy as np

from fieldsim.utils.constants import (
    CULTIVATION,
    FERTILITY,
    FOOD,
    INFRASTRUCTURE,
    POPULATION,
    SOIL,
    WATER,
    WATER_SOURCES,
)

DOMAIN_LENGTH = 10.0


class Mulberry32:
    """The uint32 Mulberry32 stream used by the browser implementation."""

    def __init__(self, seed: int):
        self.state = int(seed) & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


def _specs(rng, count, amp_range, sigma_range, length):
    specs = []
    for _ in range(count):
        x = length * rng.random()
        y = length * rng.random()
        amplitude = amp_range[0] + (amp_range[1] - amp_range[0]) * rng.random()
        sigma = sigma_range[0] + (sigma_range[1] - sigma_range[0]) * rng.random()
        rng.random()  # Reserved so future additions do not perturb later fields.
        specs.append((x, y, amplitude, sigma))
    return specs


def _render(X, Y, specs, floor, length, periodic):
    result = np.full(X.shape, floor, dtype=np.float64)
    for x0, y0, amplitude, sigma in specs:
        dx = np.abs(X - x0)
        dy = np.abs(Y - y0)
        if periodic:
            dx = np.minimum(dx, length - dx)
            dy = np.minimum(dy, length - dy)
        result += amplitude * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    return result


def _validate_grid(n, bc_type):
    try:
        integer_n = int(n)
    except (TypeError, ValueError, OverflowError):
        integer_n = None
    if integer_n is None or integer_n != n or integer_n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}.")
    if bc_type not in ("neumann", "periodic"):
        raise ValueError(f"Unsupported boundary condition {bc_type!r}.")
    return integer_n


def _landscape_specs(rng, length):
    """Draw the legacy P/F/K specs without changing their RNG sequence."""
    return (
        _specs(rng, 6, (0.4, 1.2), (0.3, 0.8), length),
        _specs(rng, 8, (0.5, 1.5), (0.5, 1.2), length),
        _specs(rng, 8, (0.5, 1.5), (0.6, 1.5), length),
    )


def initialize_civilization(seed: int, n: int, bc_type: str = "neumann",
                            length: float = DOMAIN_LENGTH):
    """Return deterministic float64 ``P/F/K/I/S`` cell-centred arrays."""
    n = _validate_grid(n, bc_type)
    rng = Mulberry32(seed)
    pop_specs, food_specs, fertility_specs = _landscape_specs(rng, length)
    coords = (np.arange(n, dtype=np.float64) + 0.5) * (length / n)
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    periodic = bc_type == "periodic"
    return {
        POPULATION: _render(X, Y, pop_specs, 0.05, length, periodic),
        FOOD: _render(X, Y, food_specs, 0.2, length, periodic),
        FERTILITY: _render(X, Y, fertility_specs, 0.25, length, periodic),
        INFRASTRUCTURE: np.zeros((n, n), dtype=np.float64),
        SOIL: np.ones((n, n), dtype=np.float64),
    }


def initialize_ecology(seed: int, n: int, bc_type: str = "neumann",
                       length: float = DOMAIN_LENGTH, source_capacity: float = 2.0,
                       cultivation_scale: float = 0.6):
    """Return deterministic ecology fields, including static water sources.

    The P/F/K draws are byte-for-byte compatible with the civilization
    initializer. Five additional clipped Gaussian patches define the static
    source-quality map Q. The initial local water stock is ``source_capacity*Q``.
    """
    n = _validate_grid(n, bc_type)
    if not np.isfinite(source_capacity) or source_capacity <= 0.0:
        raise ValueError("source_capacity must be finite and positive.")
    if not np.isfinite(cultivation_scale) or cultivation_scale <= 0.0:
        raise ValueError("cultivation_scale must be finite and positive.")
    rng = Mulberry32(seed)
    pop_specs, food_specs, fertility_specs = _landscape_specs(rng, length)
    source_specs = _specs(rng, 5, (0.55, 1.0), (0.45, 1.0), length)
    coords = (np.arange(n, dtype=np.float64) + 0.5) * (length / n)
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    periodic = bc_type == "periodic"
    population = _render(X, Y, pop_specs, 0.05, length, periodic)
    water_sources = np.clip(
        _render(X, Y, source_specs, 0.0, length, periodic), 0.0, 1.0
    )
    return {
        POPULATION: population,
        FOOD: _render(X, Y, food_specs, 0.2, length, periodic),
        WATER: source_capacity * water_sources,
        SOIL: np.ones((n, n), dtype=np.float64),
        FERTILITY: _render(X, Y, fertility_specs, 0.25, length, periodic),
        WATER_SOURCES: water_sources,
        CULTIVATION: population / (population + cultivation_scale),
    }
