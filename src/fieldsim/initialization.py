"""Deterministic cross-runtime initial conditions for civilization models."""

from __future__ import annotations

import numpy as np

from fieldsim.utils.constants import (
    FERTILITY,
    FOOD,
    INFRASTRUCTURE,
    POPULATION,
    SOIL,
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


def initialize_civilization(seed: int, n: int, bc_type: str = "neumann",
                            length: float = DOMAIN_LENGTH):
    """Return deterministic float64 ``P/F/K/I/S`` cell-centred arrays."""
    try:
        integer_n = int(n)
    except (TypeError, ValueError, OverflowError):
        integer_n = None
    if integer_n is None or integer_n != n or integer_n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}.")
    if bc_type not in ("neumann", "periodic"):
        raise ValueError(f"Unsupported boundary condition {bc_type!r}.")
    n = integer_n
    rng = Mulberry32(seed)
    pop_specs = _specs(rng, 6, (0.4, 1.2), (0.3, 0.8), length)
    food_specs = _specs(rng, 8, (0.5, 1.5), (0.5, 1.2), length)
    fertility_specs = _specs(rng, 8, (0.5, 1.5), (0.6, 1.5), length)
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
