"""Shared pytest fixtures and small numerical helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def x64():
    """Enable JAX float64 for the duration of one test, then restore.

    Several numerical-property tests assert machine-precision statements
    (conservation to 1e-12, exact eigenmode decay to 1e-10) that are simply not
    expressible in float32.  Tests that assert *structural* properties (e.g.
    donor-cell positivity) deliberately do not use this fixture, so they run in
    the library's default float32 precision.
    """
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def cell_centers(n, dx):
    """Cell-centred 1D coordinates ``x_i = (i + 1/2) dx`` for ``n`` cells."""
    return (np.arange(n) + 0.5) * dx


def cell_center_grid(n, dx):
    """Cell-centred meshgrid ``(X, Y)`` with ``indexing="xy"``, shape (n, n)."""
    coords = cell_centers(n, dx)
    return np.meshgrid(coords, coords, indexing="xy")


def gaussian(X, Y, x0, y0, sigma, amp=1.0):
    """Isotropic Gaussian bump evaluated on a numpy meshgrid."""
    return amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2))


def total_mass(values, dx):
    """Discrete integral ``sum(phi) * dx^2``."""
    return float(jnp.sum(values) * dx ** 2)


def x_centroid(values, dx):
    """Mass-weighted mean x coordinate on a cell-centred grid."""
    values = np.asarray(values, dtype=np.float64)
    ny, nx = values.shape
    X, _ = cell_center_grid(nx, dx)
    assert values.shape == X.shape
    return float(np.sum(values * X) / np.sum(values))
