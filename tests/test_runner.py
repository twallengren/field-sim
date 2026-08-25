"""Phase 3: bounded history, streaming extrema, and headless CLI tests."""

import math
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # Must precede any pyplot import anywhere in this process.

import numpy as np
import pytest

from fieldsim.simulation_runner import SimulationRunner
from fieldsim.simulations import chemotaxis_demo
from fieldsim.utils.constants import FOOD, POPULATION


def _tiny_multistep_runner(max_frames=10):
    """A grid small enough to be fast, but with enough steps to exercise stride > 1."""
    cfg = chemotaxis_demo.get_config(seed=0, n=32, total_time=3.0)
    runner = SimulationRunner(cfg, max_frames=max_frames)
    runner.run()
    return runner


def test_history_stride_and_dtype():
    max_frames = 10
    runner = _tiny_multistep_runner(max_frames=max_frames)

    assert runner.steps > max_frames, "test needs steps > max_frames to exercise striding"
    assert runner.stride == max(1, math.ceil(runner.steps / max_frames))
    assert len(runner.history) <= max_frames + 1
    assert len(runner.history) >= 2  # at least the initial and final snapshots

    for frame in runner.history:
        for name in (POPULATION, FOOD):
            values = frame[name]
            assert isinstance(values, np.ndarray)
            assert values.dtype == np.float32


def test_streaming_min_max_matches_direct():
    runner = _tiny_multistep_runner(max_frames=10)

    for name in (POPULATION, FOOD):
        stacked = np.stack([frame[name] for frame in runner.history])
        direct_min, direct_max = float(stacked.min()), float(stacked.max())
        streamed_min, streamed_max = runner.value_range(name)
        assert streamed_min == pytest.approx(direct_min)
        assert streamed_max == pytest.approx(direct_max)


def test_single_frame_animate_does_not_raise(tmp_path):
    """max_frames=1 (and a directly-constructed single-frame history) must not
    raise a ZeroDivisionError in the progress-percentage computation."""
    cfg = chemotaxis_demo.get_config(seed=0, n=8, total_time=1e-3)
    runner = SimulationRunner(cfg, max_frames=1)
    runner.run()

    # Force the degenerate single-snapshot case directly, regardless of how
    # many steps the tiny config above actually took.
    runner.history = runner.history[:1]

    save_path = tmp_path / "single_frame.gif"
    runner.animate(
        field_names=[POPULATION, FOOD],
        absolute=True,
        split=True,
        save_path=str(save_path),
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_animate_supports_more_fields_than_default_cmaps(tmp_path):
    """cmap_list cycling (modulo) must not IndexError beyond len(_DEFAULT_CMAPS)."""
    cfg = chemotaxis_demo.get_config(seed=0, n=8, total_time=1e-3)
    runner = SimulationRunner(cfg, max_frames=2)
    runner.run()

    # Reuse the two real fields repeatedly to synthesize > 6 "fields" worth of
    # plots without needing a bespoke many-field config.
    many_names = [POPULATION, FOOD] * 4
    for frame in runner.history:
        for i, name in enumerate(many_names):
            frame[f"dup{i}"] = frame[name]
    many_names = [f"dup{i}" for i in range(len(many_names))]

    save_path = tmp_path / "many_fields.gif"
    runner.animate(field_names=many_names, split=True, save_path=str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


# --------------------------------------------------------------------------
# CLI end-to-end (subprocess) tests
# --------------------------------------------------------------------------

def test_cli_no_anim_headless(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    result = subprocess.run(
        [sys.executable, "-m", "fieldsim",
         "--sim", "chemotaxis_demo", "--seed", "0", "--no-anim", "--years", "0.3"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "simulation: Chemotaxis" in result.stdout
    assert "steps:" in result.stdout
    assert "final total mass per field:" in result.stdout
    assert "positivity-floor truncation diagnostics:" in result.stdout


def test_cli_save_gif_headless(monkeypatch, tmp_path):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    out_path = tmp_path / "out.gif"
    result = subprocess.run(
        [sys.executable, "-m", "fieldsim",
         "--sim", "chemotaxis_demo", "--seed", "0", "--years", "0.3",
         "--save", str(out_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    assert out_path.stat().st_size > 0
