"""Phase 1 import-safety test.

Importing any fieldsim module must be a pure side-effect-free operation:
no consumption of the global numpy RNG state, no matplotlib figures opened
(indeed, pyplot itself must never be imported), and no simulation run.

This has to run in a *fresh* interpreter (via subprocess): under pytest's
normal collection, other test modules (e.g. tests/test_configs.py) already
import fieldsim submodules at module scope before this test body ever runs,
so by the time ``importlib.import_module`` executes here almost every
fieldsim module is already in ``sys.modules`` and the import is a no-op --
silently making the RNG/figure assertions vacuous. Spawning a child
interpreter that has never touched fieldsim guarantees the imports below
actually do something.
"""

import subprocess
import sys

_CHILD_SCRIPT = r"""
import importlib
import pkgutil
import sys

import numpy as np


def _fieldsim_module_names():
    import fieldsim

    names = [fieldsim.__name__]
    for module_info in pkgutil.walk_packages(fieldsim.__path__, prefix=f"{fieldsim.__name__}."):
        # fieldsim.__main__ is the `python -m fieldsim` CLI entry point: by
        # design it unconditionally invokes main() when loaded (that is the
        # whole point of __main__.py), so it is not a safe-to-import library
        # module and is excluded from this scan.
        if module_info.name == "fieldsim.__main__":
            continue
        names.append(module_info.name)
    return names


# Third-party libraries (notably jax, on its first import in a process)
# consume global numpy RNG state as an implementation detail unrelated
# to fieldsim. Warm those imports up first so the snapshot below isolates
# fieldsim's *own* module-level behavior, which is what this test checks.
importlib.import_module("jax")

module_names = _fieldsim_module_names()

before_state = np.random.get_state()

for module_name in module_names:
    importlib.import_module(module_name)

after_state = np.random.get_state()

# Compare the numeric payload of the Mersenne Twister state (index 1);
# element 0 is the algorithm name, element 2 is the position pointer.
assert np.array_equal(before_state[1], after_state[1])
assert before_state[2] == after_state[2]

# Stronger than checking for open figures: fieldsim modules must not even
# import matplotlib.pyplot at module scope (it is only ever imported inside
# functions that actually need to plot).
assert "matplotlib.pyplot" not in sys.modules

print("IMPORT_SAFETY_OK", len(module_names))
"""


def test_importing_fieldsim_does_not_consume_global_rng_or_open_figures():
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT_SAFETY_OK" in result.stdout, result.stdout + result.stderr
    # Sanity: the walk actually found a non-trivial number of submodules,
    # so this isn't accidentally passing on an empty package.
    module_count = int(result.stdout.strip().rsplit(" ", 1)[-1])
    assert module_count >= 10
