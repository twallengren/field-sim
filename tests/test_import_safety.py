"""Phase 1 import-safety test.

Importing any fieldsim module must be a pure side-effect-free operation:
no consumption of the global numpy RNG state, no matplotlib figures opened,
and no simulation run.
"""

import importlib
import pkgutil

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


def test_importing_fieldsim_does_not_consume_global_rng_or_open_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    assert plt.get_fignums() == []
