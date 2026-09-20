import argparse
import sys

from fieldsim.simulation_runner import SimulationRunner
from fieldsim.utils.constants import FOOD, INFRASTRUCTURE, POPULATION, SOIL, WATER

#: Registry of bundled simulations: name -> (get_config, field names to plot).
_SIMULATIONS = {
    "agriculture": ("fieldsim.simulations.agriculture", [POPULATION, FOOD]),
    "chemotaxis_demo": ("fieldsim.simulations.chemotaxis_demo", [POPULATION, FOOD]),
    "civilization": (
        "fieldsim.simulations.civilization",
        [POPULATION, FOOD, INFRASTRUCTURE, SOIL],
    ),
    "ecology": (
        "fieldsim.simulations.ecology",
        [POPULATION, FOOD, WATER, SOIL],
    ),
}


def _get_config_fn(sim_name):
    import importlib

    module_name, _ = _SIMULATIONS[sim_name]
    module = importlib.import_module(module_name)
    return module.get_config


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fieldsim",
        description="Run a field-sim PDE simulation headlessly or with an animation.",
    )
    parser.add_argument(
        "--sim", choices=sorted(_SIMULATIONS), default="agriculture",
        help="Which bundled simulation to run (default: agriculture).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for initial-condition generation.")
    parser.add_argument(
        "--resolution", "--n", dest="resolution", type=int, default=None,
        help="Number of cells along each domain edge.",
    )
    parser.add_argument(
        "--boundary", choices=("neumann", "periodic"), default="neumann",
        help="Boundary condition for diffusion and transport.",
    )
    parser.add_argument(
        "--preset", default=None,
        help="Catalog preset for civilization or ecology.",
    )
    parser.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="Override a civilization or ecology parameter; may be repeated.",
    )
    parser.add_argument(
        "--years", type=float, default=None,
        help="Override the simulation's total_time (in years); default is the config's own default.",
    )
    parser.add_argument(
        "--no-anim", action="store_true",
        help="Run the simulation only; print a summary instead of animating.",
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="Save the animation to PATH (.gif via PillowWriter, .mp4 via ffmpeg if available) "
             "instead of showing it interactively.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="Maximum number of history snapshots to record (default: 150).",
    )
    return parser


def _print_summary(runner):
    print(f"simulation: {runner.config.name}")
    print(f"steps: {runner.steps}")
    print(f"dt: {runner.simulator.dt:.6g}")
    print(f"simulated time: {runner.simulator.time:.6g}")
    print(f"frames recorded: {len(runner.history)} (stride={runner.stride})")

    # Read the final state straight from the simulator's own device arrays
    # (full precision) rather than from ``runner.history``, whose frames are
    # downcast to float32 for animation/plotting and would silently distort
    # a mass summary that is meant to be a precise diagnostic.
    final = runner.simulator.get_state()
    print("final total mass per field:")
    for name, values in final.items():
        mass = float(values.sum()) * runner.dx ** 2
        print(f"  {name}: {mass:.6g}")

    print("positivity-floor truncation diagnostics:")
    for name, diag in (runner.diagnostics or {}).items():
        print(
            f"  {name}: last_truncated_mass={diag['last_truncated_mass']:.6g}, "
            f"cumulative_truncated_mass={diag['cumulative_truncated_mass']:.6g}"
        )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Force a headless backend *before* matplotlib.pyplot is imported anywhere
    # in this process, whenever we won't (or can't) show an interactive window.
    if args.save is not None or args.no_anim:
        import matplotlib
        matplotlib.use("Agg")

    get_config = _get_config_fn(args.sim)
    _, field_names = _SIMULATIONS[args.sim]

    config_kwargs = {"seed": args.seed, "bc_type": args.boundary}
    if args.resolution is not None:
        config_kwargs["n"] = args.resolution
    if args.years is not None:
        config_kwargs["total_time"] = args.years
    if args.sim in ("civilization", "ecology"):
        if args.preset is not None:
            config_kwargs["preset"] = args.preset
        parameters = {}
        for item in args.param:
            try:
                key, raw_value = item.split("=", 1)
                parameters[key] = float(raw_value)
            except (ValueError, TypeError):
                parser.error(f"--param must be KEY=VALUE, got {item!r}")
        if parameters:
            config_kwargs["parameters"] = parameters
    elif args.preset is not None or args.param:
        parser.error("--preset and --param are only valid with --sim civilization or ecology")
    cfg = get_config(**config_kwargs)

    runner = SimulationRunner(cfg, max_frames=args.max_frames)
    runner.run()

    if args.no_anim:
        _print_summary(runner)
        return 0

    runner.animate(
        field_names=field_names,
        absolute=True,
        split=True,
        split_rows=1,
        split_cols=len(field_names),
        save_path=args.save,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
