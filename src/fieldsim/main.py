import argparse

from fieldsim.simulations.agriculture import get_config
from fieldsim.simulation_runner import SimulationRunner
from fieldsim.utils.constants import POPULATION, FOOD


def main():
    parser = argparse.ArgumentParser(
        prog="fieldsim",
        description="Run a field-sim PDE simulation (agriculture population/food model).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for initial-condition generation.")
    args = parser.parse_args()

    cfg = get_config(seed=args.seed)
    runner = SimulationRunner(cfg)
    runner.run()
    runner.animate(
        field_names=[POPULATION, FOOD],
        absolute=True,
        split=True,
        split_rows=1,
        split_cols=2)


if __name__ == "__main__":
    main()
