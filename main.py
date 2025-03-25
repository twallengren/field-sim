from simulations.agriculture import get_config
from simulation_runner import SimulationRunner
from utils.constants import POPULATION, FOOD, FERTILITY, INDUSTRY

cfg = get_config()
runner = SimulationRunner(cfg)
runner.run()
runner.animate(
    field_names=[POPULATION, FERTILITY, INDUSTRY, FOOD],
    absolute=True,
    split=True,
    split_rows=2,
    split_cols=2)

if __name__ == "__main__":
    pass