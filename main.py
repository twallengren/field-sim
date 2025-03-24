from simulations.population_food import get_config
from simulation_runner import SimulationRunner

cfg = get_config()
runner = SimulationRunner(cfg)
runner.run()
runner.animate(
    field_names=["pop", "food", "res", "infra"],
    absolute=False,
    split=True,
    split_rows=2,
    split_cols=2)

if __name__ == "__main__":
    pass