from simulations.population_food import get_config
from simulation_runner import SimulationRunner

cfg = get_config()
runner = SimulationRunner(cfg)
runner.run()
runner.animate(field_names=["pop", "food"], absolute=True, split=True)

if __name__ == "__main__":
    pass