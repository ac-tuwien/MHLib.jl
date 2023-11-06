#!/usr/bin/env python3
# Demo for using SMAC3 to optimize the configuration of a Julia algorithm.

# Ensure that Python is installed with the packages 
# `julia`, `smac` and `ConfigSpace` and this version is used by Julia's `PyCall`


import ConfigSpace
from ConfigSpace import Configuration, ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario

from julia import Main
Main.include("julia-function-to-tune.jl")

# exemplary wrapper for Julia function to optimize
def f(config: Configuration, instance, seed: int=0) -> float:
    print(f'f({instance}, {config["x"]}, {config["y"]}, \"{config["z"]}\")')
    res = Main.f(instance, config["x"], config["y"], config["z"])
    print("-> ", res)
    return res

# simple way to specify a configuration space
config_space = ConfigurationSpace({
        "x": (0.1, 4.0), 
        "y": (1, 3), 
        "z":["opt1", "opt2"],
    })

# alternative long form, advanced configuration aspects possible:
config_space2 = ConfigurationSpace()
config_space2.add_hyperparameters([
    ConfigSpace.UniformFloatHyperparameter("x", 0.1, 4.0),
    ConfigSpace.UniformIntegerHyperparameter("y", 1, 3),
    ConfigSpace.CategoricalHyperparameter("z", ["opt1", "opt2"]),
])

# names of problem instances to be used for tuning
instances = [f"a{i}" for i in range(3)]

# features of the problem instances
# in the simplest case just the index, or otherwise some more relevant features
features = {f"a{i}": [i] for i in range(3)}

if __name__ == "__main__":

    # Scenario object specifying the optimization environment
    scenario = Scenario(config_space2, deterministic=False, 
                        instances=instances, instance_features=features, 
                        n_trials=200)

    # execute function sequentially in a single thread
    smac = AlgorithmConfigurationFacade(scenario, f, overwrite=True)
    
    incumbent = smac.optimize()
    print("Optimized configuration: ", incumbent)