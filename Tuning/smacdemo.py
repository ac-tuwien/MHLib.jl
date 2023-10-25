#!/usr/bin/env python3
# Demo for using SMAC3 to optimize hyperparameters of a Julia function.

# Ensure that Python is installed with the packages 
# `julia`, `smac` and `ConfigSpace` and this version is used by Julia's `PyCall`


from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

from julia import Main
Main.include("julia-function-to-tune.jl")

# exemplary wrapper for Julia function to optimize
def f(config: Configuration, seed: int=0) -> float:
    cmd = f'f({config["x"]}, {config["y"]}, \"{config["z"]}\")\n'
    print(cmd)
    res = Main.f(config["x"], config["y"], config["z"])
    print("res: ", res)
    return res


config_space = ConfigurationSpace({
        "x": (0.1, 10.0), 
        "y": (1, 3), 
        "z":["opt1", "opt2"],
    })


if __name__ == "__main__":

    # Scenario object specifying the optimization environment
    scenario = Scenario(config_space, deterministic=True, n_trials=100)

    # execute function sequentially in a single thread
    smac = HyperparameterOptimizationFacade(scenario, f, overwrite=True)
    
    incumbent = smac.optimize()
    print("Optimized configuration: ", incumbent)