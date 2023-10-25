#!/usr/bin/env -S julia --project=.

# Demo for using SMAC3 to optimize hyperparameters and algorithm configurations
# directly from within Julia. Note that this variant is limited to a sequential execution.

# Ensure that Python is installed with the packages 
# `julia`, `smac` and `ConfigSpace` and this version is used by Julia's `PyCall`.

using PyCall

@pyimport smac
@pyimport ConfigSpace

# the function whose parameters shall be tuned
include("julia-function-to-tune.jl")

# Define the configuration space, i.e., the parameters to tune with their domains
# short form:
config_space1 = ConfigSpace.ConfigurationSpace(Dict(
    "x" => (0.1, 10.0), 
    "y" => (1, 3), 
    "z" => ["opt1", "opt2"],
    ))
# long form, advanced configuration aspects possible:
config_space2 = ConfigSpace.ConfigurationSpace()
config_space2.add_hyperparameters([
    ConfigSpace.UniformFloatHyperparameter("x", 0.1, 10.0),
    ConfigSpace.UniformIntegerHyperparameter("y", 1, 3),
    ConfigSpace.CategoricalHyperparameter("z", ["opt1", "opt2"]),
])

py"""
from julia import Main

# Python wrapper for the function to optimize
def py_f(config, seed: int=0) -> float:
    res = Main.f(config["x"], config["y"], config["z"])
    print("x: ", config["x"], ", y: ", config["y"], ", z: ", config["z"], ", res: ", res)
    return res
"""

# Scenario object specifying the optimization environment
scenario = smac.Scenario(config_space2, deterministic=true, n_trials=200)

# Use SMAC to find the best configuration/parameters
facade = smac.HyperparameterOptimizationFacade(scenario, py"py_f", overwrite=true)
incumbent = facade.optimize()
@show incumbent incumbent.get("x"), incumbent.get("y"), incumbent.get("z")


