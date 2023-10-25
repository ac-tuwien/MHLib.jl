#!/usr/bin/env python3

"""
Demo for using SMAC3 to optimize hyperparameters of a Julia function.

Ensure that Python is installed with the packages 
`julia`, `smac` and `ConfigSpace` and this version is used by Julia's `PyCall`

In this variant `julia_server.py` is used to start a number of parallel Julia subprocesses,
in which the function to optimize is repeatedly evaluated.
This approach can be useful if Julia's startup times are considerably high in comparison
to one function evaluation.
"""

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

from dask.distributed import Client, LocalCluster
import julia_server


config_space = ConfigurationSpace({
        "x": (0.1, 10.0), 
        "y": (1, 3), 
        "z":["opt1", "opt2"],
    })


if __name__ == "__main__":

    # Scenario object specifying the optimization environment
    scenario = Scenario(config_space, deterministic=True, n_trials=100)

    # for parallel execution in a number of Julia subprocesses:
    # note that the number of threads per worker must be set to 1
    cluster = LocalCluster(threads_per_worker=1, n_workers=4)
    client = Client(address=cluster)
    smac = HyperparameterOptimizationFacade(scenario, julia_server.f, overwrite=True,
                                            dask_client=client)
    
    incumbent = smac.optimize()
    print("Optimized configuration: ", incumbent)