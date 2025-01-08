from ConfigSpace import Configuration, ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario
import random

config_space = ConfigurationSpace({
        "x": (1, 3), 
        "y": (1, 2),
    })

instances = ["myinst.txt"]
features = {fn: [i] for (i, fn) in enumerate(instances)}

scenario = Scenario(config_space, deterministic=False, 
                    instances=instances, instance_features=features, 
                    n_trials=200)

def f(config: Configuration, instance: str, seed: int) -> float:
    x = int(config["x"]); y = int(config["y"])
    print(f'f({instance}, {seed}, {x}, {y})', end=" -> ")
    res = x - y + random.random()
    print(res)
    return res
    
smac = AlgorithmConfigurationFacade(scenario, f, overwrite=True)

incumbent = smac.optimize()
print("Optimized configuration: ", incumbent)