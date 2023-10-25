# Algorithm/Parameter Tuning with SMAC3

[SMAC3](https://automl.github.io/SMAC3) is a powerful tool for automated algorithm 
configuration and parameter tuning. It is based on Bayesian optimization and utilizes
racing mechanisms and random forests.

This directory contains examples how SMAC3 can specifically be used for tuning 
parameters of functions/algorithms implemented in Julia. 
This can be achieved in different ways:

- `smacdemo-with-julia-processes`:
    Here, for each call of the function to tune, a new Julia process is started. While this
    approach allows for parallelization, also using a compute cluster, it can be quite
    inefficient when the function to tune is comparably fast w.r.t. Julia's startup and
    pre-compile time.

- `smacdemo.jl`:
    Here we utilize Julia's possibility to interface with Python via PyCall. This allows
    that SMAC3 can directly call the Julia function to tune. However, this approach is
    restricted to pure sequential function evaluations.

- `smacdemo.py`
    This is similar to the above `smacdemo.jl`, but here the main code for SMAC is 
    written as a Phyton script.

- `smacdemo-with-julia-server.py`:
    Here, the main Python script spawns a number of Julia "server" subprocesses which 
    accept function calls, perform them and return the result. The communication is
    done via simple stdin/stdout. In this way multiprocessing is utilized, but
    the overhead of starting a new Julia process for each function call is avoided.

Note that all demos need to be started from the `Tuning` subdirectory.

Ensure that Python is installed with the packages `julia`, `smac` and `ConfigSpace` 
and this version is used by Julia's `PyCall`.
For the Julia code, the local environment in `Tuning` need to be used.


