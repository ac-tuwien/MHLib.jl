# Algorithm/Parameter Tuning with SMAC3 in Conjunction with Julia

[SMAC3](https://automl.github.io/SMAC3) is a powerful tool for automated algorithm 
configuration and parameter tuning. It is based on Bayesian optimization and utilizes
racing mechanisms and random forests.

This directory contains an example with three different variants on how SMAC3 can 
specifically be used for tuning parameters of algorithms implemented in Julia. 
This can be achieved in different ways.

The main script is `tuning.py`, and the variant to be used can be provided as
command line argument:


1)  This variant uses the Python->Julia interface provided by the Python package `pyjulia`.
    The julia function `f` can here be called from SMAC3 via a simple
    Python wrapper function. A limitation of this approach is that no parallelization
    via multithreading or multiprocessing is possible.

2)  Here, for each call of the function to tune, a new Julia process is started. While this
    approach allows for parallelization, also using a compute cluster, it can be quite
    inefficient when the function to tune is comparably fast w.r.t. Julia's startup and
    pre-compile time.

3) Here, the main Python script spawns a fixed number of Julia "server" subprocesses
    realized by `julia_server.jl`, which accept function calls, perform them, 
    and return the result. The communication is done via pipes, respectively stdin/stdout. 
    In this way multiprocessing is utilized, but the overhead of starting 
    a new Julia process for each function call is avoided.

Note that the demo needs to be started from the `Tuning` subdirectory.

Ensure that Python is installed with the packages `julia`, `smac` and `ConfigSpace`.


