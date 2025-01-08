# Algorithm/Parameter Tuning of Julia Programs

In the examples provided in the `tuning`directory, file `tuning.julia-function-to-tune.jl`
contains a simple example of a function whose parameters shall be tuned.

## Tuning with irace

Ensure to have [irace](https://github.com/MLopez-Ibanez/irace) installed on your system.

The straight-forward to to use irace with Julia is illustrated in the 
`tuning/irace-classical` directory. All the configuration files are mostly provided by 
`irace --init` and just minimally adapted to fit our example. Most importantly, 
`target_runner` calls here Julia with the script `main.jl`, in which the provided 
arguments are parsed and the test function is called. The result is then printed to stdout.

This direct approach is less efficient when the startup time for Julia and the program to 
be tuned is high in comparison to the actual algorithm evaluation. 
For this case, directory `tuning/irace-with-julia-server` provides a different solution.
`julia-server` is a simple standalone server program, which first needs to be started. 
It listens on a socket for incoming function calls, executes them, and returns the result. 
The main script `target_runner` then sends the function calls to the server via the socket. 
This way, the overhead of starting a new Julia process for each function call is avoided.
When using this variant of applying irace, ensure to adapt also `target_runner`
concerning your parameters.

To use irace on these examples, call irace from the respective subdirectory.


## Tuning with SMAC3

*We generally recommend now to use irace in conjunction with MHLib instead of SMAC, as it is a more elaborated system for tuning parameters of metaheuristics.*

[SMAC3](https://automl.github.io/SMAC3) is another tool for automated algorithm 
configuration and parameter tuning. It is based on Bayesian optimization and utilizes
racing mechanisms and random forests.

The directory `tuning/smac`contains an example with three different variants on how SMAC3 can 
specifically be used for tuning parameters of algorithms implemented in Julia. 

The main script is `tuning.py`, and the variant to be used can be provided as
command line argument:

1)  This variant uses the Python->Julia interface provided by the Python package `juliacall`.
    The julia function `f` can here be called from SMAC3 via a simple
    Python wrapper function. A limitation of this approach is that no parallelization
    via multithreading or multiprocessing is possible.

2)  Here, for each call of the function to tune, a new Julia process is started. While this
    approach allows for parallelization, also using a compute cluster, it can be quite
    inefficient when the function to tune is comparably fast w.r.t. Julia's startup and
    pre-compile times.

3) Here, the main Python script spawns a fixed number of Julia "server" subprocesses
    realized by `julia_server.jl`, which accept function calls, perform them, 
    and return the result. The communication is done via pipes, respectively stdin/stdout. 
    In this way multiprocessing is utilized, but the overhead of starting 
    a new Julia process for each function call is avoided.

Note that the demo needs to be started from the `tuning` subdirectory.

Ensure that Python is installed with current versions of  packages `juliacall`, `smac`, 
and `ConfigSpace`.


