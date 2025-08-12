# Algorithm/Parameter Tuning of Julia Programs with irace

In the examples provided in the `tuning`directory, file `tuning.julia-function-to-tune.jl`
contains a simple example of a function whose parameters shall be tuned.

## Tuning with irace

Ensure to have [irace](https://github.com/MLopez-Ibanez/irace) installed on your system.

The straight-forward way to use irace with Julia is illustrated in the 
`tuning/irace-classical` directory. All the configuration files are mostly provided by 
`irace --init` and just minimally adapted to fit our example. Most importantly, 
`target_runner` calls here Julia to execute `main.jl`, in which the provided 
arguments are parsed and the function to tune is called. The result is then printed to stdout.

This direct approach is less efficient when the startup time for Julia and the program to 
be tuned is high in comparison to the actual algorithm evaluation.

For this case, directory `tuning/irace-with-julia-server` provides a different solution.
`julia-server` is a simple standalone server program, which first needs to be started, 
e.g., directly from the command line by `julia julia-server.jl`. 
It listens on a socket for incoming function calls, executes them, and returns the result. 
Now, the main script `target_runner` from `irace` sends the function calls to the server 
via the socket. 
This way, the overhead of starting a new Julia process for each function call is avoided.
When using this variant of applying irace, ensure to adapt also `target_runner`
concerning your parameters.

To use irace on these examples, simply call irace from the respective subdirectory.


