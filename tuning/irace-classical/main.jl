#!/usr/bin/env julia

# A main program that receives the parameters for tuning with irace via
# command line arguments. It calls the actual function `f`to be tuned.

using Random

include("../julia-function-to-tune.jl")


function (@main)(ARGS)
    # parse arguments according to expected types
    # the parameters to be tuned (here x, y , z) are always passed with a "--<name>" 
    # before the actual value, e.g., "--x 0.5", "--y 10", "--z opt2"
    @info ARGS
    inst = ARGS[1]
    seed = parse(Int, ARGS[2])
    @assert ARGS[3] === "--x"
    x = parse(Float64, ARGS[4])
    @assert ARGS[5] === "--y"
    y = parse(Int, ARGS[6])
    @assert ARGS[7] === "--z"
    z = ARGS[8]

    result = f(inst, seed, x, y, z)

    # Write resulting value as final line; irace reads this value to be minimized from stdout
    println(result)
end
