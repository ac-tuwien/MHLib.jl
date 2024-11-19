#!/usr/bin/env -S julia --project=.

# Julia script that parses command line arguments and calls function f
# returns result via stdout

# put provided command line arguments with their values into a dictionary
d = Dict{String, String}()
for arg in ARGS
    (name, value) = split(arg, "=")
    d[name] = value
end

include("julia-function-to-tune.jl")

# open("args.txt", "w") do f
#     println(f, d)
# end

c = f(d["--instance"], parse(Int, d["--seed"]),
        parse(Float64, d["--x"]), 
        parse(Int, d["--y"]), 
        d["--z"],
    )

println("cost=", c)


