#!/usr/bin/env -S julia --project=.

using ArgParse

arg_table = ArgParseSettings()
@add_arg_table arg_table begin
    "--seed"
        arg_type = Int
    "--x"
        arg_type = Float64
        required = true
    "--y"
        arg_type = Int
        required = true
    "--z"
        arg_type = String
        required = true
end

include("julia-function-to-tune.jl")

# open("args.txt", "w") do f
#     println(f, ARGS)
# end

args = parse_args(ARGS, arg_table)

c = f(args["x"], args["y"], args["z"])

println("cost=", c)
exit()

