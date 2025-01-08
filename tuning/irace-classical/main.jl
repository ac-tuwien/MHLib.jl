#!/usr/bin/env julia

# A main program that receives the parameters for tuning with irace via
# arguments, which are parsed into MHLib.settings

using Pkg
using MHLib

Pkg.activate(".", io=devnull)

include("../julia-function-to-tune.jl")

const main_settings_cfg = ArgParseSettings()

@add_arg_table main_settings_cfg begin
    "--inst"
        help = "Problem instances file"
        arg_type = String
        default = "test"
    "--new_seed"    
        help = "New seed for the random number generator"
        arg_type = Int
        default = 0
    "--x"
        help = "A float parameter"
        arg_type = Float64
        default = 0.0
    "--y"
        help = "An integer parameter"
        arg_type = Int
        default = 0
    "--z"
        help = "A string parameter"
        arg_type = String
        default = "opt1"
end


function main()
    parse_settings!([main_settings_cfg], ARGS)
    inst = settings[:inst]::String
    new_seed = settings[:new_seed]::Int
    x = settings[:x]::Float64
    y = settings[:y]::Int
    z = settings[:z]::String

    inst = ARGS[2]
    new_seed = parse(Int, ARGS[4])
    x = parse(Float64, ARGS[6])
    y = parse(Int, ARGS[8])
    z = ARGS[10]

    result = f(inst, new_seed, x, y, z)

    # Write resulting value as final line
    println(result)
end

main()