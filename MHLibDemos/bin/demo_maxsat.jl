#!/usr/bin/env julia
"""
    demo_maxsat.jl

Standalone demo program for solving the MAXSAT problem.
"""

# activate MHLibDemos environment
cd(@__DIR__()*"/..")
using Pkg; Pkg.activate(".") 

module Demo_maxsat

using ArgParse
using Revise
using MHLib
using MHLibDemos


const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--alg"
        help = "Algorithm to apply (gvns, lns, weighted-lns, alns)"
        arg_type = String
        default = "alns"
end

println("MAXSAT Demo version $(git_version())\nARGS: ", ARGS)
settings_new_default_value!(MHLib.settings_cfg, "ifile", "data/maxsat-adv1.cnf")
settings_new_default_value!(MHLib.Schedulers.settings_cfg, "mh_titer", 1000)
parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.LNSs.settings_cfg, 
    MHLib.ALNSs.settings_cfg, settings_cfg])
println(get_settings_as_string())

function solve_maxsat()
    inst = MAXSATInstance(settings[:ifile])
    sol = MAXSATSolution(inst)
    println(sol)

    if settings[:alg] === "lns"
        alg = LNS(sol, [MHMethod("construct", construct!, 0)],
            [MHMethod("de", destroy!, 1)],
            [MHMethod("re", repair!, 0)];
            meths_compat = [true;;])
    elseif settings[:alg] === "weighted-lns"
        num_de = 5
        method_selector = WeightedRandomMethodSelector(num_re:-1:1, 1:1)
        alg = LNS(sol, [MHMethod("construct", construct!, 0)],
            [MHMethod("de$i", destroy!, i) for i in 1:num_de],
            [MHMethod("re", repair!, 0)];
            method_selector)
    elseif settings[:alg] === "alns"
        num_de = 5
        alg = ALNS(sol, [MHMethod("construct", construct!, 0)],
            [MHMethod("de$i", destroy!, i) for i in 1:num_de],
            [MHMethod("re", repair!, 0)])
    elseif settings[:alg] === "gvns"
        alg = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh$i", shaking!, i) for i in 1:5])
    else
        error("Invalid parameter alg: $(settings[:alg])")
    end
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end

end  # module

Demo_maxsat.solve_maxsat()
# @profview Demo_maxsat.solve_maxsat()