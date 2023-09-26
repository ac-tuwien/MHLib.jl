#!/usr/bin/env julia
"""
    demo_graph_coloring.jl

Standalone demo program for solving the Graph Coloring Problem.
"""

# activate MHLibDemos environment
cd(@__DIR__()*"/..")
using Pkg; Pkg.activate(".") 

module Demo_GC

using ArgParse
using Random
using Revise
using MHLib
using MHLibDemos

const settings_cfg = ArgParseSettings()


function solve()
    inst = GraphColoringInstance("data/fpsol2.i.1.col")
    sol = GraphColoringSolution(inst)
    initialize!(sol)
    println(sol)

    alg = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],[MHMethod("sh1", shaking!, 1)], 
        consider_initial_sol = true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end


println("Graph Coloring Demo version $(git_version())\nARGS: ", ARGS)
settings_new_default_value!(MHLib.Schedulers.settings_cfg, "mh_titer", 10000)
parse_settings!([MHLib.Schedulers.settings_cfg, MHLibDemos.settings_cfg])
println(get_settings_as_string())

end  # module


Demo_GC.solve()
