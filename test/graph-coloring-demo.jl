#!/usr/bin/env julia
"""
    graph-coloring-demo

Standalone demo program for solving the Graph Coloring Problem.
"""

using ArgParse
using MHLib
using MHLib.Schedulers
using MHLib.GVNSs
using MHLib.GraphColoring


const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--alg"
        help = "Algorithm to apply (currently only gvns is implemented)"
        arg_type = String
        default = "gvns"
    "--gcp_colors"
        help = "number of colors for the graph coloring problem"
        arg_type = Int
        default = 3
end

println("Graph Coloring Demo version $(git_version())\nARGS: ", ARGS)
settings_new_default_value!(MHLib.settings_cfg, "ifile", "test/data/fpsol2.i.1.col")
# settings_new_default_value!(MHLib.settings_cfg, "ifile", "test/data/test.col")
settings_new_default_value!(MHLib.Schedulers.settings_cfg, "mh_titer", 1000)
settings_new_default_value!(settings_cfg, "gcp_colors", 3)
parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg, settings_cfg])
println(get_settings_as_string())

function graph_coloring()
    inst = GraphColoringInstance(settings[:ifile])
    sol = GraphColoringSolution(inst)
    println(sol)
    
    if settings[:alg] == "gvns"
        alg = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh$i", shaking!, i) for i in 1:5])
    else
        error("Invalid parameter alg $(settings[:alg])")
    end

    run!(alg)
    main_results(alg.scheduler)
    check(sol)
end

graph_coloring()
