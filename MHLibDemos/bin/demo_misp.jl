#!/usr/bin/env julia
"""
    demo_misp.jl

Standalone demo program for solving the MISP.
"""

module Demo_MISP

using ArgParse
using Random
using Revise
using MHLib
using MHLibDemos

const settings_cfg = ArgParseSettings()


function solve()
    inst = MISPInstance("data/frb40-19-1.mis")
    sol = MISPSolution(inst)
    initialize!(sol)
    check(sol)
    # println(sol)

    alg = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)], 
        consider_initial_sol = true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end


println("MISP Demo version $(git_version())\nARGS: ", ARGS)
settings_new_default_value!(MHLib.Schedulers.settings_cfg, "mh_titer", 10000)
parse_settings!([MHLib.Schedulers.settings_cfg])
println(get_settings_as_string())

end  # module

Demo_MISP.solve()
