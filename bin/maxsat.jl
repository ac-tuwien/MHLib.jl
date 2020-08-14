#!/usr/local/bin/julia

using MHLib
using MHLib.Schedulers
# using MHLib.GVNSs
using MHLib.ALNSs
using MHLib.MAXSAT
using MHLib.LCS  # load LCS functions (exported)

println("MAXSAT Demo\nARGS: ", ARGS)
# settings_new_default_value("mh_titer", 1000)
parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg], [])
println(get_settings_as_string())

function maxsat()
    inst = MAXSATInstance("../data/maxsat-adv1.cnf")

    sol = MAXSATSolution(inst)
    println(sol)

    alns = ALNS(sol, [MHMethod("construct", construct!, 0)],
        [MHMethod("destroy", destroy!, 1)],
        [MHMethod("repair", repair!, 0)])
    run!(alns)
    main_results(alns.scheduler)

    #=
    gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh$i", shaking!, i) for i in 1:5])
    run!(gvns)
    main_results(gvns.scheduler)
    =#
    check(sol)
end

maxsat()
