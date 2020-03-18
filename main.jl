#!/usr/local/bin/julia

using MHLib
using MHLib.Schedulers
#using MHLib.GVNSs
using MHLib.ALNSs
import MHLib.OneMax: OneMaxSolution
using MHLib.MAXSAT

println("Arguments: ", ARGS)
settings_new_default_value("mh_titer", 1000)
parse_settings!()
println(get_settings_as_string())

function onemax()
    s1 = OneMaxSolution{5}()
    initialize!(s1)
    s2 = OneMaxSolution{5}()
    initialize!(s2)
    s3 = copy(s1)
    initialize!(s3)
    copy!(s1,s3)
    println("$s1, $(obj(s1))\n$s2, $(obj(s2))\n$s3, $(obj(s3))")
end

function maxsat()
    inst = MAXSATInstance("data/maxsat-adv1.cnf")
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
