#!/usr/local/bin/julia

using MHLib
using MHLib.Schedulers
using MHLib.GVNSs
using MHLib.OneMax
using MHLib.MAXSAT

println(ARGS)
parse_settings!()
println(get_settings_as_string())

function onemax()
    initialize!(s1)
    s2 = OneMaxSolution{5}()
    initialize!(s2)
    s3 = copy(s1)
    initialize!(s3)
    copy!(s1,s3)
    println("$s1, $(obj(s1))\n$s2, $(obj(s2))\n$s3, $(obj(s3))")
end

# function maxsat()
    inst = MAXSATInstance("data/maxsat-simple.cnf")
    sol = MAXSATSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],) 
    run!(gvns)
    main_results(gvns.scheduler)
    check(sol)
# end

# maxsat()
