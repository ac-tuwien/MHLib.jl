using Test
using Random

using MHLib
using MHLib.Schedulers
using MHLib.GVNSs
using MHLib.ALNSs

using MHLib.OneMax
using MHLib.MAXSAT
using MHLib.MKP
using MHLib.MISP


if endswith(pwd(), "test")
    cd("..")
end


@testset "OneMaxSolution" begin
    parse_settings!([MHLib.OneMax.settings_cfg], ["--seed=1"])
    println(get_settings_as_string())
    s1 = OneMaxSolution(5)
    initialize!(s1)
    s2 = OneMaxSolution(5)
    initialize!(s2)
    s3 = copy(s1)
    initialize!(s3)
    copy!(s1,s3)
    # println("$s1, $(obj(s1))\n$s2, $(obj(s2))\n$s3, $(obj(s3))")
    @test is_equal(s1, s3)
    @test dist(s1, s3) == 0
    s1.x[1] = !s1.x[1]; invalidate!(s1)
    @test !is_equal(s1, s3)
    @test dist(s1, s3) == 1
    # println("$s1, $(obj(s1))")
    check(s1)
end

@testset "GVNS-OneMax.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.OneMax.settings_cfg],
        ["--seed=1"])
    sol = OneMaxSolution(10)
    println(sol)
    # methods = [MHMethod("con", construct!, 0),
    #     MHMethod("li1", local_improve!, 1),
    #     MHMethod("sh1", shaking!, 1),
    #     MHMethod("sh2", shaking!, 2),
    #     MHMethod("sh3", shaking!, 3)]
    # sched = Scheduler(sol, methods)
    # for m in next_method(methods)
    #     perform_method!(sched, m, sol)
    #     println(sol)
    # end
    gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    GVNSs.run!(gvns)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testset "GVNS-MAXSAT.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=10"])
    inst = MAXSATInstance("data/maxsat-simple.cnf")
    sol = MAXSATSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    GVNSs.run!(gvns)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testset "ALNS-MAXSAT.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg], ["--seed=1"])
    inst = MAXSATInstance("data/maxsat-simple.cnf")
    sol = MAXSATSolution(inst)
    println(sol)
    alns = ALNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    ALNSs.run!(alns)
    main_results(alns.scheduler)
    @test obj(sol) >= 0
end

@testset "GVNS-MKP.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=25"])
    inst = MKPInstance("data/mknapcb5-01.txt")
    sol = MKPSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    GVNSs.run!(gvns)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testset "GVNS-MISP.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1",  "--mh_titer=25"])
    inst = MISPInstance("data/frb40-19-1.mis")
    sol = MISPSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    GVNSs.run!(gvns)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end
