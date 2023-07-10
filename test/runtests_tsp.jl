# always run this code in the test directory and the test environment
cd(@__DIR__)
using Pkg; Pkg.activate(".")

using Test
using Random
using Revise

if isdefined(@__MODULE__, :LanguageServer)  # hack for VSCode to see symbols
    include("../src/MHLib.jl")
    using .MHLib
    using .MHLib.Schedulers
    using .MHLib.GVNSs
    using .MHLib.LNSs
else
    using MHLib
    using MHLib.Schedulers
    using MHLib.GVNSs
    using MHLib.LNSs
end

includet("TSP.jl")
using .TSP


@testset "Random-Init-TSP.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=10"])
    println(get_settings_as_string())
    inst = TSPInstance("data/xqf131.tsp")
    sol = TSPSolution(inst)
    println(sol)
    println(obj(sol))
    @test obj(sol) >= 0
    @test sol.obj_val_valid

    initialize!(sol)
    @test !sol.obj_val_valid
    println(sol)
    println(obj(sol))
    @test obj(sol) >= 0
end

@testset "LNS-TSP.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.LNSs.settings_cfg], 
        ["--seed=1", "--mh_titer=20000"])
    inst = TSPInstance("data/xqf131.tsp")
    sol = TSPSolution(inst)
    initialize!(sol)
    println(sol)
    println(obj(sol))
    @test obj(sol) >= 0
    @test sol.obj_val_valid
    @assert !to_maximize(sol)
    alg = LNS(sol, MHMethod[MHMethod("con", construct!, 0)],
        [MHMethod("de$i", LNSs.destroy!, i) for i in 1:3],
        [MHMethod("re", LNSs.repair!, 1)], 
        consider_initial_sol = true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testset "GVNS-2OPT-2EX-TSP.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=5000"])
    inst = TSPInstance("data/xqf131.tsp")
    sol = TSPSolution(inst)
    println(sol)
    println(obj(sol))
    @test obj(sol) >= 0
    @test sol.obj_val_valid

    @assert !to_maximize(sol)

    alg = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],[MHMethod("sh1", shaking!, 1)], 
        consider_initial_sol = true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end
