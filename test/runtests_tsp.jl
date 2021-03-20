using Test
using Random
using Revise

if isdefined(@__MODULE__, :LanguageServer)  # hack for VSCode to see symbols
    using .MHLib
    using .MHLib.Schedulers
    using .MHLib.GVNSs
    using .MHLib.ALNSs
else
    using MHLib
    using MHLib.Schedulers
    using MHLib.GVNSs
    using MHLib.ALNSs
end

includet("TSP.jl")
using .TSP

# always run this code in the test directory
cd(@__DIR__)

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

@testset "GVNS-2OPT-2EX-TSP.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=5000"])
    inst = TSPInstance("data/xqf131.tsp")
    sol = TSPSolution(inst)
    println(sol)
    println(obj(sol))
    @test obj(sol) >= 0
    @test sol.obj_val_valid

    @assert !to_maximize(sol)

    local_search = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],[MHMethod("sh1", shaking!, 1)], 
        consider_initial_sol = true)
    GVNSs.run!(local_search)
    main_results(local_search.scheduler)
    
    @test obj(sol) >= 0
end
