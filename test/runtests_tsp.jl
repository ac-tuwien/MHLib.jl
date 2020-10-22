using Test
using Random

using MHLib
using MHLib.Schedulers
using MHLib.GVNSs
using MHLib.ALNSs

include("TSP.jl")

using .TSP

# always run this code in the test directory
if !endswith(pwd(), "test")
    cd("test")
end

@testset "Random-Init-TSP.jl" begin
    parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=10"])
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
