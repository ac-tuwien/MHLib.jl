# tests.jl
#
# Unit tests for MHLibDemos.
#
# Always performed in the test directory within the test environment.

using TestItems

# This initialization is executed before each test item
@testsnippet MHLibTestInit begin
    using Random
    using MHLib
    using MHLibDemos
    Random.seed!(1)
    datapath = joinpath(@__DIR__, "..", "data")
end


@testitem "GVNS-MAXSAT" setup=[MHLibTestInit] begin
    inst = MAXSATInstance(joinpath(datapath, "maxsat-simple.cnf"))
    sol = MAXSATSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)]; titer=10)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "MAXSAT-kflip" setup=[MHLibTestInit] begin
    inst = MAXSATInstance(joinpath(datapath, "maxsat-adv1.cnf"))
    sol = MAXSATSolution(inst)

    k = 30
    old = copy(sol.x)
    k_random_flips!(sol, k)
    new = sol.x

    ndiff = sum(old .!= new)

    @test ndiff == k
end

@testitem "LNS-MAXSAT" setup=[MHLibTestInit] begin
    inst = MAXSATInstance(joinpath(datapath, "maxsat-adv1.cnf"))
    sol = MAXSATSolution(inst)
    println(sol)
    num_de = 5
    method_selector = WeightedRandomMethodSelector(num_de:-1:1, 1:1)
    alg = LNS(sol, [MHMethod("const", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)]; method_selector, titer=120)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testitem "LNS-MAXSAT" setup=[MHLibTestInit] begin
    inst = TSPInstance(50)
    sol = TSPSolution(inst)
    println(sol)
    num_de = 3
    method_selector = WeightedRandomMethodSelector(num_de:-1:1, 1:1)
    alg = LNS(sol, [MHMethod("const", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)]; method_selector, titer=120)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testitem "ALNS-MAXSAT" setup=[MHLibTestInit] begin
    inst = MAXSATInstance(joinpath(datapath, "maxsat-adv1.cnf"))
    sol = MAXSATSolution(inst)
    println(sol)
    num_de = 5
    alg = ALNS(sol, [MHMethod("const", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)], titer=120)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testitem "GVNS-MKP" setup=[MHLibTestInit] begin
    inst = MKPInstance(joinpath(datapath, "mknapcb5-01.txt"))
    sol = MKPSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)], titer=25)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "GVNS-MISP" setup=[MHLibTestInit] begin
    inst = MISPInstance(joinpath(datapath, "frb40-19-1.mis"))
    sol = MISPSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)], titer=25)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "Random-Init-TSP" setup=[MHLibTestInit] begin
    rand_inst = TSPInstance()
    inst = TSPInstance(joinpath(datapath, "xqf131.tsp"))
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

@testitem "GVNS-TSP" setup=[MHLibTestInit] begin
    inst = TSPInstance(joinpath(datapath, "xqf131.tsp"))
    sol = TSPSolution(inst)
    initialize!(sol)
    println(sol)
    println(obj(sol))
    @test obj(sol) >= 0
    @test sol.obj_val_valid
    @assert !to_maximize(sol)
    search = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh1", shaking!, 1)],
        consider_initial_sol=true, titer=300)
    run!(search)
    main_results(search.scheduler)
    @test obj(sol) >= 0
end

@testitem "GVNS-GraphColoring1" setup=[MHLibTestInit] begin
    inst = GraphColoringInstance(joinpath(datapath, "fpsol2.i.1.col"), 2)
    sol = GraphColoringSolution(inst)
    println(sol)

    @test obj(sol) >= 0
    @test sol.obj_val_valid
    @test !to_maximize(sol)

    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh$i", shaking!, i) for i in 1:5],
        titer=1000)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    @test obj(sol) >= 0
end

@testitem "GVNS-GraphColoring2" setup=[MHLibTestInit] begin
    inst = GraphColoringInstance(joinpath(datapath, "test.col"), 3)
    sol = GraphColoringSolution(inst)
    println(sol)
    @test obj(sol) >= 0
    @test sol.obj_val_valid
    @test !to_maximize(sol)
    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh$i", shaking!, i) for i in 1:5],
        titer=50)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    @test iszero(obj(sol))
end
