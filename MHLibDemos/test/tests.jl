# Unit tests for MHLibDemos

# Always performed in the test directory within the test environment.

using TestItems

# This initialization is executed before each test item
@testsnippet TestInit begin
    using Random
    using MHLib
    using MHLibDemos
    parse_settings!(mhlib_settings_cfgs, ["--seed=1", "--mh_titer=10"])
end


@testitem "GVNS-MAXSAT" setup=[TestInit] begin
    inst = MAXSATInstance("data/maxsat-simple.cnf")
    sol = MAXSATSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "MAXSAT-kflip" setup=[TestInit] begin
    inst = MAXSATInstance("data/maxsat-adv1.cnf")
    sol = MAXSATSolution(inst)

    k = 30
    old = copy(sol.x)
    k_random_flips!(sol, k)
    new = sol.x

    ndiff = sum(old .!= new)

    @test ndiff == k
end

@testitem "LNS-MAXSAT" setup=[TestInit] begin
    settings[:mh_titer] = 120
    inst = MAXSATInstance("data/maxsat-adv1.cnf")
    sol = MAXSATSolution(inst)
    println(sol)
    num_de = 5
    method_selector = WeightedRandomMethodSelector(num_de:-1:1, 1:1)
    alg = LNS(sol, [MHMethod("construct", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)]; method_selector)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testitem "LNS-MAXSAT" setup=[TestInit] begin
    settings[:mh_titer] = 120
    inst = TSPInstance(50)
    sol = TSPSolution(inst)
    println(sol)
    num_de = 3
    method_selector = WeightedRandomMethodSelector(num_de:-1:1, 1:1)
    alg = LNS(sol, [MHMethod("construct", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)]; method_selector)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testitem "ALNS-MAXSAT" setup=[TestInit] begin
    settings[:mh_titer] = 120
    inst = MAXSATInstance("data/maxsat-adv1.cnf")
    sol = MAXSATSolution(inst)
    println(sol)
    num_de = 5
    alg = ALNS(sol, [MHMethod("construct", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)])
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testitem "GVNS-MKP" setup=[TestInit] begin
    settings[:mh_titer] = 25
    inst = MKPInstance("data/mknapcb5-01.txt")
    sol = MKPSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "GVNS-MISP" setup=[TestInit] begin
    settings[:mh_titer] = 25
    inst = MISPInstance("data/frb40-19-1.mis")
    sol = MISPSolution(inst)
    println(sol)
    gvns = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)],)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "Random-Init-TSP" setup=[TestInit] begin
    rand_inst = TSPInstance()
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

@testitem "GVNS-TSP" setup=[TestInit] begin
    settings[:mh_titer] = 300
    inst = TSPInstance("data/xqf131.tsp")
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
        consider_initial_sol=true)
    run!(search)
    main_results(search.scheduler)
    @test obj(sol) >= 0
end

@testitem "GVNS-GraphColoring1" setup=[TestInit] begin
    parse_settings!([scheduler_settings_cfg, graph_coloring_settings_cfg], 
        ["--ifile=data/fpsol2.i.1.col", "--mh_titer=1000", "--gcp_colors=2"])
    inst = GraphColoringInstance(settings[:ifile])
    sol = GraphColoringSolution(inst)
    println(sol)

    @test obj(sol) >= 0
    @test sol.obj_val_valid
    @test !to_maximize(sol)

    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh$i", shaking!, i) for i in 1:5])
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    @test obj(sol) >= 0
end

@testitem "GVNS-GraphColoring2" setup=[TestInit] begin
    parse_settings!([scheduler_settings_cfg, graph_coloring_settings_cfg], 
        ["--ifile=data/test.col", "--mh_titer=50", "--gcp_colors=3"])
    inst = GraphColoringInstance(settings[:ifile])
    sol = GraphColoringSolution(inst)
    println(sol)
    @test obj(sol) >= 0
    @test sol.obj_val_valid
    @test !to_maximize(sol)
    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh$i", shaking!, i) for i in 1:5])
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    @test iszero(obj(sol))
end
