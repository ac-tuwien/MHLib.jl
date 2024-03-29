# always run this code in the test directory and the test environment
cd(@__DIR__)

using Test
using Random
using MHLib

include("../MHLibDemos/src/MHLibDemos.jl")
using .MHLibDemos


# testsets to perform:
only_testsets = ARGS
# Ignore DEBUG_MODE=... argument provided by VSCode
# if length(only_testsets) >= 1 && startswith(only_testsets[1], "DEBUG_MODE")
#     only_testsets = only_testsets[2:end]
# end
# only_testsets = ["GVNS-GraphColoring"]
# only_testsets = ["ALNS-MAXSAT"]
# only_testsets = ["GVNS-MKP"]

if isempty(only_testsets) || "OneMaxSolution" in only_testsets
    @testset "OneMaxSolution" begin
        parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1"])
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
end

if isempty(only_testsets) || "GVNS-OneMax" in only_testsets
    @testset "GVNS-OneMax.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg],
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
        run!(gvns)
        method_statistics(gvns.scheduler)
        main_results(gvns.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "GVNS-MAXSAT" in only_testsets
    @testset "GVNS-MAXSAT.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=10"])
        inst = MAXSATInstance("data/maxsat-simple.cnf")
        sol = MAXSATSolution(inst)
        println(sol)
        gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
                MHMethod("sh3", shaking!, 3)],)
        run!(gvns)
        method_statistics(gvns.scheduler)
        main_results(gvns.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "MAXSAT-kflip" in only_testsets
    @testset "MAXSAT-kflip.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=10"])
        inst = MAXSATInstance("data/maxsat-adv1.cnf")
        sol = MAXSATSolution(inst)

        k = 30
        old = copy(sol.x)
        k_random_flips!(sol, k)
        new = sol.x

        ndiff = sum(old .!= new)

        @test ndiff == k
    end
end

if isempty(only_testsets) || "LNS-MAXSAT" in only_testsets
    @testset "LNS-MAXSAT.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.LNSs.settings_cfg, 
            MHLib.ALNSs.settings_cfg], ["--seed=1", "--mh_titer=120"])
        inst = MAXSATInstance("data/maxsat-adv1.cnf")
        sol = MAXSATSolution(inst)
        println(sol)
        num_de = 5
        method_selector = WeightedRandomMethodSelector(num_de:-1:1, 1:1)
        alg = LNS(sol, [MHMethod("construct", construct!, 0)],
            [MHMethod("de$i", destroy!, i) for i in 1:num_de],
            [MHMethod("re", repair!, 0)]; method_selector)
        run!(alg)
        method_statistics(alg.scheduler)
        main_results(alg.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "LNS-TSP" in only_testsets
    @testset "LNS-MAXSAT.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.LNSs.settings_cfg,],
            ["--seed=1", "--mh_titer=120"])
        inst = TSPInstance(50)
        sol = TSPSolution(inst)
        println(sol)
        num_de = 3
        method_selector = WeightedRandomMethodSelector(num_de:-1:1, 1:1)
        alg = LNS(sol, [MHMethod("construct", construct!, 0)],
            [MHMethod("de$i", destroy!, i) for i in 1:num_de],
            [MHMethod("re", repair!, 0)]; method_selector)
        run!(alg)
        method_statistics(alg.scheduler)
        main_results(alg.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "ALNS-MAXSAT" in only_testsets
    @testset "ALNS-MAXSAT.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.LNSs.settings_cfg, 
            MHLib.ALNSs.settings_cfg], ["--seed=1", "--mh_titer=120"])
        inst = MAXSATInstance("data/maxsat-adv1.cnf")
        sol = MAXSATSolution(inst)
        println(sol)
        num_de = 5
        alg = ALNS(sol, [MHMethod("construct", construct!, 0)],
            [MHMethod("de$i", destroy!, i) for i in 1:num_de],
            [MHMethod("re", repair!, 0)])
        run!(alg)
        method_statistics(alg.scheduler)
        main_results(alg.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "GVNS-MKP" in only_testsets
    @testset "GVNS-MKP.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=25", 
            "--mh_checkit=true"])
        inst = MKPInstance("data/mknapcb5-01.txt")
        sol = MKPSolution(inst)
        println(sol)
        gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
                MHMethod("sh3", shaking!, 3)],)
        run!(gvns)
        method_statistics(gvns.scheduler)
        main_results(gvns.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "GVNS-MISP" in only_testsets
    @testset "GVNS-MISP.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1",  "--mh_titer=25"])
        inst = MISPInstance("data/frb40-19-1.mis")
        sol = MISPSolution(inst)
        println(sol)
        gvns = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
                MHMethod("sh3", shaking!, 3)],)
        run!(gvns)
        method_statistics(gvns.scheduler)
        main_results(gvns.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "Random-Init-TSP" in only_testsets
    @testset "Random-Init-TSP.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=1", "--mh_titer=10"])
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
end

if isempty(only_testsets) || "GVNS-TSP" in only_testsets
    @testset "GVNS-TSP.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg], ["--seed=10", "--mh_titer=300"]) 
            # "--mh_lfreq=-1", "--mh_lnewinc=false"])
        inst = TSPInstance("data/xqf131.tsp")
        sol = TSPSolution(inst)
        initialize!(sol)
        println(sol)
        println(obj(sol))
        @test obj(sol) >= 0
        @test sol.obj_val_valid
        @assert !to_maximize(sol)
        search = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],[MHMethod("sh1", shaking!, 1)],
            consider_initial_sol=true)
        run!(search)
        main_results(search.scheduler)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "GVNS-GraphColoring" in only_testsets
    @testset "GVNS-GraphColoring1.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg, 
            MHLibDemos.graph_coloring_settings_cfg], 
            ["--ifile=data/fpsol2.i.1.col", "--mh_titer=1000", "--gcp_colors=2"])
        inst = GraphColoringInstance(settings[:ifile])
        sol = GraphColoringSolution(inst)
        println(sol)

        @test obj(sol) >= 0
        @test sol.obj_val_valid
        @test !to_maximize(sol)

        alg = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh$i", shaking!, i) for i in 1:5])
        run!(alg)
        method_statistics(alg.scheduler)
        main_results(alg.scheduler)
        check(sol)
        @test obj(sol) >= 0
    end
end

if isempty(only_testsets) || "GVNS-GraphColoring2" in only_testsets
    @testset "GVNS-GraphColoring2.jl" begin
        parse_settings!([MHLib.Schedulers.settings_cfg, 
            MHLibDemos.graph_coloring_settings_cfg], 
            ["--ifile=data/test.col", "--mh_titer=50", "--gcp_colors=3"])
        inst = GraphColoringInstance(settings[:ifile])
        sol = GraphColoringSolution(inst)
        println(sol)
        @test obj(sol) >= 0
        @test sol.obj_val_valid
        @test !to_maximize(sol)
        alg = GVNS(sol, [MHMethod("con", construct!, 0)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh$i", shaking!, i) for i in 1:5])
        run!(alg)
        method_statistics(alg.scheduler)
        main_results(alg.scheduler)
        check(sol)
        @test obj(sol) == 0
    end
end
