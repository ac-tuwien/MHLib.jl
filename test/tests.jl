# tests.jl
# 
# Unit tests for MHLib.
#
# Always performed in the test directory within the test environment.

using TestItems

@testsnippet TestInit begin
    using Random
    using MHLib
    Random.seed!(1)
end

@testitem "OneMaxSolution" setup=[TestInit] begin
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

@testitem "GVNS-OneMax" setup=[TestInit] begin
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
    gvns = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)], titer=10, checkit=true)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "LNS-OneMax" setup=[TestInit] begin
    sol = OneMaxSolution(100)
    sol.x .= true; invalidate!(sol)  # this initial solution will be ignored
    num_de = 5
    method_selector = WeightedRandomMethodSelector(num_de:-1:1, 1:1)
    alg = LNS(sol, [MHMethod("const", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)]; method_selector, titer=120, checkit=true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test 1 < obj(sol) < 100

    sol.x .= true; invalidate!(sol)  # start with optimal solution that should be considered
    alg = LNS(sol, [MHMethod("const", construct!)],
        [MHMethod("de", destroy!, 1)],
        [MHMethod("re", repair!)]; 
        titer=3, lfreq=1, consider_initial_sol=true, checkit=true)
    run!(alg)
    @test obj(sol) == 100
end

@testitem "VND-revisits-earlier-neighborhoods" setup=[TestInit] begin
    # Method li_a can only improve once li_b has been applied; li_b honestly reports that
    # afterwards the solution is a local optimum w.r.t. itself. The VND must nevertheless
    # return to li_a after li_b's improvement instead of terminating.
    function li_a(s::OneMaxSolution, ::Nothing, res::Result)
        if s.x[2] && !s.x[1]
            s.x[1] = true; invalidate!(s); res.changed = true
        else
            res.changed = false
        end
    end
    function li_b(s::OneMaxSolution, ::Nothing, res::Result)
        res.changed = !s.x[2]
        s.x[2] = true; invalidate!(s)
        res.is_local_optimum = true
    end
    sol = OneMaxSolution(2)  # x = [false, false]
    gvns = GVNS(sol, MHMethod[], [MHMethod("a", li_a), MHMethod("b", li_b)], MHMethod[];
        consider_initial_sol=true, titer=-1, log=false)
    @test !vnd!(gvns, sol)
    @test obj(sol) == 2
end

@testitem "ALNS-OneMax" setup=[TestInit] begin
    sol = OneMaxSolution(100)
    println(sol)
    num_de = 5
    alg = ALNS(sol, [MHMethod("const", construct!)],
        [MHMethod("de$i", destroy!, i) for i in 1:num_de],
        [MHMethod("re", repair!)]; titer=120, checkit=true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    @test obj(sol) >= 0
end

@testitem "LNS-incumbent-not-aliased" setup=[TestInit] begin
    sol = OneMaxSolution(100)
    sol.x .= true; invalidate!(sol)
    lns = LNS(sol, MHMethod[], [MHMethod("de", destroy!, 5)], [MHMethod("re", repair!)];
        consider_initial_sol=true, init_temp=1000.0, temp_dec_factor=1.0, titer=-1,
        log=false)
    for _ in 1:20
        MHLib.lns_iteration!(lns)
    end
    @test obj(lns.scheduler.incumbent) == 100
    @test obj(sol) == 100
    @test lns.solution !== lns.scheduler.incumbent
end

@testitem "Constructor-argument-checks" setup=[TestInit] begin
    sol = OneMaxSolution(10)
    # method names must be unique, otherwise their statistics would silently be merged
    @test_throws ArgumentError Scheduler(sol,
        [MHMethod("m", construct!), MHMethod("m", shaking!, 1)])
    # meths_compat must have one row per destroy and one column per repair method
    meths_de = [MHMethod("de1", destroy!, 1), MHMethod("de2", destroy!, 2)]
    meths_re = [MHMethod("re", repair!)]
    @test_throws ArgumentError LNS(sol, MHMethod[], meths_de, meths_re;
        meths_compat=[true;;], consider_initial_sol=true, log=false)
    lns = LNS(sol, MHMethod[], meths_de, meths_re;
        meths_compat=[true; false;;], consider_initial_sol=true, log=false)
    @test lns.meths_compat == [true; false;;]
end
