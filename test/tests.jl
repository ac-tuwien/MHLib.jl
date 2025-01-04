# Module tests

# always performed in the test directory within the test environment

using TestItems

@testsnippet TestInit begin
    using Random
    using MHLib
    parse_settings!(mhlib_settings_cfgs, ["--seed=1", "--mh_titer=10"])
end

@testitem "OneMaxSolution" setup=[TestInit] begin
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
            MHMethod("sh3", shaking!, 3)],)
    run!(gvns)
    method_statistics(gvns.scheduler)
    main_results(gvns.scheduler)
    @test obj(sol) >= 0
end

@testitem "LNS-OneMax" setup=[TestInit] begin
    sol = OneMaxSolution(100)
    settings[:mh_titer] = 120
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

@testitem "ALNS-OneMax" setup=[TestInit] begin
    sol = OneMaxSolution(100)
    settings[:mh_titer] = 120
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
