using Test
using MHLib
using MHLib.OneMax

@testset "MHLib.jl" begin
    parse_settings!(["--seed=1"])
    println(get_settings_as_string())

    s1 = OneMaxSolution{5}()
    initialize!(s1)
    s2 = OneMaxSolution{5}()
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
