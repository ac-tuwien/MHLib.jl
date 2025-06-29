# OneMax.jl
#
# OneMax demo problem: Maximize the number of set bits in a binary string.
#
# This trivial problem is just for very simple demonstration/testing purposes.


export OneMaxSolution, solve_onemax


"""
    OneMaxSolution

A concrete solution type to solve the MAXSAT problem.

As the problem is so simply defined, we do not need a separate instance structure
but store problem size directly within the solutions.
"""
mutable struct OneMaxSolution <: BoolVectorSolution
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Bool}
end

OneMaxSolution(n) = OneMaxSolution(-1, false, Vector{Bool}(undef, n))

OneMaxSolution(s::OneMaxSolution) = OneMaxSolution(s.obj_val, s.obj_val_valid, copy(s.x))

MHLib.calc_objective(s::OneMaxSolution) = sum(s.x)

Base.show(io::IO, s::OneMaxSolution) = println(io, s.x)

function Base.copy!(s1::OneMaxSolution, s2::OneMaxSolution)
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    copy!(s1.x, s2.x)
end

Base.copy(s::OneMaxSolution) = OneMaxSolution(s.obj_val, s.obj_val_valid, copy(s.x))

"""
    destroy!(sol::OneMaxSolution, k::Int, result::Result)

`MHMethod` that destroys `k` bits in the solution `sol` by calling `shaking!`.

Note that this is not really a meaningful destroy operation for the OneMax problem.
It is just for testing purposes to be able to use the (A)LNS on this problem.
"""
destroy!(sol::OneMaxSolution, k::Int, result::Result) = shaking!(sol, k, result)

"""
    repair!(sol::OneMaxSolution, ::Nothing, result::Result)

`MHMethod` that "repairs" one bit in the solution `sol` by calling `shaking!`.

Note that this is not really a meaningful repair operation for the OneMax problem.
It is just for testing purposes to be able to use the (A)LNS on this problem.
"""
repair!(sol::OneMaxSolution, ::Nothing, result::Result) = shaking!(sol, 1, result)


# -------------------------------------------------------------------------------

"""
    solve_onemax(n::Int=100; seed=nothing, titer::Int==100, kwargs...)

Solve the OneMax problem with `n` bits, using a variable neighborhood search.

# Parameters
- `n::Int`: number of bits in the OneMax problem
- `seed::Int`: random seed to use for reproducibility, or `nothing` to pick one randomly
- `titer::Int`: number of iterations for the variable neighborhood search, gets new default value
- `kwargs...`: keyword arguments to pass to the GVNS algorithm, e.g. `ttime`, etc.
"""
function solve_onemax(n::Int=100; seed=nothing, titer=100, kwargs...)
    # Make results reproducibly by either setting a given seed or picking one randomly
    isnothing(seed) && (seed = rand(0:typemax(Int32)))
    Random.seed!(seed)

    println("OneMax Demo $(git_version())")
    println("n=$n, seed=$seed, ", NamedTuple(kwargs))
        
    sol = OneMaxSolution(n)
    initialize!(sol)
    println(sol)

    # Apply a variable neighborhood search, making use of a simple construction
    # heuristic, a local improvement method, and a shaking method.
    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!, 1)], [MHMethod("sh1", shaking!, 1)];
        consider_initial_sol=true, titer,
        kwargs...)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end

# To run from REPL, use `MHLib` and call e.g. `solve_onemax(200, titer=200, seed=1)`.

# Run with profiler:
# @profview solve_onemax()
