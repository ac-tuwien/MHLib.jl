"""
    OneMax

OneMax demo problem: Maximize the number of set bits in a binary string.

This problem is just for very simple demonstration/testing purposes.
"""

export OneMaxSolution, onemax_settings_cfg, solve_onemax

# We define an additional problem-specific parameter:
const onemax_settings_cfg = ArgParseSettings()
@add_arg_table! onemax_settings_cfg begin
    "--onemax_n"
        help = "length of solution string in the problem"
        arg_type = Int
        default = 100
end


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

function solve_onemax(args=ARGS)
    println("OneMax Demo version $(git_version())\nARGS: ", args)
    args isa AbstractString && (args = split(args))

    # We set some new default values for parameters and parse all relevant arguments
    settings_new_default_value!(MHLib.scheduler_settings_cfg, "mh_titer", 100)
    parse_settings!([MHLib.scheduler_settings_cfg, onemax_settings_cfg], args)
    println(get_settings_as_string())
        
    sol = OneMaxSolution(settings[:onemax_n])
    initialize!(sol)
    println(sol)

    # We apply here a variable neighborhood search, making use of a simple construction
    # heuristic, a local improvement method, and a shaking method.
    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!, 1)],[MHMethod("sh1", shaking!, 1)], 
        consider_initial_sol = true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end

# To run from REPL, use `MHLibDemos` and call `solve_onemax(<args>)` where `<args>`
# is a single string or list of strings being passed as arguments for setting global 
# parameters, e.g. `solve_onemax("--seed=1 --mh_titer=120")`.
# `@<filename>` may be used to read arguments from a configuration file <filename>

# Run with profiler:
# @profview solve_onemax(args)
