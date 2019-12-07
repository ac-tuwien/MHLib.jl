#=
OneMax Problem: Maximize the number of set bits in a binary string.
This problem is just for simple demonstration/debugging purposes.
=#
module OneMax

import Base: copy, copy!
using ArgParse
import MHLib: BoolVectorSolution, calc_objective, settings, settings_cfg, initialize!,
    invalidate!, k_random_flips!
import MHLib.Schedulers: Method, Result

export OneMaxSolution, construct!, local_improve!, shaking!


@add_arg_table settings_cfg begin
    "--onemax_n"
        help = "length of solution string in the problem"
        arg_type = Int
        default = 100
end


"""
A concrete solution type to solve the MAXSAT problem.
"""
mutable struct OneMaxSolution{N} <: BoolVectorSolution{N}
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Bool}
    OneMaxSolution{N}() where {N} = new{N}(-1, false, Vector{Bool}(undef, N))
    OneMaxSolution{N}(s::OneMaxSolution{N}) where {N} =
        new{N}(s.obj_val, s.obj_val_valid, copy(s.x))
end


calc_objective(s::OneMaxSolution) = sum(s.x)


function copy!(s1::S, s2::S) where {S <: OneMaxSolution}
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
end


copy(s::OneMaxSolution) = deepcopy(s)


"""
    construct!(::OneMaxSolution, par, result)

Scheduler method that constructs a new random solution.
"""
function construct!(s::OneMaxSolution, par::Int, result::Result)
    initialize!(s)
end


"""
    local_improve!(::OneMaxSolution, par, result)

Scheduler method that tries to locally improve the solution.
"""
function local_improve!(s::OneMaxSolution, par::Int, result::Result)
    # does nothing here
end


"""
    shaking!(::OneMaxSolution, par, result)

Scheduler method that performs shaking by flipping par random positions.
"""
function shaking!(s::OneMaxSolution, par::Int, result::Result)
    k_random_flips!(s, par)
end

end  # module
