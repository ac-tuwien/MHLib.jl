"""
    OneMax

OneMax demo problem: Maximize the number of set bits in a binary string.

This problem is just for simple demonstration/debugging purposes.
"""
module OneMax

using ArgParse

using MHLib
using MHLib.Schedulers

export OneMaxSolution


const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--onemax_n"
        help = "length of solution string in the problem"
        arg_type = Int
        default = 100
end


"""
    OneMaxSolution

A concrete solution type to solve the MAXSAT problem.
"""
mutable struct OneMaxSolution <: BoolVectorSolution
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Bool}
    OneMaxSolution(n) = new(-1, false, Vector{Bool}(undef, n))
    OneMaxSolution(s::OneMaxSolution) =
        new(s.obj_val, s.obj_val_valid, copy(s.x))
end

MHLib.calc_objective(s::OneMaxSolution) = sum(s.x)

Base.show(io::IO, s::OneMaxSolution) =
    println(io, s.x)

function Base.copy!(s1::OneMaxSolution, s2::OneMaxSolution)
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
end

Base.copy(s::OneMaxSolution) = deepcopy(s)


end  # module
