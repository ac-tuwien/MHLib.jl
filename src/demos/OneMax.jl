module OneMax

import Base: copy, copy!
import MHLib: BoolVectorSolution, calc_objective

export OneMaxSolution


"""OneMaxSolution

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

end  # module
