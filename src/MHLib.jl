
"""MHLib

A Toolbox for Metaheuristics and Hybrid Optimization Methods
"""
module MHLib

using StaticArrays
using Random

export greet

greet() = print("Hello World!")

export Solution, to_maximize, obj!, invalidate!, is_better, is_worse,
    is_better_obj, is_worse_obj, dist, check

"""Solution

An abstract solution to an optimization problem.

Concrete subtypes need to implement:
- `obj_val`: objective value of solution, usually some numerical type
- `obj_val_valid::Bool`: indicates if obj_val is valid
- `calc_objective(::Solution): calculate and return objective value of solution
- `to_maximize(::Type)`: for minimization this method must return false
- `copy!(::Solution, ::Solution)`: make first solution a copy of the second
- `copy(::Solution)`: return an independent copy of the solution
"""
abstract type Solution end

"""to_maximize(::Type)

Return true if the optimization goal is to maximize the objective function
in the given type of solutions.

This default implementation returns true.
"""
to_maximize(::Type) = true
to_maximize(s::Solution) = to_maximize(typeof(s))

"""obj!(::Solution)

Return obj_val if obj_val_valid or calculate it via objective(::Solution).
"""
obj!(s::Solution) = s.obj_val_valid ? s.obj_value : calc_objective(s)

"""invalidate!(::Solution)

Invalidate a possibly cached objective value, usually due to a change in the solution.
"""
invalidate!(solution::Solution) = obj_val_valid = false;

"""is_better(::Solution, ::Solution)

Return true if the first solution is better than the second.
"""
function is_better(s1::S, s2::S) where S <: Solution
    to_maximize(s1) ? obj!(s1) > obj!(s2) : obj!(s1) < obj!(s2)
end

"""is_worse(::Solution, ::Solution)

Return true if the first solution is worse than the second.
"""
function is_worse(s1::S, s2::S) where S <: Solution
    to_maximize(s1) ? obj!(s1) < obj!(s2) : obj!(s1) > obj!(s2)
end

"""is_better_obj(::Solution, obj1, obj2)

Return true if obj1 is a better objective value than obj2 in
the given solution type.
"""
function is_better_obj(s::Solution, obj1, obj2)
    to_maximize(s) ? obj1 > obj2 : obj1 < obj2
end

"""is_worse_obj(::Solution, obj1, obj2)

Return true if obj1 is a worse objective value than obj2 in
the given solution type.
"""
function is_worse_obj(s::Solution, obj1, obj2)
    to_maximize(s) ? obj1 < obj2 : obj1 > obj2
end

"""distthe distance of s1 to s2

Return true if the first solution is worse than the second.
The default implementation just returns false if the solutions are the same
and true otherwise.

"""
dist(s1::S, s2::S) where S <: Solution = s1 != s2

"""Check validity of solution.

If a problem is encountered, raise an exception.
The default implementation just re-calculates the objective value.
"""
function check(s::Solution)::Nothing
    if s.obj_val_valid
        old_obj = self.obj_val
        invalidate!(s)
        if old_obj != s.obj!()
            error("Solution has wrong objective value: $old_obj, " *
                  "should be $(s.obj!())")
        end
    end
end


export VectorSolution, copy!, len

"""VectorSolution

An abstract solution encoded by a vector of length `N` and type `T`.

Concrete subtypes need to implement:
- all requirements of the supertype `Solution`
- `x::AbstractVector`: vector representing the solution
"""
abstract type VectorSolution{N,T} <: Solution end

"""copy!(::Solution, ::Solution)

Make first solution an independent copy of the second solution.
"""
function copy!(s1::S, s2::S) where S <: VectorSolution
      # copy!(s1, s2)
end

"""len(::VectorSolution)

Length of the solution vector.
"""
len(s::VectorSolution) = length(s.x)


export BoolVectorSolution

"""BoolVectorSolution

An abstract solution encoded by a fixed-length boolean vector.
"""
abstract type BoolVectorSolution{N} <: VectorSolution{N,Bool} end

initialize!(s::BoolVectorSolution) = ( rand!(s.x); invalidate!(s) )


"""OneMaxSolution

A concrete solution type to solve the MAXSAT problem.
"""
mutable struct OneMaxSolution{N} <: BoolVectorSolution{N}
    obj_val::Int
    obj_val_valid::Bool
    x::MVector{N,Bool}
    OneMaxSolution{N}() where {N} = new{N}(-1, false, MVector{N,Bool}(undef))
    OneMaxSolution{N}(s::OneMaxSolution{N}) where {N} =
        new{N}(s.obj_val, s.obj_val_valid, MVector{N,Bool}(s.x))
end

calc_objective(s::OneMaxSolution) = sum(s.x)

function copy!(s1::S, s2::S) where {S <: OneMaxSolution}
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
end

copy(s::OneMaxSolution) = deepcopy(s)


include("MAXSAT.jl")




s = OneMaxSolution{5}()
initialize!(s)
print("$s, $(obj!(s))")

end # module
