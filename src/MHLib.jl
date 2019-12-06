"""
`MHLib` - A Toolbox for Metaheuristics and Hybrid Optimization Methods.
"""
module MHLib

using Random
using Base: copy, copy!

export Solution, to_maximize, obj, calc_objective, invalidate!, is_equal,
    is_better, is_worse, is_better_obj, is_worse_obj, dist, check


#----------------------------- Solution ------------------------------

"""
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


"""
    to_maximize(::Type)

Return true if the optimization goal is to maximize the objective function
in the given type of solutions.

This default implementation returns true.
"""
to_maximize(::Type) = true
to_maximize(s::Solution) = to_maximize(typeof(s))


"""
    obj(::Solution)

Return obj_val if obj_val_valid or calculate it via objective(::Solution).
"""
function obj(s::Solution)
    if !s.obj_val_valid
        s.obj_val = calc_objective(s)
        s.obj_val_valid = true
    end
    s.obj_val
end


"""
    calc_objective(::Solution)

Actually calculate the objective value of the given solution.
"""
calc_objective(::Solution) =
    error("calc_objective not implemented for concrete solution")


"""
    invalidate!(::Solution)

Invalidate a possibly cached objective value, usually due to a change in the solution.
"""
invalidate!(s::Solution) = s.obj_val_valid = false;


"""
    is_equal(::Solution, ::Solution)

Return true if the two solutions are considered equal and false otherwise.

The default implementation just checks if the objective values are the same.
"""
is_equal(s1::Solution, s2::Solution) = obj(s1) == obj(s2)


"""
    is_better(::Solution, ::Solution)

Return true if the first solution is better than the second.
"""
function is_better(s1::S, s2::S) where S <: Solution
    to_maximize(s1) ? obj(s1) > obj(s2) : obj(s1) < obj(s2)
end


"""
    is_worse(::Solution, ::Solution)

Return true if the first solution is worse than the second.
"""
function is_worse(s1::S, s2::S) where S <: Solution
    to_maximize(s1) ? obj(s1) < obj(s2) : obj(s1) > obj(s2)
end


"""
    is_better_obj(::Solution, obj1, obj2)

Return true if obj1 is a better objective value than obj2 in
the given solution type.
"""
function is_better_obj(s::Solution, obj1, obj2)
    to_maximize(s) ? obj1 > obj2 : obj1 < obj2
end


"""
    is_worse_obj(::Solution, obj1, obj2)

Return true if obj1 is a worse objective value than obj2 in
the given solution type.
"""
function is_worse_obj(s::Solution, obj1, obj2)
    to_maximize(s) ? obj1 < obj2 : obj1 > obj2
end


"""
    dist(::Solution, ::Solution)

Return distance of the two solutions.

The default implementation just returns 0 when the objective values
are the same and 1 otherwise.

"""
dist(s1::S, s2::S) where S <: Solution = obj(s1) == obj(s2) ? 0 : 1


"""
    check(::Solution)

Check validity of solution.

If a problem is encountered, raise an exception.
The default implementation just re-calculates the objective value.
"""
function check(s::Solution)::Nothing
    if s.obj_val_valid
        old_obj = s.obj_val
        invalidate!(s)
        if old_obj != obj(s)
            error("Solution has wrong objective value: $old_obj, " *
                  "should be $(obj(s))")
        end
    end
end


#----------------------------- VectorSolution ------------------------------

export VectorSolution, copy!, len

"""
An abstract solution encoded by a vector of length `N` and type `T`.

Concrete subtypes need to implement:
- all requirements of the supertype `Solution`
- `x::AbstractVector`: vector representing the solution
"""
abstract type VectorSolution{N,T} <: Solution end


"""
    len(::VectorSolution)

Length of the solution vector.
"""
len(s::VectorSolution) = length(s.x)


is_equal(s1::VectorSolution, s2::VectorSolution) =
    obj(s1) == obj(s2) && s1.x == s2.x


#----------------------------- BoolVectorSolution ------------------------------

export BoolVectorSolution, initialize!, k_random_flips!

"""
An abstract solution encoded by a fixed-length boolean vector.
"""
abstract type BoolVectorSolution{N} <: VectorSolution{N,Bool} end


"""
    initialize!(::BoolVectorSolution)

Initializes the given solution randomly.
"""
initialize!(s::BoolVectorSolution) = ( rand!(s.x); invalidate!(s) )


"""
    dist(::BoolVectorSolution, ::BoolVectorSolution)

Return Hamming distance.
"""
dist(s1::BoolVectorSolution, s2::BoolVectorSolution) = sum(abs.(s1.x - s2.x))


"""
    k_random_flips!(::BoolVectorSolution, k)

Perform k random flips and call invalidate.
"""
function k_random_flips!(s::BoolVectorSolution, k::Int)
    for i in 1:k
        p = rand(1:len(s))
        s.x[p] = !s.x[p]
    end
    invalidate!(s)
end

#-----------------------------------------------------------

include("settings.jl")
include("Schedulers.jl")
include("GVNSs.jl")

include("demos/OneMax.jl")

end # module
