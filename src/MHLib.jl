"""
    MHLib

`MHLib` - A Toolbox for Metaheuristics and Hybrid Optimization Methods.
"""
module MHLib

using Random
using Base: copy, copy!, length

export Solution, to_maximize, obj, calc_objective, invalidate!, is_equal,
    is_better, is_worse, is_better_obj, is_worse_obj, dist, check,
    run!, git_version,

    # settings
    settings

#----------------------------- Solution ------------------------------

"""
    Solution

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


print(s::Solution) = error("abstract string(s) called")


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

export VectorSolution, copy!, length

"""
    VectorSolution

An abstract solution encoded by a vector of type `T`.

Concrete subtypes need to implement:
- all requirements of the supertype `Solution`
- `x::AbstractVector`: vector representing the solution
"""
abstract type VectorSolution{T} <: Solution end


"""
    length(::VectorSolution)

Length of the solution vector.
"""
Base.length(s::VectorSolution) = length(s.x)


is_equal(s1::VectorSolution, s2::VectorSolution) =
    obj(s1) == obj(s2) && s1.x == s2.x


#----------------------------- BoolVectorSolution ------------------------------

export BoolVectorSolution, initialize!, k_random_flips!, k_flip_neighborhood_search!,
    flip_variable!


"""
    BoolVectorSolution

An abstract solution encoded by a fixed-length boolean vector.
"""
abstract type BoolVectorSolution <: VectorSolution{Bool} end


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
        p = rand(1:length(s))
        s.x[p] = !s.x[p]
    end
    invalidate!(s)
end


"""
    k_flip_neighborhood_search!(::BoolVectorSolution, k::Int, best_improvement::Bool)

Perform one major iteration of a k-flip local search, i.e., search one neighborhood.

If `best_improvement` is set, the neighborhood is completely searched and a best neighbor is
kept; otherwise the search terminates in a first-improvement manner, i.e., keeping a first
encountered better solution. The new bjective value is returned.
"""
function k_flip_neighborhood_search!(s::BoolVectorSolution, k::Int, best_improvement::Bool)
    len_x = length(s.x)
    @assert 0 < k <= len_x
    better_found = false
    obj_val = obj(s)
    best_sol = copy(s.x)
    best_obj = obj_val
    perm = randperm(len_x)  # random permutation for randomizing enumeration order
    p = fill(-1, k)  # flipped positions
    # initialize
    i = 1  # current index in p to consider
    while i >= 1
        # evaluate solution
        if i == k + 1
            if obj_val > best_obj
                if !best_improvement
                    return true
                end
                best_sol[:] = s.x
                best_obj = obj_val
                better_found = true
            end
            i -= 1  # backtrack
        else
            if p[i] == -1
                # this index has not yet been placed
                p[i] = (i>1 ? p[i-1] : 0) + 1
                obj_val = flip_variable!(s, perm[p[i]])
                i += 1  # continue with next position (if any)
            elseif p[i] < len_x - (k - i)
                # further positions to explore with this index
                obj_val = flip_variable!(s, perm[p[i]])
                p[i] += 1
                obj_val = flip_variable!(s, perm[p[i]])
                i += 1
            else
                # we are at the last position with the i-th index, backtrack
                obj_val = flip_variable!(s, perm[p[i]])
                p[i] = -1  # unset position
                i -= 1
            end
        end
    end
    if better_found
        s.x[:] = best_sol
        obj_val = best_obj
    end
    obj_val
end


"""
    flip_variable!(::BoolVectorSolution, pos::Int)

Flip the value at the given position and return new objective value.

Should be overloaded with a problem-specific implementation realizing incremental
evaluation if possible.
"""
function flip_variable!(s::BoolVectorSolution, pos::Int)
    s.x[pos] = !s.x[pos]
    invalidate!(s)
    obj(s)
end


"""
    run!()

General function for performing an optimization algorithm in MHLib.
"""
function run!
end

"""
    git_version()

Return git version information of current directory.
"""
function git_version() :: String
    chomp(read(`git describe --abbrev=4 --dirty --always --tags`, String))
end

#-----------------------------------------------------------

include("settings.jl")
include("Schedulers.jl")
include("GVNSs.jl")
include("ALNSs.jl")
include("Environments.jl")
include("MCTSs.jl")

include("demos/OneMax.jl")
include("demos/MAXSAT.jl")
include("demos/LCS.jl")

const all_settings_cfgs = [
        Schedulers.settings_cfg,
        ALNSs.settings_cfg,
        MCTSs.settings_cfg,
        OneMax.settings_cfg,
        LCS.settings_cfg,
    ]


end # module
