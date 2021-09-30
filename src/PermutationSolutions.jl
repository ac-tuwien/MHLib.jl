"""
    PermutationSolutions

A module for solutions that are represented by a permutation of distinct elements.
"""
module PermutationSolutions

using Random
using ..MHLib

export PermutationSolution, initialize!, two_opt_neighborhood_search!, 
    random_two_exchange_move!

"""
    PermutationSolution

A type for solutions that are represented by a permutation of distinct elements.

A concrete type must implement the attributes of a vector solution.
"""
abstract type PermutationSolution{T} <: VectorSolution{T} end

"""
    initialize!(permutation_solution)

Random construction of a new solution by applying fill to an initially empty solution.
"""
function MHLib.initialize!(s::PermutationSolution)
    shuffle!(s.x)
    invalidate!(s)
end

"""
    check(permutation_solution)

Check correctness of permutation solution.
"""
function MHLib.check(s::PermutationSolution{T}) where T
    !allunique(s.x) && error("PermutationSolution is no permutation: $(s.x)")
    invoke(check, Tuple{VectorSolution{T}}, s)
end

"""
    two_opt_neighborhood_search(permutation_solution, best_improvement)

Systematic search of the 2-opt neighborhood, i.e., consider all inversions of subsequences.

The neighborhood is searched in a randomized ordering. Boolean Parameter best_improvement 
defines whether best improvement or next improvement step functions is used. Returns true 
if a better solution has been found.
"""
function two_opt_neighborhood_search!(s::PermutationSolution, best_improvement::Bool)
    order = randperm(length(s))

    best_delta = 0
    best_p1 = nothing
    best_p2 = nothing

    for (idx, p1) in enumerate(order[1:end-1])
        for p2 in order[idx+1:end]
            pa, pb = p1 < p2 ? (p1, p2) : (p2, p1)
            delta = two_opt_move_delta_eval(s, pa, pb)
            if is_better_obj(s, delta, best_delta)
                if !best_improvement
                    apply_two_opt_move!(s, pa, pb)
                    s.obj_val += delta
                    return true
                end
                best_delta = delta
                best_p1 = pa
                best_p2 = pb
            end
        end
    end

    if !isnothing(best_p1)
        apply_two_opt_move!(s, best_p1, best_p2)
        s.obj_val += delta
        return true
    end

    false
end

"""
    apply_two_opt_move(permutation_solution, p1, p2)

Perform two-opt move on given solution defined as inversion of subsequence starting from 
    position p1 up to and including position p2.
"""
function apply_two_opt_move!(s::PermutationSolution, p1::Integer, p2::Integer)
    @assert 1 <= p1 <= p2 <= length(s)
    reverse!(s.x, p1, p2)
end

"""
    two_opt_move_delta_eval(permutation_solution, p1, p2)

Return the delta in the objective value when inverting s.x from position p1 to position p2.

The function returns the difference in the objective function if the move would be performed,
the solution, however, is not changed.
This function should be overwritten in a concrete class.
Here we actually perform a less efficient complete evaluation of the modified solution.
"""
function two_opt_move_delta_eval(s::PermutationSolution, p1::Integer, p2::Integer)
    orig_obj = s.obj_val
    apply_two_opt_move!(s, p1, p2)
    invalidate!(s)
    delta = obj(s) - orig_obj
    apply_two_opt_move!(s, p1, p2)
    s.obj_val = orig_obj
    delta
end

"""
    random_two_exchange_move!(permutation_solution, p1, p2)

Perform a random two exchange move, invalidating the solution.
"""
function random_two_exchange_move!(s::PermutationSolution)
    p1, p2 = randperm(length(s))
    s.x[p1], s.x[p2] = s.x[p2], s.x[p1]
    invalidate!(s)
end

end  # module