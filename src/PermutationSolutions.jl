export PermutationSolution, clear!, initialize!

"""
    PermutationSolution

A type for solutions that are represented by a permutation of distinct elements.

A concrete type must implement the attributes of a vector solution and a fixed length n
    The permutation of the elements is defined by the ordering in the vector.
- `n`: Number of permuted elements
"""
abstract type PermutationSolution{T} <: VectorSolution{T} end

function clear!(s::PermutationSolution)
    sort!(s.x)
    invalidate!(s)
end

"""
    initialize!(permutation_solution)

Random construction of a new solution by applying fill to an initially empty solution.
"""
function initialize!(s::PermutationSolution)
    clear!(s)
    shuffle!(s.x)
    invalidate!(s)
end

"""
    check(permutation_solution)

Check correctness of permutation solution
"""
function check(s::PermutationSolution)
    length(s.x) == s.n && allunique(s.x)
end
