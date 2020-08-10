"""
    MHLib

`MHLib` - A Toolbox for Metaheuristics and Hybrid Optimization Methods.
"""
module MHLib

using Random
using Base: copy, copy!, length

export Solution, to_maximize, obj, calc_objective, invalidate!, is_equal,
    is_better, is_worse, is_better_obj, is_worse_obj, dist, check,

    # ALNSs
    get_number_to_destroy,

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


#----------------------------- SubsetVectorSolution ------------------------------
export SubsetVectorSolution, clear!, initialize!, fill!, remove_some!, two_exchange_random_fill_neighborhood_search!

"""
    SubsetVectorSolution

A generic class for solutions that are arbitrary cardinality subsets of a given set represented in vector form.

"""
abstract type SubsetVectorSolution{T} <: VectorSolution{T} end

function clear!(s::SubsetVectorSolution)
    s.sel = 0
    invalidate!(s)
end

"""Scans elements from pool (by default in random order) and selects those whose inclusion is feasible.

Elements in pool must not yet be selected.
Parameter pool must either be None, in which case x[sel:end] is used as pool,
or x[sel:_] for some _ > sel.
If random_order is set, the elements in the pool are processed in random order.
Uses element_added_delta_eval()
Reorders elements in pool so that the selected ones appear in pool[begin:return-value].
"""
function fill!(s::SubsetVectorSolution{T}, pool::Vector{T}, random_order::Bool=true) where {T}
    if !may_be_extendible(s)
        return 0
    end
    x = s.x
    if pool == nothing
        pool = x[s.sel+1:end]
    end

    selected = 1
    for i in 1:len(pool)
        if random_order
            ir = rand(i:length(pool))
            if selected != ir
                pool[selected], pool[ir] = pool[ir], pool[selected]
            end
        end
        s.sel += 1
        x[s.sel] = pool[selected]
        if element_added_delta_eval!(s)
            selected += 1
            if !may_be_extendible(s)
                break
            end
        end
    end
    if selected > 0
        sort_sel!(s)
    end
    return selected
end

"""Removes min(k,sel) randomly selected elements from the solution.

Uses element_removed_delta_eval, which should be overloaded and adapted to the problem.
The elements are removed even when the solution becomes infeasible.
"""
function remove_some!(s::SubsetVectorSolution, k::Int)
    x = s.x
    k = min(k, s.sel)
    if k > 0
        for i in 1:k
            j = rand(1:s.sel)
            s.sel -= 1
            if j != s.sel
                x[j], x[s.sel] = x[s.sel], x[j]
            end
            element_removed_delta_eval!(s,allow_infeasible=true)
        end
        sort_sel!(s)
    end
end

"""Random construction of a new solution by applying fill to an initially empty solution."""
function initialize!(s::SubsetVectorSolution, k::Int)
    clear!(s)
    fill!(s,s.all_elements)
    invalidate!(s)
end

"""Check correctness of solution; throw an exception if error detected.

:param unsorted: if set, it is not checked if s is sorted
"""
function check(s::SubsetVectorSolution, unsorted::Bool=false)
    length = length(s.all_elements)
    if !(1 <= s.sel <= length)
        error("Invalid attribute sel in solution: $(s.sel)")
    end
    if length(s.x) != length
        error("Invalid length of solution array x: $(s.x)")
    end
    if Set(s.x) != s.all_elements
        error("Invalid solution - x is not a permutation of V: $(s.x) (sorted: $(sorted!(s.x)))")
    else
        sol_set = Set(s.x[begin:s.sel])
        if !issubset(sol_set, s.all_elements) || length(sol_set) != s.sel
            error("Solution not simple subset of V: $(s.x[begin:s.sel]), $(s.all_elements)")
        end
    end
    if !unsorted
        old_v = s.x[1]
        for v in s.x[2:s.sel]
            if v <= old_v
                error("Solution not sorted: value $(v) in $(s.x)")
            end
            old_v = v
        end
    end

    if s.obj_val_valid
        old_obj = s.obj_val
        invalidate!(s)
        if old_obj != obj(s)
            error("Solution has wrong objective value: $(old_obj) should be $(obj(s))")
        end
    end
end

"""Search 2-exchange neighborhood followed by fill!() with random ordering.

Each selected location is tried to be exchanged with each unselected one followed by a fill().

The neighborhood is searched in a randomized fashion.
Overload the methods element_removed_delta_eval and element_added_delta_eval for an efficient problem-specific
delta evaluation.
Returns true if the solution could be improved, otherwise the solution remains unchanged.
"""
function two_exchange_random_fill_neighborhood_search!(s::SubsetVectorSolution, best_improvement::Bool)
    sel = s.sel
    x = s.x
    orig_obj = obj(s)
    self_backup = nothing
    x_sel_orig = copy(x[begin:sel])
    shuffle!(x[begin:sel])
    best = copy(s)
    num_neighbors = 0
    for (i, v) in enumerate(x[begin:sel])
        # move v at the end of the selection range
        if i != sel
            x[i], x[sel] = x[sel], x[i]
        end
        # delete v by reducing the selection range by one
        s.sel -= 1
        element_removed_delta_eval!(s,allow_infeasible=True)
        obj1 = obj(s)
        pool = get_extension_pool(s)
        shuffle!(pool)

        # search v (the deleted item) and place it at the front of the extension pool
        v_pos = findall(x->x==v)[1]
        if v_pos != 1
            pool[1], pool[v_pos] = pool[v_pos], pool[1]
        end

        # enumerate over all items in the extension pool except for v
        for (j, vu) in enumerate(pool[2:end])

            # increase selection range by one
            # and swap the element at the last position in the selction range
            #with a new element from the pool
            x[sel], pool[j+1] = vu, x[sel]
            s.sel += 1

            num_neighbors += 1
            if element_added_delta_eval(s)
                # neighbor is feasible
                random_fill_applied = false
                if may_be_extendible(s)
                    self_backup = copy(s)
                    fill!(get_extension_pool(s))
                    random_fill_applied = true
                end
                if is_better(s,best)
                    # new best solution found
                    if not best_improvement
                        sort_sel!(s)
                        return true
                    end
                    copy!(s, best)
                end
                if random_fill_applied
                    if i != s.sel
                        x[i], x[sel] = x[sel], x[i]
                    end
                    copy!(s, self_backup)
                end
                s.sel -= 1
                element_removed_delta_eval(s, update_obj_val=false, allow_infeasible=true)
                s.obj_val = obj1
            end
            x[sel], pool[j+1] = pool[j+1], vu
        end
        s.sel += 1
        element_added_delta_eval(s,update_obj_val=false, allow_infeasible=true)
        s.obj_val = orig_obj
        if i != sel-1
            x[i], x[sel] = x[sel], x[i]
        end
    end
    if is_better_obj(s, best.obj(), orig_obj)
        # return new best solution
        copy!(s, best)
        sort_sel!(s)
        return true
    end
    x[begin:sel] = x_sel_orig
    return false
end

"""Return a list of yet unselected elements that may possibly be added."""
function get_extension_pool(s::SubsetVectorSolution)
    return s.x[s.sel+1:end]
end

"""Quick check if the solution has chances to be extended by adding further elements."""
function may_be_extendible(s::SubsetVectorSolution)
    return s.sel < length(s.x)
end

"""Element x[sel] has been removed in the solution, if feasible update other solution data,
else revert.

This is a helper function for delta-evaluating solutions when searching a neighborhood that needs
    to be overloaded for a concrete problem.
        It can be assumed that the solution was in a correct state with a valid objective value in obj_val
        *before* the already applied move, obj_val_valid therefore is True.
        The default implementation just calls invalidate() and returns True.

        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
        the update of other data done
        :return: True if feasible, False if infeasible
        """
function element_removed_delta_eval(s::SubsetVectorSolution, update_obj_val=rrue, allow_infeasible=false)
    if update_obj_val
        invalidate(s)
    end
    return true
end

"""Element x[sel-1] was added to a solution, if feasible update further solution data, else revert.

This is a helper function for delta-evaluating solutions when searching a neighborhood that needs
    to be overloaded for a concrete problem.
        It can be assumed that the solution was in a correct state with a valid objective value in obj_val
        *before* the already applied move, obj_val_valid therefore is True.
        The default implementation just calls invalidate() and returns True.

        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
        the update of other data done
        :return: True if feasible, False if infeasible
        """
function element_added_delta_eval(s::SubsetVectorSolution, update_obj_val=true, allow_infeasible=false)
    if update_obj_val
        invalidate(s)
    end
    return true
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
