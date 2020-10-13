export SubsetVectorSolution, clear!, initialize!, remove_some!,
    two_exchange_random_fill_neighborhood_search!, element_removed_delta_eval!,
    element_added_delta_eval!, may_be_extendible

"""
    SubsetVectorSolution

A type for solutions that are arbitrary cardinality subsets of a given set
represented in vector form. The front part represents the selected
elements, the back part the unselected ones.

A concrete type must implementthe following attributes:
- `x`: Vector of different elements, first the selected ones, then the not selected ones.
- `sel`: Integer indicating the number of selected elements
"""
abstract type SubsetVectorSolution{T} <: VectorSolution{T} end

function clear!(s::SubsetVectorSolution)
    s.sel = 0
    invalidate!(s)
end

"""
    sort_sel!(subset_vector_solution)

Sort selected elements in x.
"""
function sort_sel!(s::SubsetVectorSolution)
     if s.sel > 0
        sort!(view(s.x, 1:s.sel))
    end
end

"""
    fill!(subset_vector_solution, pool, random_order)

Scans elements from pool (by default in random order) and selects those whose inclusion is
feasible.

Elements in pool must not yet be selected.
Parameter pool must either be nothing, in which case x[sel+1:end] is used as pool,
or x[sel+1:_] for some _ > sel.
If random_order is set, the elements in the pool are processed in random order.
Uses element_added_delta_eval().
Reorders elements in pool so that the selected ones appear in pool[begin:return-value].
"""
function fill!(s::SubsetVectorSolution{T}, pool::Union{Vector{T}, Nothing},
        random_order::Bool=true) where {T}
    if !may_be_extendible(s)
        return 0
    end
    x = s.x
    if pool == nothing
        pool = get_extension_pool(s)
    end

    selected = 0
    for i in 1:length(pool)
        if random_order
            ir = rand(i:length(pool))
            if selected+1 != ir
                pool[selected+1], pool[ir] = pool[ir], pool[selected+1]
            end
        end
        s.sel += 1
        x[s.sel] = pool[selected+1]
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

"""
    remove_some!(subset_vector_solution, k)

Removes min(k,sel) randomly selected elements from the solution.

Uses element_removed_delta_eval, which should be overloaded and adapted to the problem.
The elements are removed even when the solution becomes infeasible.
"""
function remove_some!(s::SubsetVectorSolution, k::Int)
    x = s.x
    k = min(k, s.sel)
    if k > 0
        for i in 1:k
            j = rand(1:s.sel)
            if j != s.sel
                x[j], x[s.sel] = x[s.sel], x[j]
            end
            s.sel -= 1
            element_removed_delta_eval!(s, allow_infeasible=true)
        end
        sort_sel!(s)
    end
end

"""
    initialize!(subset_vector_solution)

Random construction of a new solution by applying fill to an initially empty solution.
"""
function initialize!(s::SubsetVectorSolution)
    clear!(s)
    fill!(s,nothing)
    invalidate!(s)
end

"""
    check(subset_vector_solution, unsorted)

Check correctness of solution; throw an exception if error detected.

- `unsorted`: if set, it is not checked if s is sorted
"""
function check(s::SubsetVectorSolution, unsorted::Bool=true)
    if !(1 <= s.sel <= length(s.x))
        error("Invalid attribute sel in solution: $(s.sel)")
    end
    if !allunique(s.x)
        error("Missing/equal elements in solution: $(s.x) (sorted: $(sort(s.x)))")
    end
    if !unsorted && !issorted(s.x)
        error("Solution not sorted: $(s.x[1:s.sel])")
    end
    if s.obj_val_valid
        old_obj = s.obj_val
        invalidate!(s)
        if old_obj != obj(s)
            error("Solution has wrong objective value: $(old_obj) should be $(obj(s))")
        end
    end
end

"""
    two_exchange_random_fill_neighborhood_search!(subset_vector_solution, best_improvement)

Search 2-exchange neighborhood followed by fill!() with random ordering.

Each selected location is tried to be exchanged with each unselected one followed by a
fill!().

The neighborhood is searched in a randomized fashion.
Overload the methods element_removed_delta_eval and element_added_delta_eval for an
efficient problem-specific delta evaluation.
Returns true if the solution could be improved, otherwise the solution remains unchanged.
"""
function two_exchange_random_fill_neighborhood_search!(s::SubsetVectorSolution,
        best_improvement::Bool)
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
        element_removed_delta_eval!(s, update_obj_val=true, allow_infeasible=true)
        obj1 = obj(s)
        pool = get_extension_pool(s)
        shuffle!(pool[1:end])

        # search v (the deleted item) and place it at the front of the extension pool
        #v_pos = findall(pool.==v)[1]
        #if v_pos != 1
        #    pool[1], pool[v_pos] = pool[v_pos], pool[1]
        #end
        # enumerate over all items in the extension pool except for v
        for (j, vu) in enumerate(pool[2:end])

            # increase selection range by one
            # and swap the element at the last position in the selction range
            # with a new element from the pool
            x[sel], pool[j+1] = vu, x[sel]
            s.sel += 1

            num_neighbors += 1
            if element_added_delta_eval!(s)
                # neighbor is feasible
                random_fill_applied = false
                if may_be_extendible(s)
                    self_backup = copy(s)
                    fill!(s, nothing)
                    random_fill_applied = true
                end
                if is_better(s,best)
                    # new best solution found
                    if !best_improvement
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
                element_removed_delta_eval!(s, update_obj_val=false, allow_infeasible=true)
                s.obj_val = obj1
            end
            x[sel], pool[j+1] = pool[j+1], vu
        end
        s.sel += 1
        element_added_delta_eval!(s,update_obj_val=false, allow_infeasible=true)
        s.obj_val = orig_obj
        if i != sel
            x[i], x[sel] = x[sel], x[i]
        end
    end
    if is_better_obj(s, obj(best), orig_obj)
        # return new best solution
        copy!(s, best)
        sort_sel!(s)
        return true
    end
    x[begin:sel] = x_sel_orig
    return false
end

"""
    get_extension_pool(subset_vector_solution)

Return a list of yet unselected elements that may possibly be added.
"""
function get_extension_pool(s::SubsetVectorSolution)
    return @view s.x[s.sel+1:end]
end

"""
    may_be_extensible(subset_vector_solution)

Quick check if the solution has chances to be extended by adding further elements.
"""
function may_be_extendible(s::SubsetVectorSolution)
    return s.sel < length(s.x)
end

"""
    element_removed_delta_eval!(subset_vector_solution)

Element x[sel] has been removed in the solution, if feasible update other solution data,
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
function element_removed_delta_eval!(s::SubsetVectorSolution; update_obj_val::Bool=true, allow_infeasible::Bool=false)
    if update_obj_val
        invalidate!(s)
    end
    return true
end

"""
    element_added_delta_eval!(subset_vector_solution)

Element x[sel-1] was added to a solution, if feasible update further solution data, else revert.

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
function element_added_delta_eval!(s::SubsetVectorSolution; update_obj_val::Bool=true, allow_infeasible::Bool=false)
    if update_obj_val
        invalidate!(s)
    end
    return true
end
