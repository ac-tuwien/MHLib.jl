# MISP.jl
#
# Demo problem: maximum (weighted) independent set problem (MISP).
#
# Give an undirected (weighted) graph, find a maximum cardinality subset of nodes where
# no pair of nodes is adjacent in the graph.

using Graphs
using Random
using StatsBase
using MHLib

export MISPInstance, MISPSolution, solve_misp


"""
    MISPInstance

Maximum (weighted) independent set problem (MISP) instance.

Give an undirected (weighted) graph, find a maximum cardinality subset of nodes where
no pair of nodes is adjacent in the graph.

# Elements
- `graph`: undirected unweighted graph to consider
- `n`: number of nodes
- `m` number of edges
- `p`: prices (weights) of items
"""
struct MISPInstance
    graph::SimpleGraph{Int}
    n::Int
    m::Int
    p::Vector{Int}
end

"""
    MISPInstance(name)

Create or read graph with given name.

So far we only create unweighted MISP instances here.
"""
function MISPInstance(name::AbstractString)
    graph = create_or_read_simple_graph(name)
    n = nv(graph)
    m = ne(graph)
    p = ones(Int, n)
    MISPInstance(graph, n, m, p)
end

function Base.show(io::IO, inst::MISPInstance)
    println(io, "n=$(inst.n), m=$(inst.m)")
end


"""
    MISPSolution

Solution to a MISP instance.

It is realized as a `SubsetVectorSolution`.

Attributes in addition to those needed by `SubsetVectorSolution`:
- `covered`: for each node the number of selected neighbor nodes plus one if the node
    itself is selected
"""
mutable struct MISPSolution <: SubsetVectorSolution{Int}
    inst::MISPInstance
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Int}
    sel::Int
    covered::Vector{Int}
end

MISPSolution(inst::MISPInstance) =
    MISPSolution(inst, -1, false, collect(1:inst.n), 0, zeros(Int, inst.n))

function Base.copy!(s1::MISPSolution, s2::MISPSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    copy!(s1.x, s2.x)
    s1.sel = s2.sel
    copy!(s1.covered, s2.covered)
end

Base.copy(s::MISPSolution) =
    MISPSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x), s.sel, copy(s.covered))

Base.show(io::IO, s::MISPSolution) =
    println(io, s.x[1:s.sel])

MHLib.calc_objective(s::MISPSolution) =
    s.sel > 0 ? sum(s.inst.p[s.x[1:s.sel]]) : 0

function MHLib.check(s::MISPSolution; kwargs...)
    invoke(check, Tuple{SubsetVectorSolution}, s; kwargs...)
    selected = Set(s.x[1:s.sel])
    for e in edges(s.inst.graph)
        if src(e) in selected && dst(e) in selected
            error("Invalid solution - adjacent nodes selected: $(src(e)), $(dst(e))")
        end
    end
    new_covered = zeros(Int, s.inst.n)
    for u in s.x[1:s.sel]
        new_covered[u] += 1
        for v in neighbors(s.inst.graph, u)
            new_covered[v] += 1
        end
    end
    if s.covered != new_covered
        error("Invalid covered values in solution: $(self.covered)")
    end
end

function Base.empty!(s::MISPSolution)
    fill!(s.covered, 0)
    invoke(Base.empty!, Tuple{SubsetVectorSolution}, s)
end

"""
    construct!(misp_solution, ::Nothing, result)

`MHMethod that constructs new solution by random initialization.
"""
function MHLib.construct!(s::MISPSolution, ::Nothing, result::Result)
    initialize!(s)
end

"""
    local_improve!(misp_solution, ::Nothing, result)

`MHMethod` that performs two-exchange local search followed by random fill.
"""
function MHLib.local_improve!(s::MISPSolution, ::Nothing, result::Result)
    if !two_exchange_random_fill_neighborhood_search!(s, false)
        result.changed = false
    end
end

"""
    shaking!(misp_solution, par, result)

`MHMethod` that performs shaking by removing `par` random elements followed by random fill.
"""
function MHLib.shaking!(s::MISPSolution, par::Int, ::Result)
    remove_randomly_selected!(s, par)
    fillup!(s)
end

MHLib.may_be_extendible(s::MISPSolution) = s.sel < length(s.x) && any(s.covered .== 0)

function MHLib.element_removed_delta_eval!(s::MISPSolution; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel+1]
    s.covered[u] -= 1
    for v in neighbors(s.inst.graph, u)
        s.covered[v] -= 1
    end
    if update_obj_val
        s.obj_val -= s.inst.p[u]
    end
    return true
end

function MHLib.element_added_delta_eval!(s::MISPSolution; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel]
    if allow_infeasible || s.covered[u] == 0
        # accept
        s.covered[u] += 1
        for v in neighbors(s.inst.graph, u)
            s.covered[v] += 1
        end
        if update_obj_val
            s.obj_val += s.inst.p[u]
        end
        return s.covered[u] == 1
    end
    # revert
    s.sel -= 1
    return false
end


# -------------------------------------------------------------------------------

"""
    solve_misp(filename::AbstractString; seed=nothing, titer=1000, kwargs...)

Solve a given MISP instance with a variable neighborhood search.

# Parameters
- `filename`: File name of the MISP instance
- `seed`: Possible random seed for reproducibility; if `nothing`, a random seed is chosen
- `titer`: Number of iterations for the solving algorithm, gets a new default value
- `kwargs`: Additional keyword arguments for the algorithm, e.g., `ttime`
"""
function solve_misp(
        filename::AbstractString=joinpath(@__DIR__, "..", "data", "frb40-19-1.mis");
        seed=nothing, titer=1000, kwargs...)
    # Make results reproducibly by either setting a given seed or picking one randomly
    isnothing(seed) && (seed = rand(0:typemax(Int32)))
    Random.seed!(seed)
    
    println("MISP Demo version $(git_version())")
    println("filename=$filename, seed=$seed, ", (; kwargs...))
   
    inst = MISPInstance(filename)
    sol = MISPSolution(inst)
    # initialize!(sol)
    # check(sol)
    # println(sol)

    # we apply a variable neighborhod search
    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)]; 
        consider_initial_sol=true, titer, kwargs...)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end

# To run from REPL, activate `MHLibDemos` environment, use `MHLibDemos`,
# and call e.g. `solve_misp(titer=200, seed=1)`.

# Run with profiler:
# @profview solve_misp()