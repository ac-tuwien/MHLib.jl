"""
    MISP

Demo application solving the maximum (weighted) independent set problem (MISP).

Give an undirected (weighted) graph, find a maximum cardinality subset of nodes where
no pair of nodes is adjacent in the graph.
"""
module MISP

using Random
using StatsBase
using LightGraphs

using ..Graphs

using MHLib
using MHLib.Schedulers
import MHLib.Schedulers: construct!, local_improve!, shaking!
import MHLib: calc_objective, element_removed_delta_eval!, element_added_delta_eval!,
    may_be_extendible, check

import Base: copy, copy!, show

export MISPInstance, MISPSolution


"""
    MISPInstance

Maximum (weighted) independent set problem (MISP) instance.

Give an undirected (weighted) graph, find a maximum cardinality subset of nodes where
no pair of nodes is adjacent in the graph.

Attributes
- `graph`: undirected unweighted graph to consider
- `n`: number of nodes
- `m` number of edges
- `p`: prices (weights) of items
"""
struct MISPInstance
    graph
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

function show(io::IO, inst::MISPInstance)
    println(io, "n=$(inst.n), m=$(inst.m)")
end


"""
    MISPSolution

Solution to a MISP instance.

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

function copy!(s1::S, s2::S) where {S <: MISPSolution}
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
    s1.sel = s2.sel
    s1.covered[:] = s2.covered
end

copy(s::MISPSolution) =
    MISPSolution(s.inst, s.obj_val, s.obj_val_valid, Base.copy(s.x[:]), s.sel,
        Base.copy(s.covered[:]))

Base.show(io::IO, s::MISPSolution) =
    println(io, "MISP Solution: ", s.x)

calc_objective(s::MISPSolution) =
    s.sel > 0 ? sum(s.inst.p[s.x[1:s.sel]]) : 0

function check(s::MISPSolution, unsorted::Bool=true)
    invoke(check, Tuple{SubsetVectorSolution, Bool}, s, unsorted)
    selected = Set(s.x[1:s.sel])
    for e in edges(s.inst.graph)
        if src(e) in selected && dst(e) in selected
            error("Invalid solution - adjacent nodes selected: $(src(e)), $(src(v))")
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

function clear(s::MISPSolution)
    fill!(s.covered, nothing)
    invoke(clear, Tuple{SubsetVectorSolution}, s)
end

"""
    construct!(misp_solution, par, result)

Construct new solution by random initialization.
"""
function construct!(s::MISPSolution, par::Int, result::Result)
    initialize!(s)
end

"""
    local_improve!(misp_solution, par, result)

Perform two-exchange local search followed by random fill.
"""
function local_improve!(s::MISPSolution, par::Int, result::Result)
    if !two_exchange_random_fill_neighborhood_search!(s, false)
        result.changed = false
    end
end

"""
    shaking!(misp_solution, par, result)

Perform shaking by removing par randomly selected elements followed ba a random fill.
"""
function shaking!(s::MISPSolution, par::Int, result::Result)
    remove_some!(s, par)
    fill!(s, nothing)
end

"""
    may_be_extendible(misp_solution)

Quick check if the solution may possibly be extended by adding further elements.
"""
may_be_extendible(s::MISPSolution) =
    any(s.covered .== 0)

function element_removed_delta_eval!(s::MISPSolution; update_obj_val::Bool=true,
        allow_infeasible::Bool=false)
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

function element_added_delta_eval!(s::MISPSolution; update_obj_val::Bool=true,
        allow_infeasible::Bool=false)
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

end  # module
