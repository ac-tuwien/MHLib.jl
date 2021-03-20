"""
    GraphColoring

Demo application solving the graph coloring problem.

Given a graph and an number of colors, color each node with one color so that
the number of adjacent nodes having the same color is minimized.
"""
module GraphColoring

using ArgParse
using Random
using StatsBase
using LightGraphs

using MHLib
using MHLib.Schedulers
using ..Graphs

export GraphColoringInstance, GraphColoringSolution


const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--gcp_colors"
        help = "number of colors for the graph coloring problem"
        arg_type = Int
        default = 3
end


"""
    GraphColoringInstance

Graph coloring problem instance.

Given a graph and a number of colors, color each node with one color so that
the number of adjacent nodes having the same color is minimized.

Attributes
- `graph`: undirected graph we want to color
- `n`: number of nodes
- `m` number of edges
- `colors`: number of colors
"""
struct GraphColoringInstance
    graph::SimpleGraph{Int}
    n::Int
    m::Int
    colors::Int
end

"""
    GraphColoringInstance(name)

Create or read graph with given name.
"""
function GraphColoringInstance(name::AbstractString)
    graph = create_or_read_simple_graph(name)
    n = nv(graph)
    m = ne(graph)
    colors = settings[:gcp_colors]
    GraphColoringInstance(graph, n, m, colors)
end

function show(io::IO, inst::GraphColoringInstance)
    println(io, "n=$(inst.n), m=$(inst.m)")
end



"""
    GraphColoringSolution

Solution to a Graph Coloring instance.

It is a `VectorSolution{Int}`.

Attributes
- `x`: array with the assigned colors
"""
mutable struct GraphColoringSolution <: VectorSolution{Int}
    inst::GraphColoringInstance
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Int}
end

MHLib.to_maximize(::GraphColoringSolution) = false

GraphColoringSolution(inst::GraphColoringInstance) =
    GraphColoringSolution(inst, -1, false, fill(1, inst.n))

function Base.copy!(s1::GraphColoringSolution, s2::GraphColoringSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
end

Base.copy(s::GraphColoringSolution) =
    GraphColoringSolution(s.inst, s.obj_val, s.obj_val_valid, Base.copy(s.x[:]))

Base.show(io::IO, s::GraphColoringSolution) =
    println(io, s.x)

function MHLib.calc_objective(s::GraphColoringSolution)
    violations = 0
    for e in edges(s.inst.graph)
        if s.x[src(e)] == s.x[dst(e)]
            violations += 1
        end
    end
    return violations
end


"""
    check(s::GraphColoringSolution)

Check if s is a valid solution.
Raises an error if a problem is detected.
"""
function MHLib.check(s::GraphColoringSolution)
    if length(s.x) != s.inst.n
        error("Invalid length of solution")
    end
    if sum(s.x .> s.inst.colors) >= 1
        error("Too many colors used")
    end
    invoke(check, Tuple{supertype(typeof(s))}, s)
end


"""
    construct!(s::GraphColoringSolution, par, result)

Constructs a new solution. Here we just call initialize!.
"""
function MHLib.Schedulers.construct!(s::GraphColoringSolution, par::Int, result::Result)
    initialize!(s)
end



"""
    local_improve!(s::GraphColoringSolution, par, result)

Performs one iteration of a local search following a first improvement strategy.
The neighborhood used is defined by all solutions that can be created by changing the color
of a vertex involved in a conflict.
"""
function MHLib.Schedulers.local_improve!(s::GraphColoringSolution, par::Int, result::Result)
    n = length(s.x)
    order = sample(1:n, n, replace = false)
    for p in order
        # Count colors in the neighborhood
        nbh_col = fill(0, s.inst.colors)
        for v in neighbors(s.inst.graph, p)
            nbh_col[s.x[v]] += 1
        end
        old_col = s.x[p]
        if nbh_col[old_col] > 0
            # violation found
            for new_col in sample(1:s.inst.colors, s.inst.colors, replace = false)
                if nbh_col[new_col] < nbh_col[old_col]
                    # Possible improvement found
                    s.x[p] = new_col
                    s.obj_val -= nbh_col[old_col]
                    s.obj_val += nbh_col[new_col]
                    result.changed = true
                    return
                end
            end
        end
    end
    result.changed = false
end



"""
    shaking!(s::GraphColoringSolution, par, result)

Perform shaking by randomly assigning a different color
to 'par' many random vertices that are involved in conflicts.
"""
function MHLib.Schedulers.shaking!(s::GraphColoringSolution, par::Int, result::Result)
    under_conflict = Vector{Int}()
    result.changed = false

    for u in 1:length(s.x)
        for v in neighbors(s.inst.graph, u)
            if s.x[u] == s.x[v]
                # Conflict found
                append!(under_conflict, u)
                break
            end
        end
    end

    for i in 1:par
        if length(under_conflict) == 0
            return
        end

        index = sample(1:length(under_conflict))[1]
        u = under_conflict[index]

        # Pick random color (different from current)
        col_candits = sample(1:s.inst.colors, 2, replace = false)
        rand_col = s.x[u] == col_candits[1] ? col_candits[2] : col_candits[1]
        s.x[u] = rand_col

        invalidate!(s)
        result.changed = true

        # Prevent this vertex from getting changed again
        deleteat!(under_conflict, index)
    end
end



function initialize!(s::GraphColoringSolution)
    s.x = sample(1:s.inst.colors, s.inst.n, replace = true)
    invalidate!(s)
end


end  # module
