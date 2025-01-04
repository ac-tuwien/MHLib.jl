"""
    GraphColoring

Demo application solving the graph coloring problem.

Given a graph and an number of colors, color each node with one color so that
the number of adjacent nodes having the same color is minimized.
"""

using Random
using StatsBase
using Graphs
using MHLib

export GraphColoringInstance, GraphColoringSolution, solve_graph_coloring,
    graph_coloring_settings_cfg

# We define a problem-specific parameter:
const graph_coloring_settings_cfg = ArgParseSettings()
@add_arg_table! graph_coloring_settings_cfg begin
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
    colors = settings[:gcp_colors]::Int
    GraphColoringInstance(graph, n, m, colors)
end

function Base.show(io::IO, inst::GraphColoringInstance)
    println(io, "n=$(inst.n), m=$(inst.m)")
end



"""
    GraphColoringSolution

Solution to a Graph Coloring instance.

It is a `VectorSolution{Int}`.

Attributes
- `x`: vector with the assigned colors
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
    copy!(s1.x, s2.x)
end

Base.copy(s::GraphColoringSolution) =
    GraphColoringSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x))

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
    check(s::GraphColoringSolution; ...)

Check if s is a valid solution.
Raises an error if a problem is detected.
"""
function MHLib.check(s::GraphColoringSolution; kwargs...)
    if length(s.x) != s.inst.n
        error("Invalid length of solution")
    end
    if sum(s.x .> s.inst.colors) >= 1
        error("Too many colors used")
    end
    invoke(MHLib.check, Tuple{supertype(typeof(s))}, s; kwargs...)
end


"""
    construct!(s::GraphColoringSolution, ::Nothing, result)

`MHMethod` that constructs a new solution. Here we just call initialize!.
"""
MHLib.construct!(s::GraphColoringSolution, ::Nothing, ::Result) = initialize!(s)


"""
    local_improve!(s::GraphColoringSolution, par, result)

`MHMethod` that performs one iteration of a local search.

Following a first improvement strategy.
The neighborhood used is defined by all solutions that can be created by changing the color
of a vertex involved in a conflict.
"""
function MHLib.local_improve!(s::GraphColoringSolution, ::Nothing, result::Result)
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

`MHMethod` that randomly assigns different colors to 'par' vertices involved in conflicts.
"""
function MHLib.shaking!(s::GraphColoringSolution, par::Int, result::Result)
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

    for _ in 1:par
        iszero(under_conflict) && return

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


function MHLib.initialize!(s::GraphColoringSolution)
    s.x = sample(1:s.inst.colors, s.inst.n, replace = true)
    invalidate!(s)
end


# -------------------------------------------------------------------------------

function solve_graph_coloring(args=ARGS)
    println("Graph Coloring Demo version $(git_version())\nARGS: ", args)
    args isa AbstractString && (args = split(args))

    # We set some new default values for parameters and parse all relevant arguments
    settings_new_default_value!(MHLib.scheduler_settings_cfg, "mh_titer", 1000)
    settings_new_default_value!(MHLib.settings_cfg, "ifile", "data/fpsol2.i.1.col")
    parse_settings!([MHLib.schedulers_settings_cfg, graph_coloring_settings_cfg], args)
    println(get_settings_as_string())
        
    inst = GraphColoringInstance(settings[:ifile])
    sol = GraphColoringSolution(inst)
    initialize!(sol)
    println(sol)

    alg = GVNS(sol, [MHMethod("con", construct!)],
        [MHMethod("li1", local_improve!, 1)],[MHMethod("sh1", shaking!, 1)], 
        consider_initial_sol = true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end

# To run from REPL, use `MHLibDemos` and call `solve_graph_coloring(<args>)` where `<args>`
# is a single string or list of strings being passed as arguments for setting global 
# parameters, e.g. `solve_graph_coloring("--seed=1 --mh_titer=120")`.
# `@<filename>` may be used to read arguments from a configuration file <filename>

# Run with profiler:
# @profview solve_graph_coloring(args)