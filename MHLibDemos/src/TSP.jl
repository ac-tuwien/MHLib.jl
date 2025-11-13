#=
TSP.jl

Demo problem: Symmetric euclidean traveling salesperson problem.

Given points in the euclidean plane, find a Hamiltonian cycle with minimum length.
=#

using Random
using StatsBase
using MHLib

export TSPInstance, TSPSolution, solve_tsp


"""
    TSPInstance

Traveling Salesperson Problem (TSP) instance.

Given an undirected, weighted, complete graph, find a Hamiltonian cycle with minimum length.

# Attributes
- `n`: number of nodes
- `d`: distance matrix
- `coords`:Euclidean coordinates of nodes or `nothing` if not available
"""
struct TSPInstance
    n::Int
    d::Matrix{Int}
    coords::Union{Nothing, Vector{Vector{Int}}}
end

"""
    TSPInstance(coords)

Create a TSP instance from given Euclidean coordinates.
"""
function TSPInstance(coords::Vector{Vector{Int}})
    @assert length(coords) >= 2 && length(coords[1]) == 2
    n = length(coords)
    d = Matrix{Int}(undef, n, n)
    for i in 1:n
        for j in i:n
            p = coords[i]; q = coords[j]
             if i == j
                 d[i,j] = 0
             else
                 d[i,j] = d[j,i] = round(Int, sqrt((p[1] - q[1])^2 + (p[2] - q[2])^2))
             end
        end
    end
    TSPInstance(n, d, coords)
end

"""
    TSPInstance(file_name)

Read 2D Euclidean TSP instance from file in TSPLIB format.
"""
function TSPInstance(file_name::AbstractString)
    coords = Vector{Int}[]
    open(file_name) do f
        parse_coords = false
        for line in eachline(f)
           if line == "EOF"
               parse_coords = false
           end
           if parse_coords
               id, x_str, y_str = split(line, ' ')
               x = parse(Int, x_str)
               y = parse(Int, y_str)
               push!(coords, [x, y])
           end
           if line == "NODE_COORD_SECTION"
               parse_coords = true
           end
       end
       length(coords) == 0 && error("No coordinates found in file $file_name") 
   end
   TSPInstance(coords)
end

"""
    TSPInstance(n, dims::Vector=[100, 100])

Create a random Euclidean TSP instance with `n` nodes.

The nodes lie in the integer grid `[0, xdim-1] x [0, ydim-1]`.
"""
function TSPInstance(n::Int=50, dims::Vector=[100, 100])
    @assert length(dims) == 2
    coords = [trunc.(Int, rand(2) .* dims) for _ in 1:n]
    TSPInstance(coords)
end

function Base.show(io::IO, inst::TSPInstance)
    println(io, "n=$(inst.n), d=$(inst.d)")
end


"""
    TSPSolution

Solution to a TSP instance represented as permutation of integers
"""
mutable struct TSPSolution <: PermutationSolution{Int}
    inst::TSPInstance
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Int}
    destroyed::Union{Nothing, Vector{Int}}  # for LNS destroy and repair operations
end

MHLib.to_maximize(::TSPSolution) = false

TSPSolution(inst::TSPInstance) =
    TSPSolution(inst, -1, false, collect(1:inst.n), nothing)

function Base.copy!(s1::TSPSolution, s2::TSPSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.destroyed = isnothing(s2.destroyed) ? nothing : copy(s2.destroyed)
    copy!(s1.x, s2.x)
end

Base.copy(s::TSPSolution) =
    TSPSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x), 
        (isnothing(s.destroyed) ? nothing : copy(s.destroyed)))

Base.show(io::IO, s::TSPSolution) =
    println(io, s.x)

"""
    calc_objective(::TSPSolution)

Determines TSP tour length from scratch.

Can also be called for a partial solution, i.e., when `s.destroyed` is not `nothing`.
"""
function MHLib.calc_objective(s::TSPSolution)
    n = length(s.x)
    sum(map(i -> s.inst.d[s.x[i],s.x[(i%n)+1]], 1:n))
end

"""
    construct!(tsp_solution, ::Nothing, result)

`MHMethod` that constructs a new solution by random initialization.
"""
MHLib.construct!(s::TSPSolution, ::Nothing, result::Result) = initialize!(s)

"""
    local_improve!(tsp_solution, ::Any, result)

`MHMethod` that performs two-opt local search.
"""
function MHLib.local_improve!(s::TSPSolution, ::Any, result::Result)
    if !two_opt_neighborhood_search!(s, false)
        result.changed = false
    end
end

"""
    shaking!(tsp_solution, par, result)

`MHMethod` that performs shaking by making `par` random 2-exchange move.
"""
function MHLib.shaking!(s::TSPSolution, par::Int, result::Result)
    random_two_exchange_moves!(s, par)
end

"""
    destroy!(tsp_solution, par, result)

`MHMethod` that Performs a destroy operation by removing nodes from the solution.

The number of removed nodes is `3 * par`.
"""
function MHLib.destroy!(s::TSPSolution, par::Int, ::Result)
    random_remove_elements!(s, get_number_to_destroy(s, length(s.x); 
        min_abs=3par, max_abs=3par))
end

"""
    repair!(tsp_solution, ::Nothing, result)
    
`MHMethod` that performs a repair by reinserting removed nodes randomly.
"""
MHLib.repair!(s::TSPSolution, ::Nothing, ::Result) = greedy_reinsert_removed!(s)

"""
    insert_val_at_best_pos!(tsp_solution, val)

Inserts `val` greedily at the best position.
The solution's objective value is assumed to be valid and is incrementally updated.
"""
function MHLib.insert_val_at_best_pos!(s::TSPSolution, val::Int)
    x = s.x
    d = s.inst.d
    best_pos = length(x) + 1
    δ_best = δ = d[val, x[end]] + d[val, x[1]] - d[x[1], x[end]]
    for i in 2:length(s.x)
        δ = d[val, x[i-1]] + d[val, x[i]] - d[x[i-1], x[i]]       
        if δ < δ_best 
            δ_best = δ
            best_pos = i
        end
    end
    insert!(s.x, best_pos, val)
    s.obj_val = s.obj_val + δ_best
end

"""
    two_opt_move_delta_eval(permutation_solution, p1, p2)

Return efficiently the delta in the objective value when 2-opt move would be applied.
"""
function MHLib.two_opt_move_delta_eval(s::TSPSolution, p1::Integer, 
        p2::Integer)
    @assert 1 <= p1 < p2 <= length(s)
    if p1 == 1 && p2 == length(s)
        # reversing the whole solution has no effect
        return 0
    end
    prev = mod1(p1 - 1, length(s))
    nxt = mod1(p2 + 1, length(s))

    x_p1 = s.x[p1]
    x_p2 = s.x[p2]
    x_prev = s.x[prev]
    x_next = s.x[nxt]
    delta = s.inst.d[x_prev,x_p2] + s.inst.d[x_p1,x_next] - s.inst.d[x_prev,x_p1] - 
        s.inst.d[x_p2,x_next]
end


# -------------------------------------------------------------------------------

"""
    solve_tsp(alg::AbstractString, filename::AbstractString; seed=nothing, titer=1000, 
        kwargs...)

Solve a given TSP instance with the algorithm `alg`.

# Parameters
- `filename`: File name of the MAXSAT instance in CNF format
- `alg`: Algorithm to apply ("gvns" or "lns")
- `seed`: Possible random seed for reproducibility; if `nothing`, a random seed is chosen
- `titer`: Number of iterations for the solving algorithm, gets a new default value
- `kwargs`: Additional configuration parameters passed to the algorithm, e.g., `ttime`
"""
function solve_tsp(alg::AbstractString="lns",
        filename::AbstractString=joinpath(@__DIR__, "..", "data", "xqf131.tsp");
        seed=nothing, titer=1000, kwargs...)
    # Make results reproducibly by either setting a given seed or picking one randomly
    isnothing(seed) && (seed = rand(0:typemax(Int32)))
    Random.seed!(seed)

    println("TSP Demo version $(git_version())")
    println("alg=$alg, filename=$filename, seed=$seed, ", (; kwargs...))

    inst = TSPInstance(filename)
    sol = TSPSolution(inst)
    initialize!(sol)
    println(sol)

    if alg === "lns"
        heuristic = LNS(sol, MHMethod[MHMethod("con", construct!)],
            [MHMethod("de$i", destroy!, i) for i in 1:3],
            [MHMethod("re", repair!)]; 
            consider_initial_sol=true, titer, kwargs...)
    elseif alg === "gvns"
        heuristic = GVNS(sol, [MHMethod("con", construct!)],
            [MHMethod("li1", local_improve!, 1)], [MHMethod("sh1", shaking!, 1)];
            consider_initial_sol=true, titer, kwargs...)
    else
        error("Invalid parameter alg: $alg")
    end
    run!(heuristic)
    method_statistics(heuristic.scheduler)
    main_results(heuristic.scheduler)
    check(sol)
    return sol
end

# To run from REPL, activate `MHLibDemos` environment, use `MHLibDemos`,
# and call e.g. `solve_tsp("lns", titer=200, seed=1)`.

# Run with profiler:
# @profview solve_tsp(args)
