"""
    TSP

Demo problem: Symmetric traveling salesperson problem.

Given an undirected, weighted, complete graph, find a Hamiltonian cycle with minimum length.
"""

module TSP

using Random
using StatsBase

using MHLib
using MHLib.Schedulers
using MHLib.PermutationSolutions

export TSPInstance, TSPSolution

"""
    TSPInstance

Traveling Salesperson Problem (TSP) instance.

Given an undirected, weighted, complete graph, find a Hamiltonian cycle with minimum length.

Attributes
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
end

MHLib.to_maximize(::TSPSolution) = false

TSPSolution(inst::TSPInstance) =
    TSPSolution(inst, -1, false, collect(1:inst.n))

function Base.copy!(s1::TSPSolution, s2::TSPSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    copy!(s1.x, s2.x)
end

Base.copy(s::TSPSolution) =
    TSPSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x))

Base.show(io::IO, s::TSPSolution) =
    println(io, s.x)

MHLib.calc_objective(s::TSPSolution) =
    sum(map(i -> s.inst.d[s.x[i],s.x[(i%s.inst.n)+1]], 1:s.inst.n))

"""
    construct!(tsp_solution, par, result)

Construct new solution by random initialization.
"""
function MHLib.Schedulers.construct!(s::TSPSolution, par::Int, result::Result)
    initialize!(s)
end

"""
    local_improve!(tsp_solution, par, result)

Perform two-opt local search.
"""
function MHLib.Schedulers.local_improve!(s::TSPSolution, par::Int, result::Result)
    if !two_opt_neighborhood_search!(s, false)
        result.changed = false
    end
end

"""
    shaking!(tsp_solution, par, result)

Perform shaking by making a random 2-exchange move
"""
function MHLib.Schedulers.shaking!(s::TSPSolution, par::Int, result::Result)
    random_two_exchange_move!(s)
end

"""
    two_opt_move_delta_eval(permutation_solution, p1, p2)

Return efficiently the delta in the objective value when 2-opt move would be applied.
"""
function MHLib.PermutationSolutions.two_opt_move_delta_eval(s::TSPSolution, p1::Integer, 
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

end  # module