"""
    TSP

Demo application solving the traveling salesperson problem.

Given an undirected, weighted, complete graph, find a Hamiltonian cycle with minimum length.
"""

module TSP

using Random
using StatsBase

using MHLib
using MHLib.Schedulers
import MHLib.Schedulers: construct!, local_improve!, shaking!, to_maximize
import MHLib: calc_objective, check, two_opt_move_delta_eval

import Base: copy, copy!, show

export TSPInstance, TSPSolution

"""
    TSPInstance

Traveling Salesperson Problem (TSP) instance.

Given an undirected, weighted, complete graph, find a Hamiltonian cycle with minimum length.

Attributes
- `n`: number of nodes
- `d`: distance matrix
"""
struct TSPInstance
    n::Int
    d::Array{Int, 2}
end

"""
    TSPInstance(file_name)

Read TSP instance from file in tslib format
"""
function TSPInstance(file_name::AbstractString)
    coords = Vector{Float64}[]
    
    open(file_name) do f
        parse_coords = false
    
        for line in eachline(f)
            if line == "EOF"
               parse_coords = false
           end
           if parse_coords
               id, x_str, y_str = split(line, ' ')
               x = parse(Float64, x_str)
               y = parse(Float64, y_str)
               push!(coords, [x, y])
           end
           if line == "NODE_COORD_SECTION"
               parse_coords = true
           end
       end
   end

   n = length(coords)
   d = zeros(Float64, n, n)
   for (i, x_1) in enumerate(coords)
       for (j, x_2) in enumerate(coords)
            d[i, j] = trunc(sqrt((x_1[1]::Float64 - x_2[1]::Float64)^2 + (x_1[2]::Float64 - x_2[2]::Float64)^2) + 0.5)
       end
   end

   TSPInstance(n, d)
end

function show(io::IO, inst::TSPInstance)
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
    n::Int
end

to_maximize(::TSPSolution) = false

TSPSolution(inst::TSPInstance) =
    TSPSolution(inst, -1, false, collect(1:inst.n), inst.n)

function copy!(s1::S, s2::S) where {S <: TSPSolution}
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
    s1.n = s2.n
end

copy(s::TSPSolution) =
    TSPSolution(s.inst, s.obj_val, s.obj_val_valid, Base.copy(s.x[:]), s.n)

Base.show(io::IO, s::TSPSolution) =
    println(io, s.x)

calc_objective(s::TSPSolution) =
    sum(map(i -> s.inst.d[s.x[i],s.x[(i%s.inst.n)+1]], 1:s.n))

"""
    construct!(tsp_solution, par, result)

Construct new solution by random initialization.
"""
function construct!(s::TSPSolution, par::Int, result::Result)
    initialize!(s)
end

"""
    local_improve!(tsp_solution, par, result)

Perform two-opt local search.
"""
function local_improve!(s::TSPSolution, par::Int, result::Result)
    if !two_opt_neighborhood_search!(s, false)
        result.changed = false
    end
end

"""
    shaking!(tsp_solution, par, result)

Perform shaking by making a random 2-exchange move
"""
function shaking!(s::TSPSolution, par::Int, result::Result)
    random_two_exchange_move!(s)
end

"""
    two_opt_move_delta_eval(permutation_solution, p1, p2)

Return efficiently the delta in the objective value when 2-opt move would be applied.
"""
function two_opt_move_delta_eval(s::TSPSolution, p1::Integer, p2::Integer)
    @assert 1 <= p1 < p2 <= s.n
    if p1 == 1 && p2 == s.n
        # reversing the whole solution has no effect
        return 0
    end
    prev = mod1(p1 - 1, s.n)
    nxt = mod1(p2 + 1, s.n)

    x_p1 = s.x[p1]
    x_p2 = s.x[p2]
    x_prev = s.x[prev]
    x_next = s.x[nxt]
    delta = s.inst.d[x_prev,x_p2] + s.inst.d[x_p1,x_next] - s.inst.d[x_prev,x_p1] - s.inst.d[x_p2,x_next]
end

end  # module