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
import MHLib.Schedulers: construct!, local_improve!, shaking!
import MHLib: calc_objective, check

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
    println(io, "TSP Solution: ", s.x)

calc_objective(s::TSPSolution) =
    sum(map(i -> s.inst.d[s.x[i],s.x[(i%s.inst.n)+1]], 1:s.n))

function check(s::TSPSolution)
    invoke(check, Tuple{PermutationSolution}, s)
end

function clear(s::TSPSolution)
    invoke(clear, Tuple{PermutationSolution}, s)
end

function initialize!(s::TSPSolution)
    invoke(initialize!, Tuple{PermutationSolution}, s)
end

"""
    construct!(tsp_solution, par, result)

Construct new solution by random initialization.
"""
function construct!(s::TSPSolution, par::Int, result::Result)
    initialize!(s)
end

end  # module