"""
    MAXSAT

MAXSAT demo problem.

The goal is to maximize the number of clauses satisfied in a boolean function given in
conjunctive normal form.
"""
module MAXSAT

using Random
using MHLib
using MHLib.Schedulers

import Base: copy, copy!, show
import MHLib: calc_objective, flip_variable!

export MAXSATInstance, MAXSATSolution


"""
A MAXSAT problem instance.

The goal is to maximize the number of clauses satisfied in a boolean function given in
conjunctive normal form.

Attributes
- `n`: number of variables, i.e., size of incidence vector
- `m`: number of clauses
- `clauses`: vector of clauses, where each clause is represented by a
    vector of integers;
    a positive integer v refers to the v-th variable, while a negative
    integer v refers to the negated form of the v-th variable
- `variable_usage`: vector containing for each variable a vector with the
    indices of the clauses in which the variable appears;
    needed for efficient incremental evaluation
"""
struct MAXSATInstance
    n::Int
    m::Int
    clauses::Vector{Vector{Int}}
    variable_usage::Vector{Vector{Int}}
end

"""
    MAXSATInstance(file_name)

Read a MAXSATInstance from the given file.
"""
function MAXSATInstance(file_name::String)
    local n::Int
    local m::Int
    clauses = Vector{Vector{Int}}()
    local variable_usage::Vector{Vector{Int}}
    open(file_name) do f
        for line in eachline(f)
            if line[1] == 'c'
                # ignore comments
                continue
            end
            fields = split(line)
            if length(fields) == 4 && fields[1] == "p" && fields[2] == "cnf"
                # parse header line with n and m
                n = parse(Int, fields[3])
                m = parse(Int, fields[4])
                variable_usage = [Vector{Int}() for _ in 1:n]
            elseif length(fields) >= 1
                # parse clause
                clause = [parse(Int, s) for s in fields[1:length(fields)-1]]
                push!(clauses, clause)
                for v in clause
                    push!(variable_usage[abs(v)], length(clauses))
                end
            end
        end
    end
    @assert n > 0 && m > 0
    @assert length(clauses) == m
    MAXSATInstance(n, m, clauses, variable_usage)
end


"""
    MAXSATSolution

A concrete solution type to solve the MAXSAT problem.
"""
mutable struct MAXSATSolution <: BoolVectorSolution
    inst::MAXSATInstance
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Bool}
end

"""
    MAXSATSolution(::MAXSATInstance)

Create a solution object for the given `MAXSATInstance`.
"""
MAXSATSolution(inst::MAXSATInstance) =
    MAXSATSolution(inst, -1, false, Vector{Bool}(undef, inst.n))


function copy!(s1::S, s2::S) where {S <: MAXSATSolution}
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
end


copy(s::MAXSATSolution) =
    MAXSATSolution(s.inst, -1, false, Base.copy(s.x[:]))


Base.show(io::IO, s::MAXSATSolution) =
    println(io, "Solution: ", s.x)


"""
    calc_objective(::MAXSATSolution)

Count the number of satisfied clauses.
"""
function calc_objective(s::MAXSATSolution)::Int
    satisfied = 0
    for clause in s.inst.clauses
        for v in clause
            if s.x[abs(v)] == (v > 0)
                satisfied += 1
                break
            end
        end
    end
    satisfied
end


function flip_variable!(s::MAXSATSolution, pos::Int)::Int
    obj_val = obj(s)
    val = !s.x[pos]
    s.x[pos] = val
    for clause in s.inst.variable_usage[pos]
        fulfilled_by_other = false
        val_fulfills_now = false
        for v in s.inst.clauses[clause]
            if v == 0 break end
            if abs(v) == pos
                val_fulfills_now = (v>0 ? val : !val)
            elseif s.x[abs(v)] == (v>0 ? 1 : 0)
                fulfilled_by_other = true
                break  # clause fulfilled by other variable, no change
            end
        end
        if !fulfilled_by_other
            obj_val += (val_fulfills_now ? 1 : -1)
        end
    end
    return obj_val
end

#=

function destroy(self, par, _result)
    """Destroy operator for ALNS selects par*ALNS.get_number_to_destroy positions
    uniformly at random for removal.

    Selected positions are stored with the solution in list self.destroyed.
    """
    x = PyArray(self."x")
    num = min(ALNS.get_number_to_destroy(length(x)) * par, length(x))
    self.destroyed = sample(1:length(x), num, replace=false)
    self.invalidate()
end

function repair(self, _par, _result)
    """Repair operator for ALNS assigns new random values to all positions in self.destroyed."""
    @assert !(self.destroyed === nothing)
    x = PyArray(self."x")
    for p in self.destroyed
        x[p] = rand(0:1)
    end
    self.destroyed = nothing
    self.invalidate()
end


function crossover(self, other)
    """ Perform uniform crossover as crossover."""
    return self.uniform_crossover(other)
end

=#

end  # module
