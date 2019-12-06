module MAXSAT

import Base: copy, copy!
import MHLib: BoolVectorSolution, calc_objective

export MAXSATInstance, MAXSATSolution

"""MAXSATInstance

A MAXSAT problem instance.

The goal is to maximize the number of clauses satisfied in a boolean function given in conjunctive normal form.

Attributes
    - n: number of variables, i.e., size of incidence vector
    - m: number of clauses
    - clauses: vector of clauses, where each clause is represented by a
        vector of integers;
        a positive integer v refers to the v-th variable, while a negative
        integer v refers to the negated form of the v-th variable
    - variable_usage: vector containing for each variable a vector with the
        indices of the clauses in which the variable appears;
        needed for efficient incremental evaluation
"""
struct MAXSATInstance
    n::Int
    m::Int
    clauses::Vector{Vector{Int}}
    variable_usage::Vector{Vector{Int}}
end

""" MAXSATInstance

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

inst = MAXSATInstance("../data/maxsat-simple.cnf")


"""MAXSATSolution

A concrete solution type to solve the MAXSAT problem.
"""
mutable struct MAXSATSolution{N} <: BoolVectorSolution{N}
    inst::MAXSATInstance
    obj_val::Int
    obj_val_valid::Bool
    x::MVector{N,Bool}
end

"""MAXSATSolution(::MAXSATInstance)

Create a solution object for the given `MAXSATInstance`.
"""
MAXSATSolution(inst::MAXSATInstance) =
    MAXSATSolution{inst.n}(inst, -1, false, MVector{inst.n, Bool}(undef))

function copy!(s1::S, s2::S) where {S <: MAXSATSolution}
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    s1.x[:] = s2.x
end

copy(s::MAXSATSolution) =
    MAXSATSolution{s.inst.n}(s.inst, -1, false, Base.copy(s.x[:]))

"""calc_objective(::MAXSATSolution)

Count the number of satisfied clauses.
"""
function calc_objective(s::MAXSATSolution)
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

#=

function local_improve(self, par, _result)
    """Perform one k_flip_neighborhood_search."""
    x = self.x
    obj_val = self.obj()
    new_obj_val = k_flip_neighborhood_search!(x, obj_val, self.inst.julia_inst, par, false)
    if new_obj_val > obj_val
        PyArray(self."x")[:] = x
        self.obj_val = new_obj_val
        return true
    end
    return false
end

function shaking(self, par, _result)
    """Scheduler method that performs shaking by flipping par random positions."""
    x = PyArray(self."x")
    for i in 1:par
        p = rand(1:length(x))
        x[p] = !x[p]
    end
    self.invalidate()
end

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


function k_flip_neighborhood_search!(x::Vector{Bool}, obj_val::Int, julia_inst::JMAXSATInstance,
                                 k::Int, best_improvement::Bool)::Int
"""Perform one major iteration of a k-flip local search, i.e., search one neighborhood.

If best_improvement is set, the neighborhood is completely searched and a best neighbor is
kept; otherwise the search terminates in a first-improvement manner, i.e., keeping a first
encountered better solution.

:returns: Objective value.
"""
len_x = length(x)
@assert 0 < k <= len_x
better_found = false
best_sol = copy(x)
best_obj = obj_val
perm = randperm(len_x)  # random permutation for randomizing enumeration order
p = fill(-1, k)  # flipped positions
# initialize
i = 1  # current index in p to consider
while i >= 1
    # evaluate solution
    if i == k + 1
        if obj_val > best_obj
            if !best_improvement
                return true
            end
            best_sol[:] = x
            best_obj = obj_val
            better_found = true
        end
        i -= 1  # backtrack
    else
        if p[i] == -1
            # this index has not yet been placed
            p[i] = (i>1 ? p[i-1] : 0) + 1
            obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
            i += 1  # continue with next position (if any)
        elseif p[i] < len_x - (k - i)
            # further positions to explore with this index
            obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
            p[i] += 1
            obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
            i += 1
        else
            # we are at the last position with the i-th index, backtrack
            obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
            p[i] = -1  # unset position
            i -= 1
        end
    end
end
if better_found
    x[:] = best_sol
    obj_val = best_obj
end
return obj_val
end


function flip_variable!(x::Vector{Bool}, pos::Int, julia_inst::JMAXSATInstance, obj_val::Int)::Int
val = !x[pos]
x[pos] = val
for clause in view(julia_inst.variable_usage,:,pos)
    if clause == 0 break end
    fulfilled_by_other = false
    val_fulfills_now = false
    for v in view(julia_inst.clauses,:,clause)
        if v == 0 break end
        if abs(v) == pos
            val_fulfills_now = (v>0 ? val : !val)
        elseif x[abs(v)] == (v>0 ? 1 : 0)
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

=#

end  # module
