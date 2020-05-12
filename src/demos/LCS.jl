"""
Longest Common Subsequence (LCS) problem.

The goal is to find a sequence of maximum length that is a subsequence of all given input
sequences.

This is a demo problem for the Monte Carlo Tree Search (MCTS).
"""
module LCS

using Random
using MHLib

import Base: copy, copy!, show
import MHLib: calc_objective
import MHLib.MCTSs: Environment, Observation, State, MCTS,
    run!, get_state, set_state!, get_obs, action_space_size, step!

export Alphabet, LCSInstance, LCSSolution, LCSEnvironment, mcts, get_state

const Alphabet = Int16

"""
A Longest Common Subsequence (LCS) problem instance.

The goal is to find a maximum length sequence that is a subsequence of all given input
strings.

Attributes
- `n`: length of each input string
- `m`: number of input strings
- `sigma`: alphabet size, the alphabet is 1,...,sigma
- `s`: array of m input strings of length n
"""
struct LCSInstance
    n::Int
    m::Int
    sigma::Alphabet
    S::Array{Alphabet, 2}
    succ::Array{Int, 3}
    count::Array{Int, 3}
end

"""
    LCSInstance()

Create a random LCSInstance.
"""
function LCSInstance(n::Int, m::Int, sigma::Alphabet)
    @assert n > 0 && m > 0 && sigma >0
    inst = LCSInstance(n, m, sigma, rand(Alphabet(1):sigma, (m, n)),
        zeros(Int, (m, n+1, sigma)), zeros(Int, (m, n+1, sigma)))
    determine_aux_data_structures(inst)
    return inst
end

"""
    create_random_seqs(lcs, sigma)

Randomly re-initialize the sequences in the given LCS problem instance.
"""
function create_random_seqs(inst::LCSInstance, n::Int, m::Int, sigma::Alphabet)
    rand!(inst.S, one(Alphabet)::sigma)
    determine_aux_data_structures(inst)
end

Base.show(io::IO, inst::LCSInstance) =
    show(io, MIME"text/plain"(), inst.S)

"""
    determine_aux_dta_structure(inst)

Determine auxiliary data structures: succ and count.
"""
function determine_aux_data_structures(inst::LCSInstance)
    for i in 1:inst.m
        for c in 1:inst.sigma
            pos = 0
            count = 0
            for j in inst.n:-1:1
                if inst.S[i, j] == c
                    pos = j
                    count += 1
                end
                inst.succ[i, j, c] = pos
                inst.count[i, j, c] = count
            end
        end
    end
end


"""
Solution to an LCS problem instance.
"""
mutable struct LCSSolution
    inst::LCSInstance
    obj_val::Int
    obj_val_valid::Bool
    s::Vector{Alphabet}
end

"""
    LCSSolution()

Creates an empty solution for the given LCS problem instance.
"""
LCSSolution(inst::LCSInstance) =
    LCSSolution(inst, 0, true, zeros(Alphabet, inst.n))

function copy!(sol1::LCSSolution, sol2::LCSSolution)
    sol1.inst = sol2.inst
    sol1.obj_val = sol2.obj_val
    sol1.obj_val_valid = sol2.obj_val_valid
    sol1.s[:] = sol2.s
end

copy(sol::LCSSolution) = deepcopy(sol)

function Base.show(io::IO, sol::LCSSolution)
    l = sol.obj_val
    print(io, "Solution: ", l, " ")
    show(io, sol.s[1:l])
end

"""
    calc_objective(::LCSSolution)

The length of the solution is stored in sol.obj_val and must always be valid.
"""
calc_objective(sol::LCSSolution)::Int = sol.obj_val

append!(sol::LCSSolution, c) = sol.s[sol.obj_val += 1] = c


struct LCSState <: State
    p::Vector{Int}
    s::LCSSolution
end


mutable struct LCSEnvironment <: Environment
    inst::LCSInstance
    state::LCSState
    LCSEnvironment(inst::LCSInstance) =
        new(inst, LCSState(ones(Int, inst.m), LCSSolution(inst)))
end

action_space_size(env::LCSEnvironment) = env.inst.m

get_state(env::LCSEnvironment) = env.state

set_state!(env::LCSEnvironment, state::LCSState) = (env.state = state)

get_obs(env::LCSEnvironment) =
    Observation((env.inst.n+1) .- env.state.p, ones(Bool, env.inst.m))

function step!(env::LCSEnvironment, action::Int)
    done = false
    for i in 1:env.inst.m
        env.state.p[i] = env.inst.succ[i, env.state.p[i], action]
        if env.state.p[i] == 0
            done = true
        end
    end
    if done
        reward = env.state.s.obj_val
    else
        reward = 0
        append!(env.state.s, action)
    end
    return get_obs(env), reward, done
end



function mcts()
    inst = LCSInstance(10, 3, Alphabet(4))
    println(inst)
    env = LCSEnvironment(inst)
    mcts = MCTS()
    run!(mcts, env)
end

end  # module
