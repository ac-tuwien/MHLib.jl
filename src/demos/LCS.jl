"""
    LCS

Longest Common Subsequence (LCS) problem.

Find a sequence of maximum length that is a subsequence of all given input sequences.
This is a demo problem in particular for the Monte Carlo Tree Search (MCTS).
"""
module LCS

using Random
using ArgParse
using MHLib

import Base: copy, copy!, show
import MHLib: calc_objective
import MHLib.MCTSs: Environment, Observation, State, MCTS,
    mcts!, get_state, set_state!, action_space_size, step!, reset!

export Alphabet, LCSInstance, LCSSolution, LCSEnvironment, mcts_demo

@add_arg_table! settings_cfg begin
    "--lcs_always_new_seqs"
        help = "MCTS number of simulations"
        arg_type = Bool
        default = false
end


"""
    Alphabet

Type used for letters in the LCS problem.
"""
const Alphabet = Int16

const alphabets = Dict(4 => "ACGT", 20 => "ACDEFGHIKLMNPQRSTVWY")

function get_alphabet(sigma)
    if sigma in alphabets.keys
        a = alphabets[sigma]
        Dict{Char, Alphabet}(a[i] => Alphabet(i) for i in 1:sigma)
    else
        Dict{Char, Alphabet}()  # empty for unsupported alphabets
    end
end

"""
A Longest Common Subsequence (LCS) problem instance.

The goal is to find a maximum length sequence that is a subsequence of all given input
strings.

Attributes
- `m`: number of input strings
- `n`: maximum length of input strings
- `sigma`: alphabet size, the alphabet is 1,...,sigma
- `alphabet`: dictionary translating ASCII letters to numerical values in sequences
    for supported alphabets
- `s`: vector of m input sequences of length at most n
- `succ[i, j, c]`: index of next occurrence of c in s[i] from position j onward
- `count[i, j, c]`: number of further appearances of c in s[i] from position j onward
"""
struct LCSInstance
    m::Int
    n::Int
    sigma::Alphabet
    alphabet::Dict{Char, Alphabet}
    s::Vector{Vector{Alphabet}}
    succ::Array{Int, 3}
    count::Array{Int, 3}
end

"""
    LCSInstance(m, n, sigma)

Create a random LCSInstance with m strings of length n from alphabet 1,...,sigma.
"""
function LCSInstance(m::Int, n::Int, sigma)
    @assert n > 0 && m > 0 && sigma > 0
    inst = LCSInstance(m, n, Alphabet(sigma), get_alphabet(sigma),
        [rand(Alphabet(1):Alphabet(sigma), n) for i in 1:m],
        zeros(Int, (m, n+1, sigma)), zeros(Int, (m, n+1, sigma)))
    determine_aux_data_structures(inst)
    return inst
end

"""
    LCSInstance(file)

Read LCS probem instance from file with given name.
"""
function LCSInstance(file::String)
    local s, m, sigma, alphabet
    open(file) do f
        m, sigma = [parse(Int, x) for x in split(readline(f))]
        alphabet = get_alphabet(sigma)
        s = Vector{Vector{Alphabet}}(undef, m)
        for i in 1:m
            n_str, str = split(readline(f))
            @assert length(str) == parse(Int,n_str)
            s[i] = [alphabet[c] for c in str]
        end
    end
    n = maximum(length(si) for si in s)
    inst = LCSInstance(m, n, Alphabet(sigma), alphabet, s,
        zeros(Int, (m, n+1, sigma)), zeros(Int, (m, n+1, sigma)))
    determine_aux_data_structures(inst)
    return inst
end

"""
    create_random_seqs!(inst)

Randomly re-initialize the sequences in the given LCS problem instance.
"""
function create_random_seqs!(inst::LCSInstance)
    for i in 1:m
        rand!(inst.s[i], one(Alphabet)::inst.sigma)
    end
    determine_aux_data_structures(inst)
end

Base.show(io::IO, inst::LCSInstance) = show(io, MIME"text/plain"(), inst.s)

"""
    determine_aux_dta_structure(inst)

Determine auxiliary data structures succ and count.
"""
function determine_aux_data_structures(inst::LCSInstance)
    for i in 1:inst.m
        for c in 1:inst.sigma
            pos = 0
            count = 0
            for j in inst.n:-1:1
                if inst.s[i][j] == c
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
    update_p(inst, p, c)

Update position vector p to refer to positions after the next occurrence of letter c in
each string.

Letter c must occur in each string s[i] from positions p[i] onward.
"""
function update_p(inst::LCSInstance, p::Vector, c)
    for i in 1:inst.m
        j = inst.succ[i, p[i], c]
        @assert j > 0
        p[i] = j + 1
    end
end

"""
    get_sigma_valid(inst, p)

Return sigma_valid, i.e., binary vector indicating valid actions.
"""
function get_sigma_valid(inst::LCSInstance, p::Vector)
    sigma_valid = ones(Bool, inst.sigma)
    for c in 1:inst.sigma
        for i in 1:inst.m
            if p[i] == inst.n+1  # end of sequence reached
                return zeros(Bool, inst.sigma)
            end
            if inst.count[i, p[i], c] == 0
                sigma_valid[c] = false
                break
            end
        end
    end
    return sigma_valid
end

"""
    get_observation_values(inst, p)

Return observation values for the given positin vector.

This is a vector consisting of:
- for each sequence the length of the remaining sequence from p onward
- for each letter its minimum number of occurrences over all remaining sequences
"""
function get_observation_values(inst, p)
    values = fill(Float32(inst.n), inst.m + inst.sigma)
    values[1:inst.m] = (inst.n + 1) .- p
    counts = view(values, inst.m+1:inst.m+inst.sigma)
    for i in 1:inst.m
        for c in 1:inst.sigma
            count = inst.count[i, p[i], c]
            if count < counts[c]
                counts[c] = count
            end
        end
    end
    return counts
end

#------------------------------------------------------------------------------

"""
Solution to an LCS problem instance.

Attributes
- `inst`: LCS problem instance
- `obj_val`: Length of solution string, must always be correct
- `obj_val_valid`: Should always be true
- `s::Vector{Alphabet}`: Vector containing solution sequence of length `obj_val`,
    the vector may be longer than `obj_val`
"""
mutable struct LCSSolution <: Solution
    inst::LCSInstance
    obj_val::Int
    obj_val_valid::Bool
    s::Vector{Alphabet}
end

"""
    LCSSolution(inst)

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

"""
    append!(sol, c)

Append letter c to solution.
"""
append!(sol::LCSSolution, c) = sol.s[sol.obj_val += 1] = c


"""
    LCSState

State in the LCSEnvironment.

Attributes
- `p`: position vector: the sequences are still relevant from this positions onward
- `s`: current (partial) solution
"""
struct LCSState <: State
    p::Vector{Int}
    s::LCSSolution
end

copy!(state::LCSState, state1::LCSState) =
    begin state.p[:] = state1.p; copy!(state.s, state1.s) end


"""
    LCSEnvironment

Environment for solving the LCS problem.

Attributes
- `inst`: `LCSInstance` to solve
- `state`: current state
"""
mutable struct LCSEnvironment <: Environment
    inst::LCSInstance
    state::LCSState
    LCSEnvironment(inst::LCSInstance) =
        new(inst, LCSState(ones(Int, inst.m), LCSSolution(inst)))
end

action_space_size(env::LCSEnvironment) = env.inst.sigma

get_state(env::LCSEnvironment) = env.state

set_state!(env::LCSEnvironment, state::LCSState) = copy!(env.state, state)

"""
    reset!(env)

Reset the environment.

If configuration parameter lcs_always_new_seqs is set, a new set of random sequences
is created.
The intention here is to learn a more general strategy that works not just on a single
instance.
"""
function reset!(env::LCSEnvironment)
    if settings[:lcs_always_new_seqs]
        create_random_seqs!(env.inst)
    end
    env.state = LCSState(ones(env.inst.m), LCSSolution(env.inst))
    sigma_valid = get_sigma_valid(env.inst, env.state.p)
    Observation(get_observation_values(env.inst, env.state.p), sigma_valid)
end

"""
    step!(env, action)

Appends letter given by action to the solution string.

The letter/action must always be valid, which is ensured by the valid_actions
component in the observations.
"""
function step!(env::LCSEnvironment, action::Int)
    done = false
    inst = env.inst
    state = env.state
    append!(env.state.s, action)
    update_p(inst, state.p, action)
    sigma_valid = get_sigma_valid(inst, state.p)
    not_done = any(sigma_valid)
    println("step: ", action, " to ", state.s, " ", not_done)
    if not_done
        reward = 0
        obs = Observation(get_observation_values(inst, state.p), sigma_valid)
    else
        reward = state.s.obj_val
        obs = Observation(zeros(Float32, inst.m+inst.sigma), sigma_valid)
    end
    return obs, reward, !not_done
end


"""
    mcts_demo()

Test function that runs MCTS on a small LCS instance.
"""
function mcts_demo()
    parse_settings!(["--seed=1"])
    inst = LCSInstance(3, 10, 4)
    inst = LCSInstance("data/rat-04_010_600.lcs")
    println(inst)
    env = LCSEnvironment(inst)
    mcts = MCTS()
    mcts!(mcts, env)
end

end  # module
