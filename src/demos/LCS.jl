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
import MHLib.Environments:
    Environment,
    Observation,
    State,
    get_state,
    set_state!,
    action_space_size,
    step!,
    reset!
import MHLib.MCTSs: MCTS, mcts!

export Alphabet, LCSInstance, LCSSolution, LCSEnvironment, mcts_demo

@add_arg_table! settings_cfg begin
    "--lcs_always_new_seqs"
        help = "LCS: Always create new strings when reset is called"
        arg_type = Bool
        default = false
    "--lcs_reward_mode"
        help = "LCS reward mode: direct or smallsteps"
        arg_type = String
        default = "direct"
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
        Dict{Char,Alphabet}(a[i] => Alphabet(i) for i = 1:sigma)
    else
        Dict{Char,Alphabet}()  # empty for unsupported alphabets
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
    alphabet::Dict{Char,Alphabet}
    s::Vector{Vector{Alphabet}}
    succ::Array{Int,3}
    count::Array{Int,3}
end

"""
    LCSInstance(m, n, sigma)

Create a random LCSInstance with m strings of length n from alphabet 1,...,sigma.
"""
function LCSInstance(m::Int, n::Int, sigma)
    @assert n > 0 && m > 0 && sigma > 0
    inst = LCSInstance(
        m,
        n,
        Alphabet(sigma),
        get_alphabet(sigma),
        [rand(Alphabet(1):Alphabet(sigma), n) for i = 1:m],
        zeros(Int, (m, n + 1, sigma)),
        zeros(Int, (m, n + 1, sigma)),
    )
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
        for i = 1:m
            n_str, str = split(readline(f))
            @assert length(str) == parse(Int, n_str)
            s[i] = [alphabet[c] for c in str]
        end
    end
    n = maximum(length(si) for si in s)
    inst = LCSInstance(
        m,
        n,
        Alphabet(sigma),
        alphabet,
        s,
        zeros(Int, (m, n + 1, sigma)),
        zeros(Int, (m, n + 1, sigma)),
    )
    determine_aux_data_structures(inst)
    return inst
end

"""
    create_random_seqs!(inst)

Randomly re-initialize the sequences in the given LCS problem instance.
"""
function create_random_seqs!(inst::LCSInstance)
    for i = 1:m
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
    for i = 1:inst.m
        for c = 1:inst.sigma
            pos = 0
            count = 0
            for j = inst.n:-1:1
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
    for i = 1:inst.m
        j = inst.succ[i, p[i], c]
        @assert j > 0
        p[i] = j + 1
    end
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


function Base.string(sol::LCSSolution)
    res = "LCSSolution:"
    res = res * "\n  obj_val: " * string(sol.obj_val)
    res = res * "\n  s:" * string(sol.s)
    return (res)
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
append!(sol::LCSSolution, c) = sol.s[sol.obj_val+=1] = c


"""
    LCSState

State in the LCSEnvironment.

Attributes
- `p`: position vector: the sequences are still relevant from this positions onward
- `s`: current (partial) solution; TODO: now just for debugging, can be replaced later
    by just the length of the solution (if necessary at all)
- `action_valid_mask`: boolean vector indicating valid further actions
"""
struct LCSState <: State
    p::Vector{Int}
    s::LCSSolution
    action_valid_mask::Vector{Bool}
end

function Base.string(state::LCSState)
    res = "State:"
    res = res * "\n  " * Base.string(state.p)
    return res
end

function copy!(state::LCSState, state1::LCSState)
    state.p[:] = state1.p
    copy!(state.s, state1.s)
    state.action_valid_mask[:] = state1.action_valid_mask
end

"""
    update_action_valid_mask(state, inst)

Return action_valid_mask, i.e., binary vector indicating valid actions.
"""
function update_action_valid_mask(state::LCSState, inst::LCSInstance)
    for c = 1:inst.sigma
        if !state.action_valid_mask[c]
            continue
        end
        for i = 1:inst.m
            if state.p[i] == inst.n + 1  # end of sequence reached
                fill!(state.action_valid_mask, false)
            end
            if inst.count[i, state.p[i], c] == 0
                state.action_valid_mask[c] = false
                break
            end
        end
    end
end


"""
    LCSEnvironment

Environment for solving the LCS problem.

Attributes
- `inst`: `LCSInstance` to solve
- `state`: current state
- `seq_order`: order of sequences in current observation
- `action_order`: order of actions in current observation
"""
mutable struct LCSEnvironment <: Environment
    inst::LCSInstance
    state::LCSState
    seq_order::Vector{Int}
    action_order::Vector{Int}

    function LCSEnvironment(inst::LCSInstance)
        p = ones(Int, inst.m)
        state = LCSState(p, LCSSolution(inst), ones(Bool, inst.sigma))
        update_action_valid_mask(state, inst)
        new(inst, state, Vector{Int}(undef, 0), Vector{Int}(undef, 0))
    end
end

action_space_size(env::LCSEnvironment) = env.inst.sigma

observation_space_size(env::LCSEnvironment) =
    env.inst.m + env.inst.sigma + env.inst.sigma * env.inst.m

get_state(env::LCSEnvironment) = env.state

function set_state!(env::LCSEnvironment, state::LCSState)::Observation
    copy!(env.state, state)
    return get_observation(env)
end

"""
    reset!(env)

Reset the environment.

If configuration parameter lcs_always_new_seqs is set, a new set of random sequences
is created.
The intention here is to learn a more general strategy that works not just on a single
instance.
"""
function reset!(env::LCSEnvironment)::Observation
    if settings[:lcs_always_new_seqs]
        create_random_seqs!(env.inst)
    end
    p = ones(Int, env.inst.m)
    env.state = LCSState(p, LCSSolution(env.inst), ones(Bool, env.inst.sigma))
    update_action_valid_mask(env.state, env.inst)
    get_observation(env)
end

"""
    step!(env, action)

Perform given action, i.e., append letter corresponding to action to solution string.

The letter/action must always be valid, which is ensured by the valid_actions
component in the observations.
"""
function step!(env::LCSEnvironment, action::Int)
    done = false
    inst = env.inst
    state = env.state
    c = action  # env.action_order[action]
    append!(env.state.s, c)
    update_p(inst, state.p, c)
    update_action_valid_mask(state, inst)
    not_done = any(state.action_valid_mask)
    # println("step: ", c, " appended to ", state.s, " ", not_done)
    reward_mode = settings[:lcs_reward_mode]
    if not_done
        if reward_mode === "direct"
            reward = 0.0
        elseif reward_mode === "simplesteps"
            reward = 0.05
        else
            error("Invalid reward_mode $reward_mode")
        end
        obs = get_observation(env)
    else
        if reward_mode === "direct"
            reward = state.s.obj_val
        elseif reward_mode === "simplesteps"
            reward = -1.0
        else
            error("Invalid reward_mode $reward_mode")
        end
        obs = Observation(
            zeros(Float32, observation_space_size(env)),
            ones(Bool, inst.sigma),
        )
    end
    return obs, reward, !not_done
end

"""
    get_observation(env)

Return observation for the current state in the environment.

This is a vector consisting of:
- for each sequence the length of the remaining sequence from p onward sorted
    in non-decreasing order
- for each letter its minimum number of occurrences over all remaining sequences
    sorted in non-decreasing order
- for each letter the lengths of the remaining sequences after appending the letter
    to the partial solution, sorted according to the sequence and letter orderings from
    above
"""
function get_observation(env::LCSEnvironment)::Observation
    m = env.inst.m
    sigma = env.inst.sigma
    p = env.state.p
    s = env.inst.s
    values = Vector{Float32}(undef, observation_space_size(env))
    lengths = [length(s[i]) - p[i] + 1 for i = 1:m]
    counts = fill(env.inst.n, sigma)
    for i = 1:m
        for c = 1:sigma
            count = env.inst.count[i, p[i], c]
            if count < counts[c]
                counts[c] = count
            end
        end
    end
    env.seq_order = sortperm(lengths)
    values[1:m] = lengths[env.seq_order]
    # env.action_order = sortperm(counts)
    values[m+1:m+sigma] = counts  # [env.action_order]
    idx = m + sigma + 1
    for i = 1:m
        for c = 1:sigma
            values[idx] = length(s[i]) - env.inst.succ[i, p[i], c]
        end
    end
    action_mask = env.state.action_valid_mask  # [env.action_order]
    return Observation(values, action_mask)
end


"""
    mcts_demo()

Test function that runs MCTS on a small LCS instance.
"""
function mcts_demo()
    parse_settings!(["--seed=1", "--mh_mcts_num_sims=100", "--mh_mcts_c_uct=50"])
    inst = LCSInstance(3, 8, 4)
    # inst = LCSInstance("data/rat-04_010_600.lcs")
    println(inst)
    env = LCSEnvironment(inst)
    mcts = MCTS()
    println("Anzahl der Iterationen: ", mcts.num_sims)
    actions = mcts!(mcts, env)
    println(actions)
end



function test()
    parse_settings!(["--seed=1", "--mh_mcts_num_sims=100", "--mh_mcts_c_uct=1"])
    inst = LCSInstance(3, 8, 4)
    # inst = LCSInstance("data/rat-04_010_600.lcs")
    println(inst)
    env = LCSEnvironment(inst)
    mcts = MCTS()

    append!(actions, compute_action!(mcts, root))
end

end  # module
