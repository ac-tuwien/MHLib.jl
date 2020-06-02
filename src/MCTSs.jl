"""
    MCTSs

Monte Carlo Tree Search (MCTS) based on Prior Upper Confidence Bound (PUCT) strategy.

This module realizes a Monte Carlo Tree Search.
This implementation is inspired by
https://github.com/ray-project/ray/blob/master/rllib/contrib/alpha_zero/core/mcts.py
"""
module MCTSs

using ArgParse
using MHLib
using StatsBase

import MHLib.Environments: Environment, Observation, State, get_state, set_state!, reset!,
    action_space_size, step!

export MCTS, perform_mcts!, get_child

@add_arg_table! settings_cfg begin
    "--mh_mcts_num_sims"
        help = "MCTS number of simulations"
        arg_type = Int
        default = 100
    "--mh_mcts_c_uct"
        help = "MCTS c_uct coefficient"
        arg_type = Float64
        default = 1.0
    "--mh_mcts_tree_policy"
        help = "MCTS tree Policy to apply: UCB or PUCT"
        arg_type = String
        default = "UCB"
end


"""
    argmax_rand(vector)

Return index of maximum value in vector and in case of ties make a random selection.
"""
function argmax_rand(a::Vector)
    indices = findall(a.==maximum(a))
    if length(indices) > 1
        rand(indices)
    else
        indices[1]
    end
end


"""
    Node

A node of the MCTS tree.

Attributes
- `env`: Environment on which the MCTS shall be performed
- `action`: Action performed to go to this state
- `is_expanded`: Indicates if this node is already expanded or not
- `parent`: Reference to parent node or Nothing
- `children`: Vector of references to child nodes or Nothing
- `action_space_size`: Size of action space
- `child_W`: Total values of child nodes
- `child_P`: Priors of child nodes
- `child_N`: Number of visits of child nodes
- `valid_actions`: Array indicating if an action ist valid (true) or not (false)
- `reward`: Reward received at this node
- `done`: Indicates end of episode
- `state`: State corresponding to this node
- `obs`: Observation at the state of this node
"""
mutable struct Node{TState <: State}
    action::Int
    is_expanded::Bool
    parent::Union{Node{TState}, Nothing}
    children::Vector{Union{Node{TState}, Nothing}}
    action_space_size::Int
    child_W::Vector{Float32}
    child_P::Vector{Float32}
    child_N::Vector{Int32}
    valid_actions::Vector{Bool}
    reward::Float32
    done::Bool
    state::TState
    obs::Observation
end


"""
    MCTS

Monte Carlo Tree Search.
"""
mutable struct MCTS{TEnv <: Environment}
    num_sims::Int
    c_uct::Float64
    tree_policy::String

    env::TEnv
    root::Node
end


function Node{TState}(env::Environment, action::Int, state::TState, obs::Observation,
    done::Bool, reward, parent::Union{Node, Nothing}) where {TState <: State}
    sigma = action_space_size(env)
    return Node{TState}(action, false, parent, Vector{Union{Node, Nothing}}(nothing, sigma),
            sigma, zeros(Float32, sigma), zeros(Float32, sigma), zeros(Float32, sigma),
            copy(obs.valid_actions), reward, done, deepcopy(state), obs)
end

function Base.string(node::Node)
    res = "Node:"
    res = res * "\n  action = $(node.action)"
    res = res * "\n  child_N: " * Base.string(node.child_N)
    res = res * "\n  child_W: " * Base.string(node.child_W)
    res = res * "\n  child_Q: " * Base.string(child_Q(node))
    res = res * "\n  child_P: " * Base.string(node.child_P)
    res = res * "\n  valid_actions:" * Base.string(node.valid_actions)
    res = res * "\n  reward: " * Base.string(node.reward)
    res = res * "\n  done: " * Base.string(node.done)
    res = res * "\n" * string(node.state)
    return res
end

N(node::Node) = node.parent.child_N[node.action]
N!(node::Node, n) = (node.parent.child_N[node.action] = Int32(n))

W(node::Node) = node.parent.child_W[node.action]
W!(node::Node, n) = (node.parent.child_W[node.action] = Float32(n))

child_Q(node::Node) = node.child_W ./ (1 .+ node.child_N)

child_U(node::Node) = sqrt(N(node)) .* node.child_P ./ (1 .+ node.child_N)

function best_action(mcts::MCTS, node::Node)::Int
    child_score = nothing
    if mcts.tree_policy === "PUCT"
        child_score = child_Q(node) + node.mcts.c_uct * child_U(node)
    elseif mcts.tree_policy === "UCB"
        child_score = child_Q(node) .+ 2 .* mcts.c_uct .*
            sqrt.(2 * log(N(node)) ./ (node.child_N .+ 1))
    else
        error("Invalid tree policy " + mcts.tree_policy)
    end
    masked_child_score = child_score
    masked_child_score[.~node.valid_actions] .= typemin(Float32)
    return argmax_rand(masked_child_score)
end

function select_leaf(mcts::MCTS, node::Node)::Node
    current_node = node
    while current_node.is_expanded
        action = best_action(mcts, current_node)
        current_node = get_child(mcts, current_node, action)
    end
    return current_node
end

function expand(node::Node, child_P)
    node.is_expanded = true
    node.child_P = child_P
end

function get_child(mcts::MCTS, node::Node, action::Int) :: Node
    @assert 1 <= action <= action_space_size(mcts.env)
    if node.children[action] == nothing
        set_state!(mcts.env, node.state)
        obs, reward, done = step!(mcts.env, action)
        next_state = get_state(mcts.env)
        node.children[action] = Node{typeof(node.state)}(mcts.env, action, next_state,
            obs, done, reward, node)
    end
    return node.children[action]
end

function backup(node::Node, value)
    current = node
    while current.parent != nothing
        N!(current, N(current) + 1)
        W!(current, W(current) + value)
        current = current.parent
    end
end

"""
    rollout!(mcts, leaf)

Do a rollout always taking random actions according to the prior probabilities
until the episode is done, return reward.

The episode is not done in the current leaf, i.e., at least one action can be performed.
"""
function rollout!(mcts::MCTS, leaf::Node)
    value = leaf.reward
    env = mcts.env
    set_state!(env, leaf.state)
    done = false
    obs = leaf.obs
    sigma = length(obs.valid_actions)
    while !done
        # Sample one action
        action = StatsBase.sample(Vector(1:sigma)[obs.valid_actions],
            Weights(leaf.child_P[obs.valid_actions]))
        obs, reward, done = step!(env, action)
        value += reward
    end
    return value
end


"""
    MCTS(env)

Create MCTS, i.e., root node and reset environment
"""
function MCTS{TEnv}(env::TEnv) where {TEnv <: Environment}
    root_obs = reset!(env)
    root_state = get_state(env)
    TState = typeof(root_state)
    root_parent = Node{TState}(env, 1, root_state, root_obs, false, 0, nothing)
    root = Node{TState}(env, 1, root_state, root_obs, false, 0, root_parent)
    MCTS(settings[:mh_mcts_num_sims], settings[:mh_mcts_c_uct],
        settings[:mh_mcts_tree_policy], env, root)
end

"""
    compute_priors_and_value(mcts, obs)

Evaluate observed state and return priors P(s,a) for all actions and est. state value Q(s).

So far only constant values 0.5 are returned.
This method is supposed to be extended with e.g. a neural network or some problem specific
heuristic.
"""
function compute_priors_and_value(mcts::MCTS, obs::Observation)
    sigma = length(obs.valid_actions)
    # return rand(action_space_size), rand(Float32)
    return fill(1.0 / sigma, sigma), 0.5
end

"""
    perform_MCTS!(mcts)

Perform MCTS by running episodes from the current root node.

Finally return best action from root, which is the subnode most often visited.
"""
function perform_mcts!(mcts::MCTS) :: Integer
    for i in 1:mcts.num_sims
        leaf = select_leaf(mcts, mcts.root)
        if leaf.done
            value = leaf.reward
        else
            # evaluate leaf node and expand
            child_priors, value = compute_priors_and_value(mcts, leaf.obs)
            value = rollout!(mcts, leaf)
            expand(leaf, child_priors)
        end
        backup(leaf, value)
    end
    println(mcts.root.child_N)
    return argmax_rand(mcts.root.child_N)
end



end  # module
