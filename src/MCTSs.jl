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


@add_arg_table! settings_cfg begin
    "--mh_mcts_num_sims"
        help = "MCTS number of simulations"
        arg_type = Int
        default = 100
    "--mh_mcts_c_puct"
        help = "MCTS c_puct"
        arg_type = Float64
        default = 1.0
    "--mh_mcts_tree_policy"
        help = "Tree Policy to apply: UCB (default) or PUCT"
        arg_type = String
        default = "UCB"
end


"""
    MCTS

Monte Carlo Tree Search.
"""
mutable struct MCTS
    num_sims::Int
    c_puct::Float64
    tree_policy::String
end

MCTS() = MCTS(settings[:mh_mcts_num_sims], settings[:mh_mcts_c_puct], settings[:mh_mcts_tree_policy])

"""
    Node

A node of the MCTS tree.

Attributes
- `mcts`: MCTS instance
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
mutable struct Node{TEnv <: Environment, TState <: State}
    mcts::MCTS
    env::TEnv
    action::Int
    is_expanded::Bool
    parent::Union{Node{TEnv, TState}, Nothing}
    children::Vector{Union{Node{TEnv, TState}, Nothing}}
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

function Node(mcts::MCTS, env::TEnv, action::Int, state::TState,
    obs::Observation, done::Bool, reward, parent::Union{Node, Nothing}) where
    {TEnv <: Environment, TState <: State}
    sigma = action_space_size(env)
    return Node{TEnv, TState}(
            mcts, env, action, false, parent,
            Vector{Union{Node, Nothing}}(nothing, sigma),
            sigma, zeros(Float32, sigma), zeros(Float32, sigma), zeros(Float32, sigma),
            copy(obs.valid_actions), reward, done, deepcopy(state), obs)
end


function Base.string(node::Node)
    res = "** Current Node:"
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

child_Q(node::Node) = node.child_W ./ node.child_N

child_U(node::Node) = sqrt(N(node)) .* node.child_P ./ (1 .+ node.child_N)

function best_action(node::Node)::Int
    child_score = nothing
    if node.mcts.tree_policy == "PUCT"
        child_score = child_Q(node) + node.mcts.c_puct * child_U(node)
    elseif node.mcts.tree_policy == "UCB"
        temp = 2 * log(sum(N(node)))
        child_score = child_Q(node) .+ 2 .* node.mcts.c_puct .* sqrt(temp ./ N(node))
    else error("Tree Policy not correctly defined!")
    end

    masked_child_score = child_score
    masked_child_score[.~node.valid_actions] .= typemin(Float32)
    return argmax(masked_child_score)
end

function select_leaf(node::Node)::Node
    current_node = node
    while current_node.is_expanded
        action = best_action(current_node)
        current_node = get_child(current_node, action)
    end
    return current_node
end

function expand(node::Node, child_P)
    node.is_expanded = true
    node.child_P = child_P
end

function get_child(node::Node, action::Int) :: Node
    @assert 1 <= action <= action_space_size(node.env)
    if node.children[action] == nothing
        set_state!(node.env, node.state)
        obs, reward, done = step!(node.env, action)
        next_state = get_state(node.env)
        node.children[action] = Node(node.mcts, node.env, action, next_state,
                                     obs, done, reward, node)
    end
    return node.children[action]
end

function backpropagate(node::Node, value)
    current = node
    while current.parent != nothing
        N!(current, N(current) + 1)
        W!(current, W(current) + value)
        current = current.parent
    end
end



"""
    rollout!(leaf)

Do a rollout always taking random actions according to the prior probabilities
until the episode is done, return reward.

The episode is not done in the current leaf, i.e., at least one action can be performed.
"""
function rollout!(leaf::Node)
    value = leaf.reward
    env = leaf.env
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
    compute_priors_and_value(mcts, obs)

Evaluate observed state and return priors P(s,a) for all actions and est. state value Q(s).

So far only constant values 0.5 are returned.
This method is supposed to be extended with e.g. a neural network or some problem specific
heuristic.
"""
function compute_priors_and_value(mcts::MCTS, obs::Observation)
    sigma = length(obs.valid_actions)
    # return rand(action_space_size), rand(Float32)
    return fill(1.0/sigma, sigma), 0.5
end

"""
    compute_action!(mcts, node)

Perform MCTS by running episodes, considering the given node as root.
Finally return best action from root, which is the subnode most often visited.
"""
function compute_action!(mcts::MCTS, node::Node) :: Integer
    for i in 1:mcts.num_sims
        leaf = select_leaf(node)
        if leaf.done
            value = leaf.reward
        else
            # evaluate leaf node and expand
            child_priors, value = compute_priors_and_value(mcts, leaf.obs)
            value = rollout!(leaf)
            expand(leaf, child_priors)
        end
        backpropagate(leaf, value)
    end
    return argmax(node.child_N)
end

"""
    mcts!(mcts, env)

Perform MCTS.

Create root node, perform simulations and return all performed actions as Array.
"""
function mcts!(mcts::MCTS, env::Environment)
    root_obs = reset!(env)
    root_state = get_state(env)
    root_parent = Node(mcts, env, 1, root_state, root_obs, false, 0, nothing)

    root = Node(mcts, env, 1, root_state, root_obs, false, 0, root_parent)
    actions = Int[]

# println(string(root))

    while (!root.done)
        append!(actions, compute_action!(mcts, root))
println(string(root))
        root = get_child(root, actions[length(actions)])
println(actions)
    end

    return actions
end


end  # module
