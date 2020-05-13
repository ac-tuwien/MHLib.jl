#=
    MCTSs

Monte Carlo Tree Search (MCTS).

This module realizes Monte Carlo Tree Search.
This implementation is inspired by
https://github.com/ray-project/ray/blob/master/rllib/contrib/alpha_zero/core/mcts.py

=#
module MCTSs

using MHLib

export Environment, Node, State, MCTS, run!, action_space_size, step!


"""
    Observation

An observation at a state in the environment, from which predictions are made.

Attributes
- values::Vector{Float32}: Observed values
- valid_actions::Vector{Bool}: True for each action that is valid in the current state
"""
struct Observation
    values::Vector{Float32}
    valid_actions::Vector{Bool}
end


"""
    State

    Abstract type for a state in the environment.
"""
abstract type State end


"""
    Environment

Abstract type for an environment on which MCTS acts.

Must implement:
- action_space_size(end)::Int: Size of action space
- get_state(env)::State: Return current state
- set_state!(env, state::State): Set state
- get_obs(env)::Observation: Return current observation
- step!(env, action): Perform action in environment and
    return new observation, reward and a Bool indicating end of episode
"""
abstract type Environment end

action_space_size(env::Environment)::Int =
    error("abstract action_space_size(env) called")

get_state(env::Environment)::State = error("abstract get_state(env) called")

set_state!(env::Environment, state::State) = error("abstract set_state(env) called")

get_obs(env::Environment)::Observation = error("abstract get_obs(env) called")

step!(env::Environment, action::Int)::(Observation, Float32, Bool) =
    error("abstract step!(env, action) called")


"""
    MCTS

Monte Carlo Tree Search.
"""
mutable struct MCTS
    num_sims::Int
    c_puct::Float32
end

MCTS() = MCTS(1000, 5)


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
- `Q`: Total values of child nodes
- `P`: Priors of child nodes
- `N`: Number of visits of child nodes
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
    parent::Union{Node, Nothing}
    children::Vector{Union{Node, Nothing}}
    action_space_size::Int
    child_Q::Vector{Float32}
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

N(node::Node) = node.parent.child_N[node.action]
N!(node::Node, n) = (node.parent.child_N[node.action] = Int32(n))

Q(node::Node) = node.parent.child_Q[node.action]
Q!(node::Node, n) = (node.parent.child_Q[node.action] = Float32(n))

child_Q_rel(node::Node) = node.child_Q ./ (1 .+ node.child_N)

child_U(node::Node) = sqrt(N(node)) .* node.child_P ./ (1 .+ node.child_N)

function best_action(node::Node)::Int
    child_score = child_Q_rel(node) + node.mcts.c_puct * child_U(node)
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

function get_child(node::Node, action::Int)::Node
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

function backup(node::Node, value)
    current = node
    while current.parent != nothing
        N!(current, N(current) + 1)
        Q!(current, Q(current) + value)
        current = current.parent
    end
end


function compute_priors_and_value(mcts::MCTS, obs::Observation, action_space_size::Int)
    # return rand(action_space_size), rand(Float32)
    return fill(0.5, action_space_size), 0.5
end


function compute_action(mcts::MCTS, node::Node)
    for i in 1:mcts.num_sims
        leaf = select_leaf(node)
        if leaf.done
            value = leaf.reward
        else
            child_priors, value = compute_priors_and_value(mcts, leaf.obs,
                node.action_space_size)
            expand(leaf, child_priors)
        end
        backup(leaf, value)
    end
    return argmax(node.child_N)
end

function run!(mcts::MCTS, env::Environment)
    root_state = get_state(env)
    root_obs = get_obs(env)
    root_parent = Node(mcts, env, 1, root_state, root_obs, false, 0, nothing)
    root = Node(mcts, env, 1, root_state, root_obs, false, 0, root_parent)
    compute_action(mcts, root)
end



end  # module
