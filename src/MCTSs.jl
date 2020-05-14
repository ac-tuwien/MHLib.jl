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

export Environment, Node, State, MCTS, mcts!

@add_arg_table! settings_cfg begin
    "--mh_mcts_sims"
        help = "MCTS number of simulations"
        arg_type = Int
        default = 1000
end


"""
    State

Abstract type for a state in the environment on which an optimization/learning agent acts.
"""
abstract type State end


"""
    Observation

Observation of a state in the environment, from which predictions are made.

Attributes
- values::Vector{Float32}: Observed values
- valid_actions::Vector{Bool}: Boolean vector indicating valid actions
"""
struct Observation
    values::Vector{Float32}
    valid_actions::Vector{Bool}
end


"""
    Environment

Abstract type for an environment on which an optimization or learning agent acts.

Abstract methods
- `action_space_size(env)::Int`: Size of action space
- `reset(env)::Observation`: Reset environment to initial state and return observation
- `get_state(env)::State`: Return current state
- `set_state!(env, state::State)`: Set state
- `step!(env, action)`: Perform action in environment and
    return new observation, reward and a Bool indicating end of episode
"""
abstract type Environment end

action_space_size(env::Environment)::Int =
    error("abstract action_space_size(env) called")

reset!(env::Environment)::Observation = error("abstract reset!(env) called")

get_state(env::Environment)::State = error("abstract get_state(env) called")

set_state!(env::Environment, state::State) = error("abstract set_state!(env) called")

step!(env::Environment, action::Int)::(Observation, Float32, Bool) =
    error("abstract step!(env, action) called")


#--------------------------------------------------------------------------------

"""
    MCTS

Monte Carlo Tree Search.

TODO: Parameters should be provided by settings.jl.
"""
mutable struct MCTS
    num_sims::Int
    c_puct::Float32
end

MCTS() = MCTS(settings[:mh_mcts_sims], 5)

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
- `W`: Total values of child nodes
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

N(node::Node) = node.parent.child_N[node.action]
N!(node::Node, n) = (node.parent.child_N[node.action] = Int32(n))

W(node::Node) = node.parent.child_W[node.action]
W!(node::Node, n) = (node.parent.child_W[node.action] = Float32(n))

child_Q(node::Node) = node.child_W ./ node.child_N

child_U(node::Node) = sqrt(N(node)) .* node.child_P ./ (1 .+ node.child_N)

function best_action(node::Node)::Int
    child_score = child_Q(node) + node.mcts.c_puct * child_U(node)
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
        W!(current, W(current) + value)
        current = current.parent
    end
end

"""
    naive_rollout!(leaf)

Do a naive rollout always taking random actions until the episode is done, return reward.

The episode is not done in the current leaf, i.e., at least one action can be performed.
"""
function naive_rollout!(leaf::Node)
    value = leaf.reward
    env = leaf.env
    set_state!(env, leaf.state)
    done = false
    obs = leaf.obs
    sigma = length(obs.valid_actions)
    while !done
        action = rand(Vector(1:sigma)[obs.valid_actions])
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
    return fill(0.5, sigma), 0.5
end

"""
    compute_action!(mcts, node)

Perform MCTS by running episodes, considering the given node as root.
Finally return best action from root, which is the subnode most often visited.
"""
function compute_action!(mcts::MCTS, node::Node)
    for i in 1:mcts.num_sims
        leaf = select_leaf(node)
        if leaf.done
            value = leaf.reward
        else
            # evaluate leaf node and expand
            child_priors, value = compute_priors_and_value(mcts, leaf.obs)
            value = naive_rollout!(leaf)
            expand(leaf, child_priors)
        end
        backup(leaf, value)
    end
    return argmax(node.child_N)
end

"""
    run!(mcts, env)

Perform MCTS.

Create root node, perform simulations and return best action from root.
"""
function mcts!(mcts::MCTS, env::Environment)
    root_obs = reset!(env)
    root_state = get_state(env)
    root_parent = Node(mcts, env, 1, root_state, root_obs, false, 0, nothing)
    root = Node(mcts, env, 1, root_state, root_obs, false, 0, root_parent)
    compute_action!(mcts, root)
end


end  # module
