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

export MCTS, perform_mcts!, get_child, set_function!

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
    "--mh_mcts_gamma"
        help = "MCTS discout factor for rewards"
        arg_type = Float64
        default = 1.0
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
- `V`: Predicted value for non-terminal nodes and 0 for terminal nodes
- `done`: Indicates end of episode
- `state`: State corresponding to this node
- `obs`: Observation at the state of this node
"""
mutable struct Node{TState <: State}
    action::Int
    is_expanded::Bool
    parent::Union{Node{TState}, Nothing}
    children::Vector{Union{Node{TState}, Nothing}}
    child_W::Vector{Float32}
    child_P::Vector{Float32}
    child_N::Vector{Int32}
    valid_actions::Vector{Bool}
    reward::Float32
    V::Float32
    done::Bool
    state::TState
    obs::Observation
end


"""
    MCTS

Monte Carlo Tree Search.

- tree_policy: PUCT or UCB
"""
mutable struct MCTS{TEnv <: Environment}
    num_sims::Int
    c_uct::Float64
    tree_policy::String
    gamma::Float64

    env::TEnv
    root::Node
    default_policy::Function
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
        settings[:mh_mcts_tree_policy], settings[:mh_mcts_gamma], env, root,
        compute_priors_random)
end


"""
    setFunction!(mcts, default_policy)

Sets the default_policy of the MCTS to the given default_policy.
Default: compute_priors_random: uniform distributed priors
default_policy must have two arguments: An MCTS object and an Observation object
TODO: Argumente von default_policy prüfen
"""

function set_function!(mcts::MCTS, default_policy::Function)
    mcts.default_policy = default_policy
end


function Node{TState}(env::Environment, action::Int, state::TState, obs::Observation,
    done::Bool, reward, parent::Union{Node, Nothing}) where {TState <: State}
    n_actions = action_space_size(env)
    return Node{TState}(action, false, parent, Vector{Union{Node, Nothing}}(nothing,
        n_actions), zeros(Float32, n_actions), zeros(Float32, n_actions),
        zeros(Float32, n_actions), copy(obs.valid_actions), reward, 0, done,
        deepcopy(state), obs)
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
    res = res * "\n  V: " * Base.string(node.V)
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

function best_action(node::Node, tree_policy::String, c_uct)::Int
    child_score = nothing
    if tree_policy === "PUCT"
        child_score = child_Q(node) + c_uct * child_U(node)
    elseif tree_policy === "UCB"
        child_score = child_Q(node) .+ 2 .* c_uct .*
            sqrt.(2 * log(N(node)) ./ (node.child_N .+ 1))
    else
        error("Invalid tree policy " * tree_policy)
    end
    masked_child_score = child_score
    masked_child_score[.~node.valid_actions] .= typemin(Float32)
    return argmax_rand(masked_child_score)
end

function select_leaf(node::Node, env::Environment, tree_policy::String, c_uct)::Node
    current_node = node
    while current_node.is_expanded
        action = best_action(current_node, tree_policy, c_uct)
        current_node = get_child(env, current_node, action)
    end
    return current_node
end

function expand(node::Node, child_P)
    node.is_expanded = true
    node.child_P = child_P
end

function get_child(env::Environment, node::Node, action::Int) :: Node
    @assert 1 <= action <= action_space_size(env)
    if node.children[action] == nothing
        set_state!(env, node.state)
        obs, reward, done = step!(env, action)
        next_state = get_state(env)
        node.children[action] = Node{typeof(node.state)}(env, action, next_state,
            obs, done, reward, node)
    end
    return node.children[action]
end

function backup(node::Node, gamma)
    R = node.V
    while node.parent != nothing
        R = node.reward + gamma * R
        N!(node, N(node) + 1)
        W!(node, W(node) + R)
        node = node.parent
    end
end

"""
    rollout!(mcts, leaf)

Do a rollout always taking random actions according to the prior probabilities
until the episode is done, return reward.

The episode is not done in the current leaf, i.e., at least one action can be performed.
"""
function rollout!(mcts::MCTS, leaf::Node; trace::Bool = false)
    value = leaf.reward
    env = mcts.env
    set_state!(env, leaf.state)
    done = false
    obs = leaf.obs
    sigma = length(obs.valid_actions)
    child_priors = leaf.child_P

    while !done
        # Sample one action according to the current priors
        action = StatsBase.sample(Vector(1:sigma)[obs.valid_actions],
            Weights(child_priors[obs.valid_actions]))

if trace
println("rollout!: child_priors: ", string(child_priors), " Action: ", action)
println("    ", env.state)
end

        # Apply a step with the given action
        obs, reward, done = step!(env, action)
        value += reward

        # If not done, recompute the priors according to the current observation
        if !done
            child_priors, V = compute_priors_and_value(mcts, obs)
        end
    end
    return value
end



"""
    compute_priors_and_value_random(mcts, obs)

Compute

"""

function compute_priors_random(mcts::MCTS, obs::Observation)
    n_actions = length(obs.valid_actions)
    # return fill(1.0 / n_actions, n_actions)
    return fill(1.0, n_actions)
end


"""
    compute_priors_and_value(mcts, obs)

Evaluate observed state and return priors P(s,a) for all actions and est. state value Q(s).

So far only constant values 0.5 are returned.
This method is supposed to be extended with e.g. a neural network or some problem specific
heuristic.
"""
function compute_priors_and_value(mcts::MCTS, obs::Observation)
    priors = mcts.default_policy(mcts, obs)
    return priors, 0.5
end

"""
    perform_MCTS!(mcts)

Perform MCTS by running episodes from the current root node.

Finally return best action from root, which is the subnode most often visited.
"""
function perform_mcts!(mcts::MCTS; trace::Bool = false) :: Integer
    for i in 1:mcts.num_sims
        leaf = select_leaf(mcts.root, mcts.env, mcts.tree_policy, mcts.c_uct)
        if !leaf.done
            # Compute priors for the first time
            child_priors, V = compute_priors_and_value(mcts, leaf.obs)
            leaf.child_P = child_priors   # TODO: Must be set!?

            # evaluate leaf node and expand
            V = rollout!(mcts, leaf; trace = trace)
            leaf.V = V
            expand(leaf, child_priors)
        end
        backup(leaf, mcts.gamma)
    end
    return argmax_rand(mcts.root.child_N)
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

    i = 0

    while (!root.done)
print(i += 1, " ")
        append!(actions, compute_action!(mcts, root))
# println(string(root))
        root = get_child(root, actions[length(actions)])
# println(actions)
    end

    return actions
end


end  # module
