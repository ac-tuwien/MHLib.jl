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

const settings_cfg = ArgParseSettings()

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
        default = "PUCT"
    "--mh_mcts_gamma"
        help = "MCTS discout factor for rewards"
        arg_type = Float64
        default = 1.0
    "--mh_mcts_rollout_policy"
        help = "Rollout Policy to apply: random (according to priors) or epsilon-greedy"
        arg_type = String
        default = "random"
    "--mh_mcts_epsilon_greedy_epsilon"
        help = "If epsilon-greedy is used, what value of epsilon in [0, 1] should be used?"
        arg_type = Float64
        default = 0.2
    "--mh_mcts_child_criterion"
        help = "After which criterion should the action at the root selected?
                robust_child: select the most visited child.
                exp_visit_count: select the child randomly according to exponentiated visit counts"
        arg_type = String
        default = "robust_child"
    "--mh_mcts_exp_visit_counts_temp"
        help = "Temperature for exponentiated visit count criterion"
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
- `child_W`: Total values of child nodes
- `child_P`: Priors of child nodes
- `child_N`: Number of visits of child nodes
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
- rollout_policy: random (according to priors) or epsilon-greedy
- epsilon: Value of epsilon in epsilon-greedy strategy
"""
mutable struct MCTS{TEnv <: Environment}
    num_sims::Int
    c_uct::Float64
    tree_policy::String
    gamma::Float64

    env::TEnv
    root::Node
    best_solution::Vector{Int}

    rollout_policy::String
    epsilon::Float64

    child_criterion::String
    exp_visit_counts_temp::Float64
end


function Node{TState}(env::Environment, action::Int, state::TState, obs::Observation,
    done::Bool, reward, parent::Union{Node, Nothing}) where {TState <: State}
    n_actions = action_space_size(env)
    return Node{TState}(action, false, parent, Vector{Union{Node, Nothing}}(nothing,
        n_actions), zeros(Float32, n_actions), zeros(Float32, n_actions),
        zeros(Float32, n_actions), reward, 0, done,
        deepcopy(state), deepcopy(obs))
end

function Base.string(node::Node)
    res = "Node:"
    res = res * "\n  action = $(node.action)"
    res = res * "\n  child_N: " * Base.string(node.child_N)
    res = res * "\n  child_W: " * Base.string(node.child_W)
    res = res * "\n  child_Q: " * Base.string(child_Q(node))
    res = res * "\n  child_P: " * Base.string(node.child_P)
    res = res * "\n  reward: " * Base.string(node.reward)
    res = res * "\n  V: " * Base.string(node.V)
    res = res * "\n  done: " * Base.string(node.done)
    res = res * "\n" * Base.string(node.state)
    res = res * "\n" * Base.string(node.obs)
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
    masked_child_score[.~node.obs.action_mask] .= typemin(Float32)
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
        set_state!(env, node.state, node.obs)
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
        settings[:mh_mcts_tree_policy], settings[:mh_mcts_gamma], env, root, Int[],
        settings[:mh_mcts_rollout_policy], settings[:mh_mcts_epsilon_greedy_epsilon],
        settings[:mh_mcts_child_criterion], settings[:mh_mcts_exp_visit_counts_temp])
end

"""
    rollout!(mcts, leaf)

Do a naive rollout always taking random actions until the episode is done, return reward.

The episode is not done in the current leaf, i.e., at least one action can be performed.
"""
function rollout!(mcts::MCTS, leaf::Node; trace::Bool = false)
    value = leaf.reward
    env = mcts.env
    set_state!(env, leaf.state, leaf.obs)
    done = false
    obs = leaf.obs
    n_actions = length(obs.action_mask)
    child_priors = leaf.child_P
    solution = Int[]

    while !done
        if length(obs.priors) > 0
            if mcts.rollout_policy === "random"
                # Sample one action according to the current priors
                action = StatsBase.sample(Vector(1:n_actions)[obs.action_mask],
                    Weights(obs.priors[obs.action_mask]))
            elseif mcts.rollout_policy === "epsilon-greedy"
                @assert 0 <= mcts.epsilon <= 1
                if rand() < mcts.epsilon
                    # Sample one action completely at random
                    action = rand(Vector(1:n_actions)[obs.action_mask])
                else
                    # Take the action with one highest prior value
                    masked_priors = obs.priors[:]
                    masked_priors[.~obs.action_mask] .= typemin(Float32)
                    action = argmax_rand(masked_priors)
# println(string(masked_priors), " ", string(obs.priors), " Action: ", action)
                end
            else
                error("Invalid mcts.rollout_policy " * mcts.rollout_policy)
            end
        else
            # If the priors are uniform => espilon-greedy makes no sense
            action = rand(Vector(1:n_actions)[obs.action_mask])
        end

        if trace
            println("rollout!: child_priors: ", string(child_priors), ", priors: ",
                obs.priors, ", Action: ", action)
            println(string(leaf))
        end

        # Apply a step with the given action
        obs, reward, done = step!(env, action)
        append!(solution, action)
        value += reward
    end
    # TODO should be rebplaced by generic reward check
    if length(solution)+length(leaf.state.s) > length(mcts.best_solution)
        copy!(mcts.best_solution, [leaf.state.s; solution])
    end
    return value
end


"""
    perform_MCTS!(mcts)

Perform MCTS by running episodes from the current root node.

Finally return best action from root, which is the subnode most often visited
(child_criterion = robust_child) or one random action according to the
exponentiated visit counts (child_criterion = exp_visit_count).
"""
function perform_mcts!(mcts::MCTS; trace::Bool = false) :: Integer
    for i in 1:mcts.num_sims
        leaf = select_leaf(mcts.root, mcts.env, mcts.tree_policy, mcts.c_uct)

        if !leaf.done
# println("\n  Leaf Not Done")
            # child_priors, V = compute_priors_and_value(mcts, leaf.obs)
            child_priors = leaf.obs.priors
            if length(child_priors) == 0
                child_priors = leaf.obs.action_mask / sum(leaf.obs.action_mask)
            end

            # evaluate leaf node and expand
            V = rollout!(mcts, leaf; trace = trace)
            leaf.V = V
            expand(leaf, child_priors)
        else
            # TODO should be rebplaced by generic reward check
            solution = leaf.state.s
            if length(solution) > length(mcts.best_solution)
                copy!(mcts.best_solution, solution)
            end
        end
        backup(leaf, mcts.gamma)
    end

    if mcts.child_criterion === "robust_child"
        return argmax_rand(mcts.root.child_N)
    elseif mcts.child_criterion === "exp_visit_count"
        weights = mcts.root.child_N .^ (1 / exp_visit_counts_temp)
        return StatsBase.sample(Vector(1:n_actions)[obs.action_mask], weights)
            #Weights(weights))
    else
        error("invalid child_criterion!")
    end
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
