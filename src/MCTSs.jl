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

export MCTS, perform_mcts!, get_child, set_function!, set_new_root!

const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--mh_mcts_num_sims"
        help = "MCTS number of simulations"
        arg_type = Int
        default = 100
    "--mh_mcts_c_uct"
        help = "MCTS c_uct coefficient for tree policy"
        arg_type = Float64
        default = 1.0
    "--mh_mcts_tree_policy"
        help = "MCTS tree policy to apply: UCB or PUCT"
        arg_type = String
        default = "PUCT"
    "--mh_mcts_gamma"
        help = "MCTS discount factor for rewards"
        arg_type = Float64
        default = 1.0
    "--mh_mcts_rollout_policy"
        help = "MCTS rollout policy: random or epsilon-greedy"
        arg_type = String
        default = "random"
    "--mh_mcts_epsilon_greedy_epsilon"
        help = "MCTS epsilon for epsilon-greedy rollout strategy"
        arg_type = Float64
        default = 0.2
    "--mh_mcts_visit_counts_policy_temp"
        help = "MCTS temperature for visit count policy to select action; 0: greedy"
        arg_type = Float64
        default = 0.0
    "--mh_mcts_reuse_subtrees"
        help = "MCTS: Reuse subtrees after performing an action"
        arg_type = Bool
        default = true
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

    function Node(env::Environment, action::Int, state::TState, obs::Observation,
        done::Bool, reward, parent::Union{Node, Nothing}) where {TState <: State}
        n_actions = action_space_size(env)
        new{TState}(action, false, parent, Vector{Union{Node, Nothing}}(nothing,
            n_actions), zeros(Float32, n_actions), zeros(Float32, n_actions),
            zeros(Float32, n_actions), reward, 0, done,
            deepcopy(state), deepcopy(obs))
    end
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

"""
    best_action(node, tree_policy, c_uct)

Select best action according to the provided tree policy.
"""
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

"""
    select_leaf(node, env, tree_policy, c_uct)

Traverse tree down from current until until a leaf is reached according
to the provided tree policy.
"""
function select_leaf(node::Node, env::Environment, tree_policy::String, c_uct)::Node
    current_node = node
    while current_node.is_expanded
        action = best_action(current_node, tree_policy, c_uct)
        current_node = get_child(env, current_node, action)
    end
    return current_node
end

"""
    expand(node, child_P)

The given node is expanded and initialized with the given prior `child_P`.
"""
function expand(node::Node, child_P)
    node.is_expanded = true
    node.child_P = child_P
end

"""
    get_child(env, node, action)

Return child node of the given node w.r.t. the given action.

If the child node does not yet exist the corresponding step is performed in
the environment and a new node initialized.
"""
function get_child(env::Environment, node::Node, action::Int) :: Node
    @assert 1 <= action <= action_space_size(env)
    if node.children[action] == nothing
        set_state!(env, node.state, node.obs)
        obs, reward, done = step!(env, action)
        next_state = get_state(env)
        node.children[action] = Node(env, action, next_state,  obs, done, reward, node)
    end
    return node.children[action]
end

"""
    backup(node, gamma)

Perform backup by updating the rewards and visit counters of the given node and its
predecessors.
"""
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
    create_root_node(env, obs)

Create MCTS root node including is artificial parent for the current environment and
observation.
"""
function create_root_node(env::Environment, obs::Observation)
    root_obs = obs
    root_state = get_state(env)
    root_parent = Node(env, 1, root_state, root_obs, false, 0, nothing)
    Node(env, 1, root_state, root_obs, false, 0, root_parent)
end


#------------------------------------------------------------------------------

"""
    MCTS{TEnv}

Monte Carlo Tree Search for environment of type `TEnv`.

TODO update docstring
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
    best_action_sequence::Vector{Int}

    rollout_policy::String
    epsilon::Float64

    visit_counts_policy_temp::Float64
    reuse_subtrees::Bool

    policy_value_function::Union{Function, Nothing}

    """
        MCTS(env, observation, policy_value_function=nothing)

    Create MCTS with root node for given environment and initial observation.

    A policy-value function can optionally be provided and is then used for
    evaluating new leaf nodes instead of performing a rollout.
    """
    function MCTS(env::Environment, obs::Observation,
            policy_value_function::Union{Function, Nothing} = nothing)
        root = create_root_node(env, obs)
        new{typeof(env)}(settings[:mh_mcts_num_sims], settings[:mh_mcts_c_uct],
            settings[:mh_mcts_tree_policy], settings[:mh_mcts_gamma], env, root, Int[],
            settings[:mh_mcts_rollout_policy], settings[:mh_mcts_epsilon_greedy_epsilon],
            settings[:mh_mcts_visit_counts_policy_temp], settings[:mh_mcts_reuse_subtrees],
            policy_value_function)
    end
end

"""
    rollout!(mcts, leaf)

Perform rollout always taking random actions until the episode is done, return total reward.

The episode is not done in the current leaf, i.e., at least one action can be performed.
"""
function rollout!(mcts::MCTS{TEnv}, leaf::Node; trace::Bool = false) :: Float32 where
        {TEnv <: Environment}
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
            # TODO remove parameter rollout_policy, no priors in observations
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
                    # Take an action with highest prior value
                    masked_priors = obs.priors[:]
                    masked_priors[.~obs.action_mask] .= typemin(Float32)
                    action = argmax_rand(masked_priors)
                end
            else
                error("Invalid mcts_rollout_policy " * mcts.rollout_policy)
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
    if length(solution)+length(leaf.state.s) > length(mcts.best_action_sequence)
        copy!(mcts.best_action_sequence, [leaf.state.s; solution])
    end
    return value
end

"""
    visit_count_policy(root_node, temperature)

Select action according to exponentiated visit counts of children nodes of root node.

If the temperature is zero a greedy selection is performed.
Returns selected action and probability distribution corresponding to normalized
visit counts as policy.
"""
function visit_count_policy(mcts::MCTS{TEnv}, temperature) ::
        Tuple{Int, Vector{Float32}} where {TEnv <: Environment}
    visits = mcts.root.child_N
    if sum(visits) == 0
        visits .+= 1  # uniform policy for zero total visits
    end
    rescaled_visits = visits / sum(visits)
    if temperature == 0  # greedy selection
        action = argmax_rand(mcts.root.child_N)
    else
        if temperature != 1
            weights = visits .^ (1 / temperature)
        else
            weights = visits
        end
        # weights = probs ./ sum(probs)
        action = sample(Vector(1:n_actions)[obs.action_mask], weights)
    end
    return action, rescaled_visits
end

"""
    perform_MCTS!(mcts)

Perform MCTS by running simulations from the current root node.

Return policy, i.e., probability distribution on actions, and selected action
according to `visit_count_policy`.
"""
function perform_mcts!(mcts::MCTS{TEnv}; trace::Bool = false) ::
        Tuple{Int, Vector{Float32}} where {TEnv <: Environment}
    for i in 1:mcts.num_sims
        leaf = select_leaf(mcts.root, mcts.env, mcts.tree_policy, mcts.c_uct)
        if !leaf.done
            # evaluate leaf node and expand
            if mcts.policy_value_function == nothing
                # perform rollout to evaluate leaf
                child_priors = leaf.obs.action_mask / sum(leaf.obs.action_mask)
                V = rollout!(mcts, leaf; trace = trace)
            else
                # policy_value_function given, call it instead of performing a rollout
                child_priors, V = policy_value_function(leaf.obs)
            end
            leaf.V = V
            expand(leaf, child_priors)
        else
            # TODO should be replaced by generic reward check
            # TODO remove solution from s, store sequence of actions in MCTS instead
            solution = leaf.state.s
            if length(solution) > length(mcts.best_action_sequence)
                copy!(mcts.best_action_sequence, solution)
            end
        end
        backup(leaf, mcts.gamma)
    end
    set_state!(mcts.env, mcts.root.state, mcts.root.obs)
    return visit_count_policy(mcts, mcts.visit_counts_policy_temp)
end

"""
    set_new_root!(mcts::MCTS, action, obs)

Set the root node to the current state of the environment obtained from
the original root node's state by the given action yielding the given observation.

In dependence of `mcts.reuse_subtrees`, an existing subtree is either reused or
an entirely new root node is created.
"""
function set_new_root!(mcts::M, action::Int, obs::Observation) where {M <: MCTS}
    if mcts.reuse_subtrees && mcts.root.children[action] != nothing
        mcts.root = mcts.root.children[action]
    else
        mcts.root = create_root_node(mcts.env, obs)
    end
end


end  # module
