using Flux
using Infiltrator

export DummyPolicyValueFunction, DensePolicyValueNN


#------------------------------------------------------------------------------

"""
    DummyPolicyValueFunction(action_space_size)

A `PolicyValueFunction` that just returns a uniform policy and random value.

Used just for testing purposes.
"""
struct DummyPolicyValueFunction <: PolicyValueFunction
    action_space_size::Int
end

function (f::DummyPolicyValueFunction)(obs_values::Vector{Float32},
    action_mask::Vector{Bool})::Tuple{Vector{Float32}, Float32}
    actions = f.action_space_size
    return fill(Float32(1)/actions, actions), Float32(rand(1:actions))
end

function train!(f::DummyPolicyValueFunction, obs_values::AbstractArray{Float32, 2},
    action_masks::AbstractArray{Bool, 2}, actions::AbstractVector{Int},
    policies::AbstractArray{Float32, 2}, targets::AbstractVector{Float32})
    # do nothing
end

#------------------------------------------------------------------------------

"""
    DensePolicyValueNN(obs_space_size, action_space_size, inner_nodes)

A generic dense multi-layer feed-forward neural network (MLP) as `PolicyValueFunction`.

`inner_nodes` is a `Tuple{Int}` specifying the number of nodes in each inner layer.
The network is actually realized by two independent components for the policy and value.
"""
struct DensePolicyValueNN <: PolicyValueFunction
    n_obs::Int
    n_actions::Int
    value_network::Chain
    policy_network::Chain
    opt_value::ADAM
    opt_policy::ADAM

    function DensePolicyValueNN(observation_space_size::Int, action_space_size::Int,
            inner_nodes::Vector{Int}=[50, 50])
        @assert length(inner_nodes) >= 1
        value_inner = [Dense(inner_nodes[i-1], inner_nodes[i], relu)
            for i in 2:length(inner_nodes)]
        value_net = Chain(
            Dense(observation_space_size, inner_nodes[1], relu),
            value_inner...,
            Dense(inner_nodes[end], 1))
        policy_inner = [Dense(inner_nodes[i-1], inner_nodes[i], relu)
            for i in 2:length(inner_nodes)]
        policy_net = Chain(
            Dense(observation_space_size, inner_nodes[1], relu),
            policy_inner...,
            Dense(inner_nodes[end], action_space_size))
        new(observation_space_size, action_space_size, value_net, policy_net,
            Flux.Optimise.ADAM(0.001, (0.9, 0.999)),
            Flux.Optimise.ADAM(0.001, (0.9, 0.999)))
    end
end

function (f::DensePolicyValueNN)(obs_values::Vector{Float32},
    action_mask::Vector{Bool})::Tuple{Vector{Float32}, Float32}
    logits = f.policy_network(obs_values)
    value = f.value_network(obs_values)
    softmax(logits), value[1]
end

function train!(f::DensePolicyValueNN, obs_values::AbstractArray{Float32, 2},
        action_masks::AbstractArray{Bool, 2}, actions::AbstractVector{Int},
        policies::AbstractArray{Float32, 2}, targets::AbstractVector{Float32})

    # println("Train!")
    logits = f.policy_network(obs_values)
    p_params = params(f.policy_network)
    p_gradient = gradient(p_params) do
        Flux.Losses.logitcrossentropy(logits, policies)
    end
    # @infiltrate
    Flux.Optimise.update!(f.opt_policy, p_params, p_gradient)

    values = f.value_network(obs_values)
    v_params = params(f.value_network)
    v_gradient = gradient(v_params) do
        Flux.Losses.mse(values, targets)
    end
    Flux.Optimise.update!(f.opt_value, v_params, v_gradient)
end
