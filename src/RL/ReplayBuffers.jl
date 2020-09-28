using StatsBase

export ReplayBuffer, ReplayBufferAdder, add!, flush!

"""
ReplayBuffer(buffer_size, obs_space_size, action_space_size)

A FIFO buffer holding at most `max_size` tuples of data for actions in episodes
from which learning is performed.

Attributes
- `max_size`: maximum size of the replay buffer
- `current_size`: current size of the replay buffer
- `oldest`: index of oldest entry
- `obs_values`: observation values before applied actions
- `action_masks`: respective action masks
- `actions`: actions taken
- `policies`: corresponding policies; each row represents one policy
- `targets`: obsered target values
"""
mutable struct ReplayBuffer
    max_size::Int
    current_size::Int
    oldest::Int

    obs_values::Array{Float32, 2}
    action_masks::Array{Bool, 2}
    actions::Vector{Int}
    policies::Array{Float32, 2}
    targets::Vector{Float32}

    function ReplayBuffer(buffer_size, obs_space_size, action_space_size)
        obs_vals = Array{Float32}(undef, buffer_size, obs_space_size)
        masks = Array{Bool}(undef, buffer_size, action_space_size)
        act = Vector{Int}(undef, buffer_size)
        pol = Array{Float32}(undef, buffer_size, action_space_size)
        targ = Vector{Float32}(undef, buffer_size)
        new(buffer_size, 0, 1, obs_vals, masks, act, pol, targ)
    end
end

"""
    append!(buffer, obs_values, action_masks, actions, policies, targets)

Append provided episode data to the replay buffer. If the replay buffer
is full, the oldest elements are deleted in a FIFO manner.

Parameters
- `buffer`: the replay buffer
- `obs_values`: observations before applied actions
- `action_masks`: respective action masks
- `actions`: newly taken actions
- `policies`: corresponding policies
- `targets`: corresponding target values
"""
function append!(buffer::ReplayBuffer, obs_values::Array{<:AbstractFloat, 2},
    actions::Vector{Int}, policies::Array{<:AbstractFloat, 2}, targets::Vector{Int})

    @assert size(obs_values, 1) == size(action_masks, 1) == length(actions) ==
         size(policies, 1) == length(targets)

    for i in length(actions)
        buffer.obs_values[oldest] = obs_values[i]
        buffer.action_masks[oldest] = action_masks[i]
        buffer.actions[oldest] = actions[i]
        buffer.policies[oldest] = policies[i]
        buffer.targets[oldest] = targets[i]

        oldest += 1
        if oldest > buffer.max_size
            oldest = 1
        end
        if buffer.current_size < buffer.max_size
            buffer.current_size += 1
        end
    end
end

"""
    sample(replay_buffer, n)

Return `n` randomly sampled data from the replay buffer as tuple
(obs_values, action_masks, actions, policies, targets)
"""
function sample(buffer::ReplayBuffer, n::Int)
    ind = sample(1:buffer.current_size, n, replace=false)

    # TODO Unnötiges Kopieren vermeiden, einen zB einen View retournieren!
    obs_values = buffer.obj_values[ind, :]
    action_masks = buffer.action_masks[ind, :]
    actions = buffer.actions[ind]
    policies = buffer.policies[ind, :]
    targets = buffer.targets[ind, :]
    return (obs_values, action_masks, actions, policies, targets)
end


"""
    ReplayBufferAdder(replay_buffer)

Aggregate data of actor for a whole episode before appending all to replay buffer.
"""
mutable struct ReplayBufferAdder
    buffer::ReplayBuffer
    obs_values::Vector{Vector{Float32}}
    action_masks::Vector{Vector{Bool}}
    actions::Vector{Int}
    policies::Vector{Vector{Float32}}
    rewards::Vector{Float32}

    function ReplayBufferAdder(replay_buffer::ReplayBuffer)
        buffer = replay_buffer
        obs_values = Vector{Vector{Float32}}()
        action_masks = Vector{Vector{Bool}}()
        actions = Vector{Int}()
        policies = Vector{Vector{Float32}}()
        rewards = Vector{Float32}()
        new(buffer, obs_values, action_masks, actions, policies, rewards)
    end
end

"""
    add!(replay_buffer_adder, observation, action_mask, action, policy, reward)

Add data of performed action to the adder cache.
"""
function add!(adder::ReplayBufferAdder, observation::Observation, action::Int,
        policy::Vector{Float32}, reward::Float32)
    append!(adder.obs_values, observation.values)
    append!(adder.action_masks, observation.action_mask)
    append!(adder.actions, action)
    append!(adder.policies, policy)
    append!(adder.rewards, reward)
end

"""
    flush!(replay_buffer_adder)

Calculate target values from received rewards and write all to the replay buffer.
"""
function flush!(adder::ReplayBufferAdder)
    targets = similar(rewards)
    targets[end] = rewards[end]
    for i in length(rewards)-1:1
        targets[i] = targets[i+1] + rewards[i]
    end
    append!(buffer, obs_values, action_masks, actions, policies, targets)
    empty!(obs_values)
    empty!(action_masks)
    empty!(actions)
    empty!(policies)
    empty!(targets)
end
