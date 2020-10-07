using StatsBase

export ReplayBuffer, ReplayBufferAdder, add!, flush!

"""
ReplayBuffer(buffer_size, obs_space_size, action_space_size)

A FIFO buffer holding at most `max_size` tuples of data for actions in episodes
from which learning is performed.
The `buffer_size` may also be 0, in which case the buffer is actually not used.

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
        obs_vals = Array{Float32}(undef, obs_space_size, buffer_size)
        masks = Array{Bool}(undef, action_space_size, buffer_size)
        act = Vector{Int}(undef, buffer_size)
        pol = Array{Float32}(undef, action_space_size, buffer_size)
        targ = Vector{Float32}(undef, buffer_size)
        new(buffer_size, 0, 1, obs_vals, masks, act, pol, targ)
    end
end

"""
    append!(buffer, obs_values, action_masks, actions, policies, targets)

Add provided episode data to the replay buffer. If the replay buffer
is full, the oldest elements are deleted in a FIFO manner.

Parameters
- `buffer`: the replay buffer
- `obs_values`: observations before applied actions
- `action_masks`: respective action masks
- `actions`: newly taken actions
- `policies`: corresponding policies
- `targets`: corresponding target values
"""
function append!(buffer::ReplayBuffer, obs_values::Vector{Vector{Float32}},
    action_masks::Vector{Vector{Bool}}, actions::Vector{Int},
    policies::Vector{Vector{Float32}}, targets::Vector{Float32})

    @assert size(obs_values, 1) == size(action_masks, 1) == length(actions) ==
         size(policies, 1) == length(targets)

    if buffer.max_size == 0  # buffer not used, do nothing
        return
    end
    for i in 1:length(actions)
        buffer.obs_values[:, buffer.oldest] = obs_values[i]
        buffer.action_masks[:, buffer.oldest] = action_masks[i]
        buffer.actions[buffer.oldest] = actions[i]
        buffer.policies[:, buffer.oldest] = policies[i]
        buffer.targets[buffer.oldest] = targets[i]

        buffer.oldest += 1
        if buffer.oldest > buffer.max_size
            buffer.oldest = 1
        end
        if buffer.current_size < buffer.max_size
            buffer.current_size += 1
        end
    end
end

"""
    sample(replay_buffer, n)

Return up to `n` randomly sampled data from the replay buffer as tuple
(obs_values, action_masks, actions, policies, targets).
Less than `n` elements are returned when there are not enough elements yet in the buffer.
"""
function sample(buffer::ReplayBuffer, n::Int)
    if buffer.current_size < n
        n = buffer.current_size
    end
    ind = StatsBase.sample(1:buffer.current_size, n, replace=false)

    obs_values = view(buffer.obs_values, :, ind)
    action_masks = view(buffer.action_masks, :, ind)
    actions = buffer.actions[ind]
    policies = view(buffer.policies, :, ind)
    targets = buffer.targets[ind]
    obs_values, action_masks, actions, policies, targets
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
    rewards::Vector{Reward}

    function ReplayBufferAdder(replay_buffer::ReplayBuffer)
        buffer = replay_buffer
        obs_values = Vector{Vector{Float32}}()
        action_masks = Vector{Vector{Bool}}()
        actions = Vector{Int}()
        policies = Vector{Vector{Float32}}()
        rewards = Vector{Reward}()
        new(buffer, obs_values, action_masks, actions, policies, rewards)
    end
end

"""
    add!(replay_buffer_adder, observation, action_mask, action, policy, reward)

Add data of performed action to the adder cache.
"""
function add!(adder::ReplayBufferAdder, observation::Observation, action::Int,
        policy::Vector{Float32}, reward::Reward)
    push!(adder.obs_values, copy(observation.values))
    push!(adder.action_masks, copy(observation.action_mask))
    push!(adder.actions, action)
    push!(adder.policies, copy(policy))
    push!(adder.rewards, reward)
end

"""
    flush!(replay_buffer_adder)

Calculate target values from received rewards and write all to the replay buffer.
"""
function flush!(adder::ReplayBufferAdder)
    targets = Vector{Float32}(adder.rewards)
    targets[end] = adder.rewards[end]
    for i in length(adder.rewards)-1:1
        targets[i] = targets[i+1] + adder.rewards[i]
    end
    append!(adder.buffer, adder.obs_values, adder.action_masks, adder.actions,
        adder.policies, targets)
    empty!(adder.obs_values)
    empty!(adder.action_masks)
    empty!(adder.actions)
    empty!(adder.policies)
    empty!(adder.rewards)
end