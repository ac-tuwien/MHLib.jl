"""
ReplayBuffer

A FIFO buffer holding at most `max_size` tuples (actions, policies, values, targets)
from which learning is performed.

Attributes
- `max_size`: maximum size of the replay buffer
- `current_size`: current size of the replay buffer
- `oldest`: index of oldest entry
- `obs_values`: original observation; each row represents one observation
- `actions`: actions taken
- `policies`: corresponding policies; each row represents one policy
- `targets`: obsered target values
"""
mutable struct ReplayBuffer
    max_size::Int
    current_size::Int
    oldest::Int

    obs_values::Array{Float32, 2}
    actions::Vector{Int}
    policies::Array{Float32, 2}
    targets::Vector{Float32}
end

"""
    ReplayBuffer(buffer_size, obs_space_size, action_space_size)

Initialize the replay buffer.
"""
function ReplayBuffer(buffer_size, obs_space_size, action_space_size)
    obs_vals = Array{Float32}(undef, buffer_size, obs_space_size)
    act = Vector{Int}(undef, buffer_size)
    pol = Array{Float32}(undef, buffer_size, action_space_size)
    targ = Vector{Float32}(undef, buffer_size)
    ReplayBuffer(buffer_size, 0, 1, obs_vals, act, pol, targ)
end

"""
    append!(buffer, obs_values, actions, policies, targets)

Append provided episode data to the replay buffer. If the replay buffer
is full, the oldest elements are deleted in a FIFO manner.

Parameters
- `buffer`: the replay buffer
- `obs_values`: observation
- `actions`: newly taken actions
- `policies`: corresponding policies
- `targets`: corresponding target values
"""
function append!(buffer::ReplayBuffer, obs_values::Array{<:AbstractFloat, 2},
    actions::Vector{Int}, policies::Array{<:AbstractFloat, 2}, targets::Vector{Int})

    @assert length(newActions) == length(newTargets) == size(newPolicies, 1) ==
        size(newValues, 1)

    for i in length(newactions)
        buffer.obs_values[oldest] = obs_values[i]
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

Return `n randomly sampled data from the replay buffer as tuple
(actions, policies, obs_values, targets)
"""
function sampleReplayBuffer(buffer::ReplayBuffer, n_training::Int)
    # TODO: Auswahl in O(n_training machen, nicht O(buffer.max_size)!
    # -> StatsBase.sample
    ind = randperm(buffer.current_size)[1:n_training]

    # TODO Unnötiges Kopieren vermeiden, einen zB einen View retournieren!
    obs_values = buffer.obj_values[ind, :]
    actions = buffer.actions[ind]
    policies = buffer.policies[ind, :]
    targets = buffer.targets[ind, :]
    return (obs_values, policies, actions, targets)
end
