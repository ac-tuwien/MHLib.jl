using MHLib.MCTSs

export AlphaZero, PolicyValueNetwork, DummyNetwork

"""
    PolicyValueNetwork

Abstract type for a trainable policy/value function.

The function takes as input an `Observation` and outputs a policy, i.e.,
probability distribution over the possible actions, and a value that
should approximate the total discounted reward received from the current state
onwards.

A concrete class must implement:
- `forward(network, obs_values, action_mask)::Tuple{Int, Vector{Float32}}`
"""
abstract type PolicyValueNetwork end

"""
    forward(network, obs_values, action_mask)

Calculate network in forward direction returning action and policy.
The provided action_mask may or may not be considered
"""
forward(network::PolicyValueNetwork, obs_values::Vector{Float32},
        action_mask::Vector{Bool})::Tuple{Int, Vector{Float32}} =
    error("abstract forward(network, obs_values, action_mask) called")

#------------------------------------------------------------------------------

"""
    DummyNetwork(action_space_size, observation_space_size)

A `PolicyValueNetwork` that just returns a random action and uniform policy.

Used just for testing purposes.
"""
struct DummyNetwork <: PolicyValueNetwork
    action_space_size::Int
end

function forward(network::DummyNetwork, obs_values::Vector{Float32},
    action_mask::Vector{Bool})::Tuple{Int, Vector{Float32}}
    na = network.action_space_size
    return rand(1:na), fill(Float32(1)/na, na)
end


#------------------------------------------------------------------------------

"""
    AZActor(network, replay_buffer, environment)

Actor of AlphaZero agent, based on `MCTS` utilizing an `PolicyValueNetwork` for
leaf node evaluation.
"""
mutable struct AZActor
    environment::Environment
    buffer::ReplayBuffer
    adder::ReplayBufferAdder
    prev_observation::Union{Observation, Nothing}
    network::PolicyValueNetwork
    mcts::MCTS

    function AZActor(network::PolicyValueNetwork, buffer::ReplayBuffer, env::Environment)
        adder = ReplayBufferAdder(buffer)
        mcts = MCTS(env)
        new(env, buffer, adder, nothing, network, mcts)
    end
end

"""
    select_action(az_actor, observation)

Sample policy for given observation by performing MCTS and return action and policy.
"""
function select_action(actor::AZActor, observation::Observation)::
    Tuple{Int, Vector{Float32}}
    # TODO
end

"""
    observe_first!(az_actor, environment)

Make a first observation from the environment.
"""
function observe_first!(actor::AZActor, observation::Observation)
    actor.prev_observation = observation
end

"""
    observe!(actor, action, policy, observation, reward, isfinal)

Observe the performed action and resulting observation, reward and if state is final.
"""
function observe!(actor::AZActor, action::Int, policy::Vector{Float32},
        observation::Observation, reward::Float32, isfinal::Bool)
    add!(actor.adder, prev_observation, action, policy, reward)
    if isfinal
        flush!(actor.adder)
    end
    prev_observation = observation
end

"""
    update!(actor)

Perform an update of the actor parameters from past observations, nothing to do here.
"""
function update!(::AZActor)
end

#------------------------------------------------------------------------------

"""
    AZLearner(network, replay_buffer)

Learner of AlphaZero agent, training the `PolicyValueNetwork` used in the actor's MCTS
with data from the replay buffer
"""
mutable struct AZLearner
    network::PolicyValueNetwork
    buffer::ReplayBuffer

    function AZLearner(network::PolicyValueNetwork, buffer::ReplayBuffer)
        new(network, buffer)
    end
end

"""
    step!(az_learner)

Perform an update step of the `PolicyValueNetwork`.
"""
function step!(learner::AZLearner)
    # TODO
end

#------------------------------------------------------------------------------

"""
    AlphaZero(environment, network)

An RL agent performing similarly as AlphaZero by Silver et al (2018).

Additional keyword arguments in constructor:
- `replay_capacity`
- `min_ovservations_for_learning`
- `observations_per_learning_step`
- `learning_steps_per_update`
"""
mutable struct AlphaZero <: Agent
    actor::AZActor
    learner::AZLearner
    min_observations_for_learning::Int
    observations_per_learning_step::Int
    num_observations::Int
    learning_steps_per_update::Int

    network::PolicyValueNetwork
    replay_buffer::ReplayBuffer

    function AlphaZero(env::Environment, network::PolicyValueNetwork;
            replay_capacity=1000,                   # TODO 5n when calling for LCS
            min_observations_for_learning=100,      # TODO 4n when calling for LCS
            observations_per_learning_step=1,
            learning_steps_per_update=1,)
        replay_buffer = ReplayBuffer(replay_capacity, observation_space_size(env),
            action_space_size(env))
        actor = AZActor(network, replay_buffer, env)
        learner = AZLearner(network, replay_buffer)
        new(actor, learner, min_observations_for_learning,
            observations_per_learning_step, 0, learning_steps_per_update, network,
            replay_buffer)
    end
end
