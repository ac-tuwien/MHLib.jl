"""
    ActionValueNetwork

Abstract class for a trainable action-value network.

A concrete class must implement:
- TODO
"""
abstract type ActionValueNetwork end

#------------------------------------------------------------------------------

"""
    AZActor(network, replay_buffer)

Actor of AlphaZero agent, based on `MCTS` utilizing an `ActionValueNetwork` for
leaf node evaluation.
"""
mutable struct AZActor
    buffer::ReplayBuffer
    adder::ReplayBufferAdder
    prev_observation::Union{Observation, Nothing}
    network::ActionValueNetwork

    function AZActor(network::ActionValueNetwork, buffer::ReplayBuffer)
        actor.adder = ReplayBufferAdder(buffer)
        new(buffer, adder, Nothing, network)
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

Learner of AlphaZero agent, training the `ActionValueNetwork` used in the actor's MCTS
with data from the replay buffer
"""
mutable struct AZLearner
    network::ActionValueNetwork
    buffer::ReplayBuffer

    function AZLearner(network::AZLearner, buffer::ReplayBuffer)
        new(network, buffer)
    end
end

"""
    step!(az_learner)

Perform an update step of the `ActionValueNetwork`.
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

    network::ActionValueNetwork
    replay_buffer::ReplayBuffer

    function AlphaZero(env::Environment, network::ActionValueNetwork;
            replay_capacity=1000,                   # TODO 5n when calling for LCS
            min_observations_for_learning=100,      # TODO 4n when calling for LCS
            observations_per_learning_step=1,
            learning_steps_per_update=1,)
        replay_buffer = ReplayBuffer(replay_capacity, obervation_space_size(env),
            action_space_size(env))
        actor = AZActor(network, replay_buffer)
        learner = AZLearner(network, replay_buffer)
        AlphaZero(actor, learner, min_observations_for_learning,
            observations_per_learning_step, 0, learning_steps_per_update, network,
            replay_buffer)
    end
end
