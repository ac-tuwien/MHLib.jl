"""
    ActionValueNetwork

Abstract class for a trainable action-value network.

A concrete class must implement:
- TODO
"""
abstract type ActionValueNetwork end


"""
    AZActor

Actor of AlphaZero agent, based on `MCTS` utilizing an `ActionValueNetwork` for
leaf node evaluation.
"""
mutable struct AZActor
end

"""
    AZLearner

Learner of AlphaZero agent, training the `ActionValueNetwork` used in the actor's MCTS
with data from the replay buffer
"""
mutable struct AZLearner
end

"""
    AlphaZero

An RL agent performing similarly as AlphaZero by Silver et al (2018).
"""
mutable struct AlphaZero <: Agent
    actor::AZActor
    learner::AZLearner
    min_observations::Int
    observations_per_step::Int
    num_observations::Int
    learning_steps_per_update::Int
    network::ActionValueNetwork
    replay_buffer::ReplayBuffer
end

function AlphaZero(env::Environment, network::ActionValueNetwork)
    # TODO Tune constants and/or use parameters for more essential values
    replay_buffer = ReplayBuffer(1000, obervation_space_size(env), action_space_size(env))
    actor = AZActor(network, replay_buffer)
    learner = AZLearner(network, replay_buffer)
    AlphaZero(actor, learner, 10, 100, 0, 1, network, replay_buffer)
end
