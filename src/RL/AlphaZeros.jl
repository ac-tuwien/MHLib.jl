using MHLib.MCTSs

export AlphaZero, AZActor


"""
    AZActor(environment, network, replay_buffer)

Actor of AlphaZero agent, based on `MCTS` utilizing an `PolicyValueFunction` for
leaf node evaluation.
"""
mutable struct AZActor <: Actor
    environment::Environment
    network::PolicyValueFunction
    buffer::ReplayBuffer
    adder::ReplayBufferAdder

    prev_observation::Observation
    mcts::MCTS

    function AZActor(env::Environment, network::PolicyValueFunction,
            buffer::ReplayBuffer = ReplayBuffer(0, observation_space_size(env),
                action_space_size(env)))
        adder = ReplayBufferAdder(buffer)
        new(env, network, buffer, adder)  # prev_observation and mcts left uninitialized
    end
end

"""
    select_action(az_actor, observation)

Sample policy for given observation by performing MCTS and return action and policy.
"""
function select_action(actor::AZActor, observation::Observation) ::
        Tuple{Int, Vector{Float32}}
    perform_mcts!(actor.mcts)
end

"""
    observe_first!(az_actor, environment)

Make a first observation from the environment: create new MCTS incl. policy_value_function
"""
function observe_first!(actor::AZActor, observation::Observation)
    actor.prev_observation = observation
    policy_value_function(obs::Observation)::Tuple =
        (actor.network)(obs.values, obs.action_mask)
    actor.mcts = MCTS(actor.environment, observation, policy_value_function)
end

"""
    observe!(actor, action, policy, observation, reward, isfinal)

Observe the performed action and resulting observation, reward and if state is final.
"""
function observe!(actor::AZActor, action::Int, policy::Vector{Float32},
        observation::Observation, reward::Reward, isfinal::Bool)
    add!(actor.adder, actor.prev_observation, action, policy, reward)
    if isfinal
        flush!(actor.adder)
    end
    set_new_root!(actor.mcts, action, observation)
    actor.prev_observation = observation
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

Learner of AlphaZero agent, training the `PolicyValueFunction` used in the actor's MCTS
with data from the replay buffer
"""
mutable struct AZLearner
    network::PolicyValueFunction
    buffer::ReplayBuffer
    batch_size::Int

    function AZLearner(network::PolicyValueFunction, buffer::ReplayBuffer,
        batch_size::Int)
        new(network, buffer, batch_size)
    end
end

"""
    step!(az_learner)

Perform an update step of the `PolicyValueFunction` with data sampled from the buffer.
"""
function step!(learner::AZLearner)
    training_data = sample(learner.buffer, learner.batch_size)
    if size(training_data[1], 2) > 0
        train!(learner.network, training_data...)
    end
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

    network::PolicyValueFunction
    replay_buffer::ReplayBuffer

    function AlphaZero(env::Environment, network::PolicyValueFunction;
            replay_capacity = 1000,
            min_observations_for_learning = 100,
            observations_per_learning_step = 1,
            learning_steps_per_update = 1,
            learning_batch_size = 32)
        replay_buffer = ReplayBuffer(replay_capacity, observation_space_size(env),
            action_space_size(env))
        actor = AZActor(env, network, replay_buffer)
        learner = AZLearner(network, replay_buffer, learning_batch_size)
        new(actor, learner, min_observations_for_learning,
            observations_per_learning_step, 0, learning_steps_per_update,
            network, replay_buffer)
    end
end
