using MHLib.MCTSs

export AlphaZero, AZActor, PolicyValueNetwork, DummyNetwork

"""
    PolicyValueNetwork

Abstract type for a trainable policy/value function.

The function takes as input an `Observation` and outputs a policy, i.e.,
probability distribution over the possible actions, and a value that
should approximate the total discounted reward received from the current state
onwards.

A concrete class must implement:
- `forward(network, obs_values, action_mask)::Tuple{Vector{Float32}, Float32}`
- `learn!(...TODO)`
"""
abstract type PolicyValueNetwork end

"""
    forward(network, obs_values, action_mask)

Calculate network in forward direction returning policy and value.
The provided action_mask may or may not be considered
"""
forward(network::PolicyValueNetwork, obs_values::Vector{Float32},
        action_mask::Vector{Bool})::Tuple{Float32, Vector{Float32}} =
    error("abstract forward(network, obs_values, action_mask) called")

#------------------------------------------------------------------------------

"""
    DummyNetwork(action_space_size)

A `PolicyValueNetwork` that just returns a uniform policy and random value.

Used just for testing purposes.
"""
struct DummyNetwork <: PolicyValueNetwork
    action_space_size::Int
end

function forward(network::DummyNetwork, obs_values::Vector{Float32},
    action_mask::Vector{Bool})::Tuple{Vector{Float32}, Float32}
    na = network.action_space_size
    return fill(Float32(1)/na, na), Float32(rand(1:na))
end


#------------------------------------------------------------------------------

"""
    AZActor(environment, network, replay_buffer)

Actor of AlphaZero agent, based on `MCTS` utilizing an `PolicyValueNetwork` for
leaf node evaluation.
"""
mutable struct AZActor <: Actor
    environment::Environment
    network::PolicyValueNetwork
    buffer::ReplayBuffer
    adder::ReplayBufferAdder

    prev_observation::Observation
    mcts::MCTS

    function AZActor(env::Environment, network::PolicyValueNetwork,
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
    policy_value_function(env::Environment, obs::Observation)::Tuple =
        forward(actor.network, obs.values, obs.action_mask)
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
    obervations_per_training::Int
    min_observations_for_training::Int
    batch_size::Int

    function AZLearner(network::PolicyValueNetwork, buffer::ReplayBuffer,
        obervations_per_training::Int, min_observations_for_training::Int,
        batch_size::Int)
        new(network, buffer, obervations_per_training, min_observations_for_training, batch_size)
    end
end

"""
    step!(az_learner)

Perform an update step of the `PolicyValueNetwork`.
"""
function step!(learner::AZLearner)
    train_data = sample(buffer, batch_size)
    # TODO train!(network, train_data(relevanter Teil))
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
    # TODO Daniel: Nach AZLearner verschoben, da dort mehr Sinn
    #min_observations_for_learning::Int
    #observations_per_learning_step::Int
    # TODO Daniel: Wofür stehen diese beiden Parameter?
    num_observations::Int
    learning_steps_per_update::Int

    network::PolicyValueNetwork
    replay_buffer::ReplayBuffer

    function AlphaZero(env::Environment, network::PolicyValueNetwork;
            replay_capacity=1000,
            min_observations_for_learning=100,
            observations_per_learning_step=1,
            learning_steps_per_update=1,)
        replay_buffer = ReplayBuffer(replay_capacity, observation_space_size(env),
            action_space_size(env))
        actor = AZActor(env, network, replay_buffer)
        learner = AZLearner(network, replay_buffer)
        new(actor, learner, min_observations_for_learning,
            observations_per_learning_step, 0, learning_steps_per_update, network,
            replay_buffer)
    end
end
