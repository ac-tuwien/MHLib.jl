"""
    RL

Problem-independent types for reinforcement learning (RL).
"""
module RL

using ArgParse
using Logging
using TensorBoardLogger: TBLogger
using Logging
using Dates
using Printf

using MHLib
using MHLib.Environments
import MHLib.run!

export Actor, Learner, Agent, EnvironmentLoop


const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--rl_lfreq"
        help = "RL: frequency of writing iteration logs"
        arg_type = Int
        default = 1
    "--rl_ldir"
        help = "RL: directory where to write TensorBoard logs"
        arg_type = AbstractString
        default = "tblog"
    "--rl_titer"
        help = "RL: number of main iterations to perform"
        arg_type = Int
        default = 300
end


"""
    Actor

Abstract type for an actor in RL.

A concrete actor must implement:
- `select_action(actor)`
- `observe_first!(actor, observation)`
- `observe!(actor, action, observation, reward, isfinal)`
- `update!(actor)`
"""
abstract type Actor end

"""
    select_action(actor, observation)

Sample policy for given observation and return action and policy.
"""
select_action(::Actor, ::Observation)::Tuple{Int, Vector{Float32}} =
    error("abstract select_action(actor, observation) called")

"""
    observe_first!(actor, observation)

Make a first observation from the environment.
"""
observe_first!(::Actor, ::Observation) =
    error("abstract observe_first!(actor, observation) called")

"""
    observe!(actor, action, policy, observation, reward, isfinal)

Observe the performed action and resulting observation, reward and if state is final.
"""
observe!(::Actor, action::Int, ::Vector{Float32}, ::Observation, ::Float32, ::Bool) =
    error("abstract observe!(actor, action, policy, observation, reward, isfinal) called")

"""
    update!(actor)

Perform an update of the actor parameters from past observations.
"""
update!(::Actor) =
    error("abstract update!(actor) called")


#------------------------------------------------------------------------------

"""
  Learner

Abstract interface for a learner in RL.

Concrete classes must implement:
- `step!(learner)`: perform a learning step
"""
abstract type Learner end

"""
    step!(learner)

Perform an update step of the learner's parameters.
"""
step!(::Learner) =
    error("abstract step!(learner) called")


#------------------------------------------------------------------------------

"""
  Agent

Agent structure which combines acting and learning.

This provides an implementation of acting and learning.
It takes as input instances of both `Actor` and `Learner`  and implements the policy,
observation, and update methods which defer to the underlying actor and learner.
The only real logic implemented is that it controls the number
of observations to make before running a certain number of learner step.

Concrete subtypes must implement:
- `actor::Actor`
- `learner::Learner`
- `min_observations_for_learning::Int`: minimum number of observations for learning
- `observations_per_learning_step::Int`: number of observations between learning steps
- `num_observations::Int`: number of performed observations
- `learning_steps_per_update::Int`: number of learning steps per update call of learner
"""
abstract type Agent <: Actor end

select_action(agent::Agent, obs::Observation)::Tuple{Int, Vector{Float32}} =
    select_action(agent.actor, obs)

observe_first!(agent::Agent, observation::Observation) =
    observe_first!(agent.actor, observation)

function observe!(agent::Agent, action::Int, policy::Vector{Float32}, obs::Observation,
        reward::Float32, isfinal::Bool)
    println("-- RL.observe!(agent, action, policy, obs, reward, isfinal) called")
    println("   Typ von agent.actor: " * string(typeof(agent.actor)))
    observe!(agent.actor, action, policy, obs, reward, isfinal)
    agent.num_observations += 1
end

"""
    update!(agent::Agent)

Updates the learner and the agent if there are enough observations
"""
function update!(agent::Agent)
    println("-- RL.update!(agent) called")
    println("   agent.num_observations: " * string(agent.num_observations))
    println("   agent.min_observations: " * string(agent.min_observations_for_learning))
    println("   current_buffer_size:    " * string(agent.learner.buffer.current_size))
    # TODO Daniel learning only after full episodes.
    # Hier wird nach jedem MCTS-step gelernt! Wie abschalten, sodass
    # nur nach einer Episode gelernt wird?
    n_obs = agent.num_observations - agent.min_observations_for_learning
    if n_obs >= 0 && n_obs % agent.observations_per_learning_step == 0
        for i in 1:agent.learning_steps_per_update
            step!(agent.learner)
        end
        update!(agent.actor)
    end
end


#------------------------------------------------------------------------------

"""
    EnvironmentLoop

A simple RL environment loop.

This takes `Environment` and `Actor` instances and coordinates their
interaction. Agent is updated. This can be used as:
loop = EnvironmentLoop(environment, actor)
loop.run(num_episodes)
A `Counter` instance can optionally be given in order to maintain counts
between different Acme components. If not given a local Counter will be
created to maintain counts between calls to the `run` method.
"""
mutable struct EnvironmentLoop
    environment::Environment
    actor::Actor
    tblogger::AbstractLogger

    function EnvironmentLoop(env::Environment, actor::Actor)
        subdir = replace("RL-" * string(now()) * tempname(".")[3:end], ":"=>"-")
        logdir = joinpath(settings[:rl_ldir], subdir)
        new(env, actor, TBLogger(logdir))
    end
end

"""
    run_episode!(environment_loop)

Perform a whole episode and return a tuple with statistical results.
"""
function run_episode!(el::EnvironmentLoop)
    println("-- RL.run_episode!(el) called")
    start_time = time()
    episode_steps = 0
    episode_reward = 0
    isfinal = false

    observation = reset!(el.environment)
    observe_first!(el.actor, observation)

    iter = 0

    # perform a whole episode
    while !isfinal
        iter += 1
        println("Episodeniteration " * string(iter))

        # generate an action from the agent's policy and step the environment
        action, policy = select_action(el.actor, observation)
        observation, reward, isfinal = Environments.step!(el.environment, action)
        # println("action: ", action, "->", el.environment.state.p)

        # have the agent observe the timestep and let the actor update itself
        observe!(el.actor, action, policy, observation, reward, isfinal)
        update!(el.actor)

        episode_reward += reward
        episode_steps += 1
    end
    steps_per_s = episode_steps / (time() - start_time)
    (episode_length=episode_steps, reward=episode_reward, steps_per_s=steps_per_s)
end

"""
    run!(environment_loop, num_episodes)

Perform environment loop for the given number of episodes.
"""
function run!(el::EnvironmentLoop, num_episodes::Int)
    episode_count = 0
    println("episode\tlength\treward\tsteps_s")
    while episode_count < num_episodes
        episode_count += 1
        println("\nEPISODE NUMMER " * string(episode_count))
        results = run_episode!(el)

        # TODO Daniel Hier kommt step!() rein?!

        if (episode_count-1) % settings[:rl_lfreq] == 0
            with_logger(el.tblogger) do
                @info("MHlib.RL",
                    episode_length = results.episode_length,
                    reward = results.reward,
                    steps_per_s = results.steps_per_s)
            end
            @printf("%5d\t%6d\t%6.4f\t%6.4f\n",
                episode_count,
                results.episode_length,
                results.reward,
                results.steps_per_s)
        end
    end
    return true
end


include("ReplayBuffers.jl")
include("AlphaZeros.jl")


end  # module
