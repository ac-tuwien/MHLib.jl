"""
    Environments

Abstract types for environments and state graphs on which algorithms/agents may act.

Used for construction or tree search based methods, reinforcement learning etc.
"""
module Environments

export Environment, State, Observation, reset!, action_space_size, observation_space_size,
    get_state, set_state!, step!


"""
    State

Abstract type for a state in the environment on which an optimization/learning agent acts.
"""
abstract type State end

Base.string(state::State) = error("abstract string(state) called")


"""
    Observation

Observation of a state in the environment, from which predictions are made.

Attributes
- values::Vector{Float32}: Observed values
- action_mask::Vector{Bool}: Boolean vector indicating valid actions

TODO GR: Weg hier mit den Priors, das war ein unschöner Hack.
Priors sind etwas sehr MCTS-Spezifisches und LCS sollte davon nichts wissen müssen.
Der Typ könnte anstattdessen um eine Funktion `heuristic` erweitert werden, die optional eine
problemspezifische heuristische Policy zurückliefert,
"""
struct Observation
    values::Vector{Float32}
    action_mask::Vector{Bool}
    priors::Vector{Float32}
end


"""
    Environment

Abstract type for an environment on which an optimization or learning agent may act.

Abstract methods
- `action_space_size(env)::Int`: Size of action space
- `observation_space_size(env)::Int`: Size of observation space
- `reset!(env)::Observation`: Reset environment to initial state and return observation
- `get_state(env)::State`: Return current state
- `set_state!(env, state::State, obs::Observation)`: Set state of environment
- `step!(env, action)`: Perform action in environment and
    return new observation, reward and a Bool indicating end of episode
"""
abstract type Environment end

"""
    action_space_size(env)

Return size of the action space.
"""
action_space_size(env::Environment)::Int =
    error("abstract action_space_size(env) called")

"""
    obs_space_size(env)

Return size of the action space.
"""
observation_space_size(env::Environment)::Int =
    error("abstract observation_space_size(env) called")

"""
    reset!(env)

Reset the environment and return initial observation.
"""
reset!(env::Environment)::Observation = error("abstract reset!(env) called")

"""
    get_state(env)

Return complete current state that can later be set again.
"""
get_state(env::Environment)::State = error("abstract get_state(env) called")

"""
    set_state(env)

Set the state formerly obtained by `get_state(env)`.
"""
set_state!(env::Environment, state::State, obs::Observation) =
    error("abstract set_state!(env, state, obs) called")

"""
    step!(env, action)

Perform action in environment.

Return new observation, reward and a Bool indicating end of episode.
"""
step!(env::Environment, action::Int)::(Observation, Float32, Bool) =
    error("abstract step!(env, action) called")

end  # module
