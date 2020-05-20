"""
    Environments

Abstract classes for environments and state graphs on which agents may act.

Used for tree search methods, reinforcement learning etc.
"""
module Environments

export Environment, State, Observation


"""
    State

Abstract type for a state in the environment on which an optimization/learning agent acts.
"""
abstract type State end


"""
    Observation

Observation of a state in the environment, from which predictions are made.

Attributes
- values::Vector{Float32}: Observed values
- valid_actions::Vector{Bool}: Boolean vector indicating valid actions
"""
struct Observation
    values::Vector{Float32}
    valid_actions::Vector{Bool}
end


"""
    Environment

Abstract type for an environment on which an optimization or learning agent may act.

Abstract methods
- `action_space_size(env)::Int`: Size of action space
- `reset(env)::Observation`: Reset environment to initial state and return observation
- `get_state(env)::State`: Return current state
- `set_state!(env, state::State)`: Set state
- `step!(env, action)`: Perform action in environment and
    return new observation, reward and a Bool indicating end of episode
"""
abstract type Environment end

action_space_size(env::Environment)::Int =
    error("abstract action_space_size(env) called")

reset!(env::Environment)::Observation = error("abstract reset!(env) called")

get_state(env::Environment)::State = error("abstract get_state(env) called")

set_state!(env::Environment, state::State)::Obseration =
    error("abstract set_state!(env) called")

step!(env::Environment, action::Int)::(Observation, Float32, Bool) =
    error("abstract step!(env, action) called")

end  # module
