"""
    DeepL

Deep Learning module for MHLib

"""
module DeepLearning

using Random
using MHLib
using MHLib.LCS
using MHLib.MCTSs
using Flux
using Logging
using TensorBoardLogger

export iterate_deepl

import MHLib.Environments:
    Environment,
    Observation,
    State,
    get_state,
    set_state!,
    action_space_size,
    step!,
    reset!


"""
Replay Buffer

The Replay Buffer stores at most `max_size` data points (actions, policies, values, targets).

Attributes
- `max_size`: maximum size of the replay buffer
- `current_size`: current size of the replay buffer
- `actions`: actions taken
- `policies`: corresponding policies suggested by MCTS. Each row represents one policy.
- `values`: corresponding original observation (without sorting).
  Each row represents one observation
- `targets`: string length of the episode
"""
mutable struct ReplayBuffer
    max_size::Int
    current_size::Int

    actions::Array{Int, 1}
    policies::Array{Float32, 2}
    values::Array{Float32, 2}
    targets::Array{Int, 1}
end



"""
    ReplayBuffer(n_buffer, sigma, n_values)

Initializes the Replay Buffer accordingly.

Parameters
- `n_buffer`: max size of the replay buffer
- `sigma`: alphabet size
- `n_values`: observation size (= input length of action network)
"""
function ReplayBuffer(n_buffer::Int, sigma::Int16, n_values::Int)
    act = Array{Int}(undef, 0)  # Vector{Int} does not work (type DataType)
    pol = Array{Float32}(undef, 0, sigma)
    vals = Array{Float32}(undef, n_values, 0)
    targ = Array{Int}(undef, 0)
    ReplayBuffer(n_buffer, 0, act, pol, vals, targ)
end



"""
Deep Learning incredients consisting of the two networks and the replay buffer.

Attributes
- `value_nn`: network for the values
- `action_nn`: network for the actions
- `replay_buffer`: the replay buffer for sampling data points for training
- `n_training`: number of samples taken out of the replay buffer for training
- `n_min_buffer`: minimum size of the replay buffer needed to train
- `opt_value`: ADAM-optimizer for value network
- `opt_action`: ADAM-optimizer for action network
"""
mutable struct DeepL
    value_nn::Chain
    action_nn::Chain
    replay_buffer::ReplayBuffer
    n_training::Int
    n_min_buffer::Int

    opt_value::ADAM
    opt_action::ADAM
end



"""
    DeepL(n_inp_value, n_inp_action, n_buffer, sigma, n_training, n_min_buffer)

Constructor for the Deep Learning object. The networks as well as the
replay buffer are initialized.

Parameters
- `n_inp_value`: number of inputs in value network
- `n_inp_action`: number of inputs in value network
- `n_buffer`: maximum buffer size
- `sigma`: alphabet size
- `n_training`: number of samples taken out of the replay buffer for training
- `n_min_buffer`: minimum size of the replay buffer needed to train
"""
function DeepL(n_inp_value::Int, n_inp_action::Int, n_buffer::Int, sigma, n_training::Int,
    n_min_buffer::Int)
    value_nn = Chain(
        Dense(n_inp_value, 50, relu),
        Dense(50, 50, relu),
        Dense(50, 1, relu))
    # TODO DANIEL per_action_nn: how to incorporate action?
    action_nn = Chain(
        Dense(n_inp_action, 50, relu),
        Dense(50, 50, relu),
        Dense(50, sigma, relu))
    replayBuffer = ReplayBuffer(n_buffer, sigma, n_inp_action)

    opt_value = Flux.Optimise.ADAM(0.001, (0.9, 0.999))
    opt_action = Flux.Optimise.ADAM(0.001, (0.9, 0.999))

    DeepL(value_nn, action_nn, replayBuffer, n_training, n_min_buffer,
        opt_value, opt_action)
end



"""
    DeepL(env, n_buffer, n_training)

Constructor for the Deep Learning object. The networks as well as the
replay buffer are initialized. All the necessary information for initializing
the DeepL object are derived from the environment (exception: n_buffer).

Parameters
- `env`: the environment of the problem
- `n_buffer`: maximum buffer size
- `n_training`: number of samples taken out of the replay buffer for training
- `n_min_buffer`: minimum size of the replay buffer needed to train
"""
function DeepL(env::Environment, n_buffer::Int, n_training::Int, n_min_buffer::Int)
    # value network has only state information as input:
    # 1.) Remaining string lengths (m)
    # 2.) Minimum letter appearances (sigma)
    #TODO n_inp_value = state_space_size(env)
    n_inp_value = env.inst.m + env.inst.sigma

    # action network has observation
    n_inp_action = observation_space_size(env)

    DeepL(n_inp_value, n_inp_action, n_buffer, env.inst.sigma, n_training, n_min_buffer)
end



"""
    updateReplayBuffer(buffer, newActions, newPolicies, newValues, newTargets)

Appends the newly generated data to the replay buffer. If the replay buffer
is overfull, older data are deleted.

Parameters
- `buffer`: the replay buffer
- `newActions`: newly taken actions
- `newPolicies`: corresponding policies suggested by MCTS
- `newValues`: corresponding original observation (unsorted)
- `newTargets`: corresponding target values
  (i.e. string length of the final solution found by MCTS)
"""
function updateReplayBuffer!(buffer::ReplayBuffer, newActions::Array{Int, 1},
    newPolicies::Array{<:AbstractFloat, 2}, newValues::Array{<:AbstractFloat, 2}, newTargets::Array{Int, 1})

    @assert length(newActions) == length(newTargets) == size(newPolicies, 1) ==
        size(newValues, 2)

    # Step 1: Append new information
    append!(buffer.actions, newActions)
    append!(buffer.targets, newTargets)
    buffer.policies = vcat(buffer.policies, newPolicies) # append new policies as rows
    buffer.values = hcat(buffer.values, newValues)       # append new values as columns

    # Step 2: Check, if buffer is full
    buffer.current_size += length(newActions)
    if buffer.current_size > buffer.max_size
        diff = buffer.current_size - buffer.max_size

        # Delete diff elements
        buffer.actions = buffer.actions[(diff + 1):length(buffer.actions)]
        buffer.targets = buffer.targets[(diff + 1):length(buffer.targets)]
        buffer.policies = buffer.policies[(diff + 1):size(buffer.policies, 1), :]
        buffer.values = buffer.values[:, (diff + 1):size(buffer.values, 2)]

        buffer.current_size = buffer.max_size
    end
end



"""
    logistic!(logitarr)

Applies the logistic function to all elements of the given logit array.
Needed for transforming logits to [0, 1].

Parameters
- `logitarr`: The array
"""
function logistic!(logitarr::Array{T}) where T <: Real
    for i in 1:length(logitarr)
        logitarr[i] = 1 / (1 + exp(- logitarr[i]))
    end
end



"""
    get_prior_function(deepl)

Returns a function, that can compute priors out of the environment.
The DeepL object is implicitely used.
"""
function get_prior_function(deepl::DeepL)
    function tempfun(env::Environment, action_values::Vector{Float32})
        sigma = env.inst.sigma
        logits = Array{Real}(undef, sigma)

        # values are sorted according to sortperm
        # i.e. actions must be used in correct order.
        # every time when step! is applied (whenever a new letter is appended),
        # the observation is recomputed (get_observation() in LCS).

        temp = deepl.action_nn(action_values)
        logits[env.action_order] = temp

        # Determine probability weights
        logistic!(logits)
        return(logits / sum(logits))
    end
    return tempfun
end



"""
    actor!(deepl, env)

Applies MCTS for one a given instance. Each step of the actor is stored and
the replay buffer of deepl is updated.

Parameters
- `deepl`: The deep learning component
- `env`: environment of the LCS
"""
function actor!(deepl::DeepL, env::LCSEnvironment)
    mcts = MCTS{LCSEnvironment}(env)  # TODO Daniel evn, netzwerk

    # Buffer information
    actions = Int[]
    policies = Array{Float32}(undef, 0, env.inst.sigma)
    values = Array{Float32}(undef, observation_space_size(env), 0)

    trace_rollout = false
    trace = false
    trace_actions = false#true

    while (!mcts.root.done)
        # Attention: -mh_mcts_child_criterion should be set to exp_visit_count
        action = perform_mcts!(mcts; trace = trace_rollout)

        # TODO Daniel verwende policy von MCTS (als tupel zurückgegeben)
        policy = softmax(log.(mcts.root.child_N)) # log, since we want to normalize vector

        # m x Remaining String Lengths
        # sigma x Minimum Letter Appearances
        # m x sigma x Jumping Positions
        # Implicitely the priors are computed in get_observation
        obs = LCS.get_observation(env).values

        append!(actions, action)
#println("\n\npolicies + policy")
#println(size(policies))
#println(size(policy))
#println(size(reshape(policy, (1, size(policies)[2]))))
        policies = vcat(policies, reshape(policy, (1, size(policies)[2])))
#println(policies)
#println("\n\n")
        values = hcat(values, reshape(obs, (size(values)[1], 1)))

        if trace
            println(string(mcts.root))
        end

        mcts.root = get_child(mcts.env, mcts.root, actions[length(actions)])

        if trace_actions
            println("Iteration ", length(actions))
            println("Actions taken so far: ", string(actions))
        end
    end

    targets = fill(length(actions), (length(actions)))

    updateReplayBuffer!(deepl.replay_buffer, actions, policies, values, targets)

    return actions
end



"""
    sampleReplayBuffer(buffer, n_training)

Returns n_training training data as tuple (tactions, tpolicies, tvalues, ttargets)

Parameters
- `buffer`: the replay buffer
- `n_training`: number of training data to be sampled
"""
function sampleReplayBuffer(buffer::ReplayBuffer, n_training::Int)
    ind = randperm(buffer.current_size)[1:n_training]

    tactions = buffer.actions[ind]
    tpolicies = buffer.policies[ind, :]
    tvalues = buffer.values[:, ind]
    ttargets = buffer.targets[ind, :]

    return (tactions, tpolicies, tvalues, ttargets)
end



"""
Logging struct for displaying the learning success.

Attributes
- `number_optimal`: Number of times, given instance(s) was/were solved to optimality (with network-priors)
- `number_optimal_mcts`: Number of times, given instance(s) was/were solved to optimality (without networks)
- `value_loss`: The current value loss
- `policy_loss`: The current policy loss
"""
struct LearnLogger
    number_optimal::AbstractFloat
    number_optimal_mcts::AbstractFloat
    value_loss::AbstractFloat
    policy_loss::AbstractFloat
end



"""
    learning!(deepl, buffer, env)

Returns an object of the type LearnLogger after training

Parameters
- `deepl`: the deep learning object
- `buffer`: the replay buffer
- `env`: the environment
"""
function learning!(deepl::DeepL, buffer::ReplayBuffer, env::LCSEnvironment) :: LearnLogger

    if buffer.current_size < deepl.n_min_buffer
        error("Not enough data for training!")
    end

    # Step 1: Sample training data
    tactions, tpolicies, tvalues, ttargets = sampleReplayBuffer(buffer, deepl.n_training)

    # Take the relevant values for value network (each column is one data point)
    X_value = tvalues[1:(env.inst.m + env.inst.sigma), :]
    X_policy = tvalues

    y_value = ttargets
    y_policy = transpose(tpolicies)  # each column = training datum

    y_value = reshape(y_value, size(y_value)[1])

#println("\n\nX_value + y_value")
#println(X_value)
#println(y_value)
#println(size(X_value))
#println(size(y_value))
#println("\n\n")


    train_loader_value = Flux.Data.DataLoader((X_value, y_value),
        batchsize = deepl.n_training)
    train_loader_policy = Flux.Data.DataLoader((X_policy, y_policy),
        batchsize = deepl.n_training)

    # Step 2: Define Model: done during initialization

    # Step 3: Define Loss functions
#println("\n\nX_value value_nn(X_value) y_value")
#println(X_value)
#println(deepl.value_nn(X_value))
#println(y_value)
#println("\n\n")
    #value_loss = Flux.Losses.mse(y_value, deepl.value_nn(X_value))
    value_loss(X, y) = Flux.Losses.mse(deepl.value_nn(X), y)

#println("\n\nX_policy action_nn(X_policy) y_policy")
#println(X_policy)
#println(transpose(deepl.action_nn(X_policy)))
#println(y_policy)
#println("\n\n")
    #policy_loss = Flux.Losses.mse(y_policy, transpose(deepl.action_nn(X_policy)))
    policy_loss(X, y) = Flux.Losses.mse(deepl.action_nn(X), y)
    #policy_loss(X, y) = Flux.Losses.mse(y, deepl.action_nn(X))

    # TODO DANIEL Policy loss function definieren
    # TODO DANIEL Wie genau policy network trainieren? softmax_cross_entropy_with_logits
    # tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=pi_t)

    # Step 4: Adam with default values: done during initialization

    println("\nLoss before training")
    @show value_loss(X_value, y_value)
    @show policy_loss(X_policy, y_policy)

    # Step 5: Training
    Flux.train!(value_loss, params(deepl.value_nn), train_loader_value,
        deepl.opt_value)
    Flux.train!(policy_loss, params(deepl.action_nn), train_loader_policy,
        deepl.opt_action)
    #Flux.train!(policy_loss, params(deepl.action_nn), (X_policy, y_policy),
    #    deepl.opt_action)

    println("\nLoss after training")
    value_loss_value = value_loss(X_value, y_value)
    policy_loss_value = policy_loss(X_policy, y_policy)

    @show value_loss_value
    @show policy_loss_value

    # TODO Daniel Mach es generischer (mehrere Testfiles möglich)
    inst = LCSInstance("data/test-04_003_050.lcs")
    solarray = fill(0, 10)
    println("\nWERTE FÜR NETZWERK")
    for i in 1:length(solarray)
        env = LCSEnvironment(inst)
        # TODO Daniel set.prior.function aufrufen
        env.prior_function = get_prior_function(deepl)
        mcts = MCTS{LCSEnvironment}(env)
        actions_temp = Int[]
        while (!mcts.root.done)
            action = perform_mcts!(mcts; trace = false)
            append!(actions_temp, action)
            mcts.root = get_child(mcts.env, mcts.root, actions_temp[length(actions_temp)])
        end
        println(length(actions_temp))
        solarray[i] = length(actions_temp)
    end

    solarray_mcts = fill(0, 10)
    println("\nWERTE OHNE NETZWERK")
    temp = settings[:lcs_prior_heuristic]
    settings[:lcs_prior_heuristic] = "UB1"
    for i in 1:length(solarray)
        env = LCSEnvironment(inst)
        mcts = MCTS{LCSEnvironment}(env)
        actions_temp = Int[]
        while (!mcts.root.done)
            action = perform_mcts!(mcts; trace = false)
            append!(actions_temp, action)
            mcts.root = get_child(mcts.env, mcts.root, actions_temp[length(actions_temp)])
        end
        println(length(actions_temp))
        solarray_mcts[i] = length(actions_temp)
    end
    settings[:lcs_prior_heuristic] = temp

    number_opt = sum(solarray) / length(solarray)
    number_opt_mcts = sum(solarray_mcts) / length(solarray_mcts)

    res = LearnLogger(number_opt, number_opt_mcts, value_loss_value, policy_loss_value)
    return res
end



"""
    iterate_deepl(m, n, sigma, n_buffer, n_min_buffer, n_training, n_episodes)

Main function for deep learning. Applies n_episodes episodes for training.
# 2.) Führe Actor aus
# 3.) Lerne nach jedem Actor
# 4.) neue Instanz generieren
# Actor: Führe MCTS an Wurzel aus, speichere jede Aktion in Buffer
# Bei Rollouts werden priors des Netzwerks genommen
# FRAGE: Wie werden priors in MCTS eingespeist?
# Nach xy Schritten: Lerne Netzwerk

Parameters
- `m`: number of sequences
- `n`: length of input strings
- `sigma`: number of letters
- `n_buffer`: size of the replay buffer
- `n_min_buffer`: minimum size of the replay buffer needed for training
- `n_training`: batch size for training
- `n_episodes`: number of episodes for training
"""
function iterate_deepl(m::Int, n::Int, sigma::Int, n_buffer::Int,
    n_min_buffer::Int, n_training::Int, n_episodes::Int)

    bool_always_new_seqs = settings[:lcs_always_new_seqs]

    println("Start of iterate_deepl()\n")

    inst = LCSInstance(m, n, sigma)
    env = LCSEnvironment(inst)
    saveInstance(env.inst, "./data/temp.lcs")
    println("MARKOS HEURISTIK:")
    println(getHeuristicValue("./data/temp.lcs"))

    deepl = DeepL(env, n_buffer, n_training, n_min_buffer)

    # setting prior_function in environments (error by default if --lcs_prior_heuristic is RL)
    env.prior_function = get_prior_function(deepl)
    # TODO Daniel: rufe set_prior_function!() auf!

    # Logging
    tensor_board_logger = TBLogger(pwd() * "/TensorBoardLogger/Logs", min_level = Logging.Info)

    for i in 1:n_episodes
        println("\nEpisode nr " * string(i) * "\n")
        actor!(deepl, env)

        if deepl.replay_buffer.current_size >= deepl.n_min_buffer
            logging_info = learning!(deepl, deepl.replay_buffer, env)
            with_logger(tensor_board_logger) do
                @info "Training" logging=logging_info log_step_increment=1
            end
        end

        # reset the environment (incl. new instance)
        settings[:lcs_always_new_seqs] = true
        reset!(env)
        settings[:lcs_always_new_seqs] = bool_always_new_seqs
        saveInstance(env.inst, pwd() * "/data/temp.lcs")
        println("MARKOS HEURISTIK:")
        println(getHeuristicValue("./data/temp.lcs"))
    end
end


end  # module



# TODO DANIEL Ein Notebook schreiben, das die Entwicklung der Lösungsgüte für ausgewählte
# Instanzen im Laufe des Trainings widergibt.


# TODO DANIEL offene Punkte:
# 1.) Wie priors an MCTS übergeben?
#     Antwort: Die priors werden jetzt in LCS.jl via Funktion ausgewertet.
# 2.) Policy network training durchführen
#     Jeder Trainingspunkt des Samples und jede Politik wird betrachtet
#     Für jeden Trainingspunkt: Sortiere jeden Buchstaben, String.
#     Action ist dann die entsprechende Zahl, target ist die entsprechende softmax-
#     Policy von MCTS, tatsächlicher Outcome wird aus logits berechnet.
# 3.) Abstrahieren gleich (!)
# 4.) Profiler einsetzen



# iterate_deepl(m, n, sigma, n_buffer, n_min_buffer, n_training, n_episodes)
