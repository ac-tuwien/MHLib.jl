#!/usr/bin/env julia
"""
    lcs

Demo program for solving the Longest Common Subsequence (LCS) problem.
"""

using ArgParse
using Logging
using MHLib
using MHLib.LCS
using MHLib.MCTSs
using MHLib.RL
using MHLib.Environments

if endswith(pwd(), "test")
    cd("..")
end

println("LCS Demo version $(git_version())\nARGS: ", ARGS)
settings_new_default_value!(MHLib.settings_cfg, "ifile", "data/test-04_003_050.lcs")
parse_settings!([MHLib.MCTSs.settings_cfg, MHLib.LCS.settings_cfg])
println(get_settings_as_string())


"""
    mcts_demo()

Test function that runs MCTS on a small LCS instance.
"""
function mcts_demo()
    # Ignore actual arguments here, using explicitly specified ones
    parse_settings!([MHLib.MCTSs.settings_cfg, MHLib.LCS.settings_cfg],
        ["--seed=0",
        #"--ifile=data/rat-04_010_600.lcs",
        "--ifile=data/test-04_003_050.lcs",
        "--mh_mcts_num_sims=1000",
        "--lcs_reward_mode=smallsteps",
        "--mh_mcts_c_uct=0.5",
        "--mh_mcts_tree_policy=PUCT",
        "--lcs_prior_heuristic=UB1",
        "--mh_mcts_rollout_policy=epsilon-greedy",
        "--mh_mcts_epsilon_greedy_epsilon=0.2"])
    # inst = LCSInstance(3, 8, 4)  # Mit UCB, c_uct = 1, seed = 160569761 kommt [3] heraus (!)
    inst = LCSInstance(settings[:ifile])
    println(inst)
    env = LCSEnvironment(inst)
    obs = reset!(env)
    mcts = MCTS(env, obs)
    println("Seed: ", settings[:seed])
    println("Number of iterations: ", mcts.num_sims, ", c_uct: ", mcts.c_uct)

    # trace ... Sollen die Root-Nodes gedruckt werden?
    # trace_rollout ... Sollen die Rollouts gedruckt werden?
    # trace_actions ... Sollen die Aktionen gedruckt werden?
    actions = iterate_mcts!(mcts, trace = false, trace_rollout = false, trace_actions = true)
    println("Solution: ", length(actions), ' ', actions)
    println("Overall best solution encountered: ", length(mcts.best_solution), ' ',
        mcts.best_solution)
end

# TODO GR: Tracing bitte entfernen wenn diese Funktionen soweit funktionieren

"""
    iterate_mcts!(env)

Iteratively perform MCTS, taking in each iteration the action with the most visits.
TODO: trace = true druckt den kompletten Node-Output jeder Iteration aus, fürs debuggen.
"""
function iterate_mcts!(mcts::MCTS; trace::Bool = false, trace_actions::Bool = false,
    trace_rollout::Bool = false)

    actions = Int[]
    # println(string(root))

    while (!mcts.root.done)
        policy, action = perform_mcts!(mcts; trace = trace_rollout)
        append!(actions, action)
        if trace
            println(string(mcts.root))
        end
        mcts.root = mcts.root.children[action]
        if trace_actions
            println("\nIteration ", length(actions))
            println("Actions taken so far: ", string(actions))
        end
    end
    return actions
end



"""
    mcts_demo_args()

Test function that runs MCTS on a small LCS instance.
"""
function mcts_demo_args()
    # Atom hat als default working directory das vom Package,
    # das sollten wir hier auch annehmen und beibehalten.
    # Ausnahme sind die Tests im Directory test,
    # die dieses Verzeichnis als working directory haben.
    println("Working directory:" * pwd())
    # cd("MHLib.jl")
    # println("Working directory:" * pwd())

    println("Instance: ", settings[:ifile])
    inst = LCSInstance("data/test-04_003_050.lcs")

    println(inst)
    env = LCSEnvironment(inst)
    obs = reset!(env)
    mcts = MCTS(env, obs)
    println("Seed: ", settings[:seed])
    println("Number of simulations: ", mcts.num_sims, ", c_uct: ", mcts.c_uct)

    # trace ... Sollen die Root-Nodes gedruckt werden?
    # trace_rollout ... Sollen die Rollouts gedruckt werden?
    # trace_actions ... Sollen die Aktionen gedruckt werden?
    actions = iterate_mcts!(mcts, trace = false, trace_rollout = false, trace_actions = true)
    println("Solution: ", length(actions), ' ', actions)
    println("Overall best solution encountered: ", length(mcts.best_solution), ' ',
        mcts.best_solution)
end


function deepl_demo()
    # Ignore actual arguments here, using explicitly specified ones
    parse_settings!([MHLib.MCTSs.settings_cfg, MHLib.LCS.settings_cfg],
        ["--seed=0",
        #"--ifile=data/rat-04_010_600.lcs",
        "--ifile=data/test-04_003_050.lcs",
        "--mh_mcts_num_sims=1000",
        "--lcs_reward_mode=smallsteps",
        "--mh_mcts_c_uct=0.5",
        "--mh_mcts_tree_policy=PUCT",
        "--lcs_prior_heuristic=RL",
        "--mh_mcts_rollout_policy=epsilon-greedy",
        "--mh_mcts_epsilon_greedy_epsilon=0.2"])
    # inst = LCSInstance(3, 8, 4)  # Mit UCB, c_uct = 1, seed = 160569761 kommt [3] heraus (!)
    m = 3
    n = 50
    sigma = 4
    n_buffer = 5 * n
    n_min_buffer = 4 * n
    n_training = 32
    n_episodes = 20
    # iterate_deepl(m, n, sigma, n_buffer, n_min_buffer, n_training, n_episodes)
end


function lcs_alphazero()
    # Ignore actual arguments here, using explicitly specified ones
    parse_settings!([MHLib.RL.settings_cfg, MHLib.MCTSs.settings_cfg,
        MHLib.LCS.settings_cfg],
        ["--seed=7",
        #"--ifile=data/rat-04_010_600.lcs",
        "--ifile=data/test-04_003_050.lcs",
        "--mh_mcts_num_sims=100",
        "--lcs_reward_mode=smallsteps",
        "--mh_mcts_c_uct=1.0",
        "--mh_mcts_tree_policy=PUCT",
        "--lcs_prior_heuristic=UB1",])

    inst = LCSInstance(settings[:ifile])
    n = inst.n
    println(inst)
    env = LCSEnvironment(inst)
    obs = reset!(env)

    network = DummyNetwork(action_space_size(env))
    # network = LCSNetwork(env)

    # create AlphaZero agent with configuration as in Python implementation
    alphazero = AlphaZero(env, network,
        replay_capacity = 5n,
        min_observations_for_learning = 4n,
        observations_per_learning_step = 1,
        learning_steps_per_update = 1,
        learning_batch_size = 32)
    el = EnvironmentLoop(env, alphazero)
    @info "AlphaZero successfully created, running environment loop"

    num_episodes = 20
    run!(el, num_episodes)
    @info "Done"
end


# mcts_demo()
# mcts_demo_args()
# deepl_demo()
lcs_alphazero()
