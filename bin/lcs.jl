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

    actions = iterate_mcts!(mcts, trace = false, trace_rollout = false, trace_actions = true)
    println("Solution: ", length(actions), ' ', actions)
    println("Overall best solution encountered: ", length(mcts.best_solution), ' ',
        mcts.best_solution)
end

# TODO GR: Tracing bitte entfernen wenn diese Funktionen soweit funktionieren

"""
    iterate_mcts!(env)

Iteratively perform MCTS, taking in each iteration the action with the most visits.
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


function lcs_alphazero()
    inst = LCSInstance(settings[:ifile])
    n = inst.n
    println(inst)
    env = LCSEnvironment(inst)
    obs = reset!(env)

    # network = DummyPolicyValueFunction(action_space_size(env))

    network = DensePolicyValueNN(observation_space_size(env), action_space_size(env))
    # network = LCSNetwork(env)

    # create AlphaZero agent with configuration as in Python implementation
    alphazero = AlphaZero(env, network,
        replay_capacity = 5n,
        min_observations_for_learning = 4n,
        observations_per_learning_step = 1,
        learning_steps_per_update = 1,
        learning_batch_size = 32)
    el = EnvironmentLoop(env, alphazero)
    println("AlphaZero successfully created, running environment loop")

    run!(el, 25)    # episodes without (much) learning yet
    run!(el, 200)    # episodes with learning
    println("Done")
end


println("LCS-Alphazero Demo version $(git_version())\nARGS: ", ARGS)

# Ignore actual arguments here, using explicitly specified ones
settings_new_default_value!(MHLib.settings_cfg, "ifile", "data/test-04_003_050.lcs")
parse_settings!([MHLib.RL.settings_cfg, MHLib.MCTSs.settings_cfg, MHLib.LCS.settings_cfg],
    ["--seed=0",
    #"--ifile=data/rat-04_010_600.lcs",
    "--ifile=data/test-04_003_050.lcs",
    "--mh_mcts_num_sims=200",
    "--lcs_reward_mode=smallsteps",
    "--mh_mcts_c_uct=1.0",
    "--mh_mcts_tree_policy=PUCT",
    "--lcs_prior_heuristic=UB1",
    "--lcs_always_new_seqs=true",
    "--rl_ldir=none",
    ])
println(get_settings_as_string())

# mcts_demo()
lcs_alphazero()
