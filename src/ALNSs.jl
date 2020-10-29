"""
    ALNSs

A adaptive large neighborhood search which can also be used for plain large neighborhood
search.

It extends the more general scheduler module/class by distinguishing between construction
heuristics, destroy methods and repair methods. Moreover it contain score_data which
tracks the success of methods and a next_segment attribute.
"""
module ALNSs

using MHLib
using MHLib.Schedulers
using StatsBase
using ArgParse

import MHLib.run!

export ALNS, get_number_to_destroy

#=
TODO:
- own_settings
- log_scores
=#

const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--mh_alns_segment_size"
        help = "ALNS segment size"
        arg_type = Int
        default = 100
    "--mh_alns_dest_min_abs"
        help = "ALNS minimum number of elements to destroy"
        arg_type = Int
        default = 5
    "--mh_alns_dest_max_abs"
        help = "ALNS maximum number of elements to destroy"
        arg_type = Int
        default = 100
    "--mh_alns_dest_min_ratio"
        help = "ALNS minimum ratio of elements to destroy"
        arg_type = Float64
        default = 0.1
    "--mh_alns_dest_max_ratio"
        help = "ALNS maximum ratio of elements to destroy"
        arg_type = Float64
        default = 0.35
    "--mh_alns_gamma"
        help = "ALNS reaction factor for updating the method weights"
        arg_type = Float64
        default = 0.0
    "--mh_alns_sigma1"
        help = "ALNS score for new global best solution"
        arg_type = Int
        default = 10
    "--mh_alns_sigma2"
        help = "ALNS score for better than current solution"
        arg_type = Int
        default = 9
    "--mh_alns_sigma3"
        help = "ALNS score for worse accepted solution"
        arg_type = Int
        default = 3
    "--mh_alns_init_temp_factor"
        help = "ALNS factor for determining initial temperature"
        arg_type = Float64
        default = 0.0
    "--mh_alns_temp_dec_factor"
        help = "ALNS factor for decreasing the temperature"
        arg_type = Float64
        default = 0.99
    "--mh_alns_logscores"
        help = "ALNS write out log information on scores"
        arg_type = Bool
        default = true
end


"""
    ScoreData

Weight of a method and all data relevant to calculate the score and update the weight.

Attributes
    - weight: weight to be used for selecting methods
    - score: current score in current segment
    - applied: number of applications in current segment
"""
mutable struct ScoreData
    weight::Float64
    score::Int
    applied::Int
end

ScoreData() = ScoreData(1.0, 0, 0)


"""
An adaptive large neighborhood search (ALNS).

Attributes
    - scheduler: Scheduler object
    - meths_ch: list of construction heuristic methods
    - meths_de: list of destroy methods
    - methds_re: list of repair methods
    - score_data: dictionary which stores a ScoreData struct for each method
    - temperature: temperature for Metropolis criterion
    - next_segment: iteration number of next segment for updating operator weights
"""
mutable struct ALNS
    scheduler::Scheduler
    meths_ch::Vector{MHMethod}
    meths_de::Vector{MHMethod}
    meths_re::Vector{MHMethod}
    score_data::Dict{String, ScoreData}
    temperature::Float64
    next_segment::Int
end


"""
    ALNS(sol::Solution, meths_ch, meths_de, meths_re, consider_initial_sol)

Create an ALNS.

Create a GVNS for the given solution with the given construction,
and repair methods provided as `Vector{MHMethod}`.
If `consider_initial_sol`, consider the given solution as valid initial solution;
otherwise it is assumed to be uninitialized.
"""
function ALNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_de::Vector{MHMethod},
        meths_re::Vector{MHMethod}, consider_initial_sol::Bool=false)
    # TODO own_settings
    temperature = obj(sol) * settings[:mh_alns_init_temp_factor] + 0.000000001
    score_data = Dict(m.name => ScoreData() for m in vcat(meths_de, meths_re))
    ALNS(Scheduler(sol, [meths_ch; meths_de; meths_re], consider_initial_sol),
        meths_ch, meths_de, meths_re, score_data, temperature, 0)
end


"""
    select_method(meths, weights)

Randomly select a method from the given list with probabilities proportional to the given
weights. If weights is nothing, uniform probability is used.
"""
function select_method(meths::Vector{MHMethod}, weights::Vector{Float64})
    return sample(meths, Weights(weights))
end


"""
    select_method_pair(alns)

Select a destroy and repair method pair according to current weights.
"""
function select_method_pair(alns::ALNS)
    destroy = select_method(alns.meths_de, [alns.score_data[m.name].weight
        for m in alns.meths_de])
    repair = select_method(alns.meths_re, [alns.score_data[m.name].weight
        for m in alns.meths_re])
    return destroy, repair
end


"""
    metropolis_criterion(alns, sol_new, sol_current)

Apply Metropolis criterion, return true when `sol_new` should be accepted.
"""
function metropolis_criterion(alns::ALNS, sol_new::Solution, sol_current::Solution)
    if is_better(sol_new, sol_current)
        return true
    end
    return rand() <= exp(-abs(obj(sol_new) - obj(sol_current)) / alns.temperature)
end


"""
    cool_down!(alns)

Apply geometric cooling.
"""
function cool_down!(alns::ALNS)
    alns.temperature *= settings[:mh_alns_temp_dec_factor]
end


"""
    get_number_to_destroy(num_elements, own_settings,
        dest_min_abs, dest_min_ratio, dest_max_abs, dest_max_ratio)

Randomly sample the number of elements to destroy in the destroy operator based on the
parameter settings.
"""
function get_number_to_destroy(num_elements::Int, own_settings=settings,
    dest_min_abs::Float64=own_settings[:mh_alns_dest_min_abs],
    dest_min_ratio::Float64=own_settings[:mh_alns_dest_min_ratio],
    dest_max_abs::Float64=own_settings[:mh_alns_dest_max_abs],
    dest_max_ratio::Float64=own_settings[:mh_alns_dest_max_ratio])
    a = max(dest_min_abs, Int(floor(dest_min_ratio * num_elements)))
    b = min(dest_max_abs, Int(floor(dest_max_ratio * num_elements)))
    return b >= a ? rand(a:b) : b+1
end


"""
    update_operator_weights!(alns)

Update operator weights at segment ends and re-initialize scores.
"""
function update_operator_weights!(alns::ALNS)
    if alns.scheduler.iteration == alns.next_segment
        # TODO: log_scores()
        # update operator weights
        alns.next_segment = alns.scheduler.iteration + settings[:mh_alns_segment_size]
        gamma = settings[:mh_alns_gamma]
        for m in vcat(alns.meths_de, alns.meths_re)
            data = alns.score_data[m.name]
            if data.applied > 0
                data.weight = data.weight * (1 - gamma) + gamma * data.score / data.applied
                data.score = 0
                data.applied = 0
            end
        end
    end
end


"""
    update_after_destroy_and_repair_performed!(alns, destroy, repair, sol_new,
        sol_incumbent, sol)

Update current solution, incumbent, and all operator score data according to performed
destroy+repair.
"""
function update_after_destroy_and_repair_performed!(alns::ALNS, destroy::MHMethod,
    repair::MHMethod, sol_new::Solution, sol_incumbent::Solution, sol::Solution)
    destroy_data = alns.score_data[destroy.name]
    repair_data = alns.score_data[repair.name]
    destroy_data.applied += 1
    repair_data.applied += 1
    score = 0
    if is_better(sol_new, sol_incumbent)
        # print("better than incumbent")
        score = settings[:mh_alns_sigma1]
        copy!(sol_incumbent, sol_new)
        copy!(sol, sol_new)
    elseif is_better(sol_new, sol)
        # print("better than current")
        score = settings[:mh_alns_sigma2]
        copy!(sol, sol_new)
    elseif is_better(sol, sol_new) && metropolis_criterion(alns, sol_new, sol)
        score = settings[:mh_alns_sigma3]
        # print("accepted although worse")
        copy!(sol, sol_new)
    elseif sol_new != sol
        copy!(sol_new, sol)
    end
    destroy_data.score += score
    repair_data.score += score
end


"""
    alns!(alns, sol)

Perform adaptive large neighborhood search (ALNS) on the given solution.
"""
function alns!(alns::ALNS, sol::Solution)
    alns.next_segment = alns.scheduler.iteration + settings[:mh_alns_segment_size]
    sol_incumbent = copy(sol)
    sol_new = copy(sol)
    while true
        destroy, repair = select_method_pair(alns)
        res = perform_method_pair!(alns.scheduler, destroy, repair, sol_new)
        update_after_destroy_and_repair_performed!(alns, destroy, repair, sol_new,
            sol_incumbent, sol)
        if res.terminate
            copy!(sol, sol_incumbent)
            return
        end
        update_operator_weights!(alns)
        cool_down!(alns)
    end
end


"""
    run!(alns)

Perform the construction heuristics followed by the ALNS.
"""
function run!(alns::ALNS)
    sol = copy(alns.scheduler.incumbent)
    @assert alns.scheduler.incumbent_valid || !isempty(alns.meths_ch)
    perform_sequentially!(alns.scheduler, sol, alns.meths_ch)
    alns!(alns, sol)
end

end  # module
