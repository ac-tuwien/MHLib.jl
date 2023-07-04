"""
    ALNSs

A adaptive large neighborhood search.

It extends the more general scheduler module/class by distinguishing between construction
heuristics, destroy methods and repair methods. Moreover it contain score_data which
tracks the success of methods and a next_segment attribute.
"""
module ALNSs

using MHLib
using MHLib.Schedulers
using StatsBase
using ArgParse

export ALNS, get_number_to_destroy


const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--alns_segment_size"
        help = "ALNS segment size"
        arg_type = Int
        default = 100
    "--alns_gamma"
        help = "ALNS reaction factor for updating the method weights"
        arg_type = Float64
        default = 0.0
    "--alns_sigma1"
        help = "ALNS score for new global best solution"
        arg_type = Int
        default = 10
    "--alns_sigma2"
        help = "ALNS score for better than current solution"
        arg_type = Int
        default = 9
    "--alns_sigma3"
        help = "ALNS score for worse accepted solution"
        arg_type = Int
        default = 3
    "--alns_init_temp_factor"
        help = "ALNS factor for determining initial temperature"
        arg_type = Float64
        default = 0.0
    "--alns_temp_dec_factor"
        help = "ALNS factor for decreasing the temperature"
        arg_type = Float64
        default = 0.99
    "--alns_logscores"
        help = "ALNS write out log information on scores"
        arg_type = Bool
        default = true
end

"""
    ALNSParameters

Parameters for the ALNS algorithm adopted from settings by default.
"""
Base.@kwdef struct ALNSParameters
    segment_size::Int = settings[:alns_segment_size]
    gamma::Float64 = settings[:alns_gamma]
    sigma1::Int = settings[:alns_sigma1]
    sigma2::Int = settings[:alns_sigma2]
    sigma3::Int = settings[:alns_sigma3]
    init_temp_factor::Float64 = settings[:alns_init_temp_factor]
    temp_dec_factor::Float64 = settings[:alns_temp_dec_factor]
    logscores::Bool = settings[:alns_logscores]
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
    - meths_re: list of repair methods
    - meths_compat: Boolean matrix indicating which repair method can be applied
        in conjunction with which destroy method
    - score_data: dictionary which stores a ScoreData struct for each method
    - temperature: temperature for Metropolis criterion
    - next_segment: iteration number of next segment for updating operator weights
    - params: ALNSParameters, by default adopted from global settings
"""
mutable struct ALNS
    scheduler::Scheduler
    meths_ch::Vector{MHMethod}
    meths_de::Vector{MHMethod}
    meths_re::Vector{MHMethod}
    meths_compat::Union{Nothing, Matrix{Bool}}
    score_data::Dict{String, ScoreData}
    temperature::Float64
    next_segment::Int
    params::ALNSParameters
end


"""
    ALNS(sol::Solution, meths_ch, meths_de, meths_re, consider_initial_sol)

Create an ALNS.

Create an ALNS for the given solution with the given construction,
and repair methods provided as `Vector{MHMethod}`.
If `consider_initial_sol`, consider the given solution as valid initial solution;
otherwise it is assumed to be uninitialized.
"""
function ALNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_de::Vector{MHMethod},
        meths_re::Vector{MHMethod}; consider_initial_sol::Bool=false, meths_compat=nothing,
        params=ALNSParameters())
    temperature = obj(sol) * params.init_temp_factor + 0.000000001
    score_data = Dict(m.name => ScoreData() for m in vcat(meths_de, meths_re))
    ALNS(Scheduler(sol, [meths_ch; meths_de; meths_re], consider_initial_sol),
    meths_ch, meths_de, meths_re, meths_compat, score_data, temperature, 0, params)
end


"""
    select_method_pair(alns)

Select a destroy and repair method pair according to current weights.
"""
function select_method_pair(alns::ALNS)
    weights = [alns.score_data[m.name].weight for m in alns.meths_de]
    de_idx = sample(Weights(weights))
    destroy = alns.meths_de[de_idx]
    if isnothing(alns.meths_compat)
        meths_re = alns.meths_re
    else
        meths_re = alns.meths_re[alns.meths_compat[de_idx, :]]
    end
    repair = sample(meths_re, Weights([alns.score_data[m.name].weight for m in meths_re]))
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
    alns.temperature *= alns.params.temp_dec_factor
end


"""
    get_number_to_destroy(num_elements;
        dest_min_abs, dest_min_ratio, dest_max_abs, dest_max_ratio)

Randomly sample the number of elements to destroy in the destroy operator based on
minimum and maximum numbers and ratios.
"""
function get_number_to_destroy(num_elements::Int;
    min_abs=5, max_abs=100, min_ratio=0.5, max_ratio=0.35)
    a = max(min_abs, floor(Int, min_ratio * num_elements))
    b = min(max_abs, floor(Int, max_ratio * num_elements))
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
        alns.next_segment = alns.scheduler.iteration + alns.params.segment_size
        gamma = alns.params.gamma
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
        score = alns.params.sigma1
        copy!(sol_incumbent, sol_new)
        copy!(sol, sol_new)
    elseif is_better(sol_new, sol)
        # print("better than current")
        score = alns.params.sigma2
        copy!(sol, sol_new)
    elseif is_better(sol, sol_new) && metropolis_criterion(alns, sol_new, sol)
        score = alns.params.sigma3
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
    alns.next_segment = alns.scheduler.iteration + alns.params.segment_size
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
function MHLib.run!(alns::ALNS)
    sol = copy(alns.scheduler.incumbent)
    @assert alns.scheduler.incumbent_valid || !isempty(alns.meths_ch)
    perform_sequentially!(alns.scheduler, sol, alns.meths_ch)
    alns!(alns, sol)
end

end  # module
