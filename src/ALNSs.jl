#     ALNSs.jl

# A adaptive large neighborhood search.

# It extends the more general scheduler module/class by distinguishing between construction
# heuristics, destroy methods and repair methods. Moreover it contain score_data which
# tracks the success of methods and a next_segment attribute.


export ALNS, ALNSParameters, alns_settings_cfg, ALNSMethodSelector


const alns_settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--alns_segment_size"
        help = "ALNS segment size"
        arg_type = Int
        default = 100
    "--alns_gamma"
        help = "ALNS reaction factor for updating the method weights"
        arg_type = Float64
        default = 0.025
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
end


"""
    ALNSScoreData

Weight of a method and all data relevant to calculate the score and update the weight.

Attributes
- `weight`: weight to be used for selecting methods
- `score`: current score in current segment
- `applied`: number of applications in current segment
"""
mutable struct ALNSScoreData
    weight::Float64
    score::Int
    applied::Int
end

ALNSScoreData() = ALNSScoreData(1.0, 0, 0)


"""
An adaptive large neighborhood search (ALNS).

Attributes
- `score_data_de`: dictionary which stores a ScoreData struct for each destroy method
- `score_data_re`: dictionary which stores a ScoreData struct for each repair method
- `next_segment`: iteration number of next segment for updating operator weights
- `params`: `ALNSParameters`, by default adopted from global settings
"""
mutable struct ALNSMethodSelector <: MethodSelector
    score_data_de::Vector{ALNSScoreData}
    score_data_re::Vector{ALNSScoreData}
    next_segment::Int
    params::ALNSParameters
end

ALNSMethodSelector(meths_de::Vector{MHMethod}, meths_re::Vector{MHMethod};
        params::ALNSParameters=ALNSParameters()) =
    ALNSMethodSelector([ALNSScoreData() for _ in 1:length(meths_de)], 
        [ALNSScoreData() for _ in 1:length(meths_re)], 0, params)


"""
    ALNS(sol::Solution, meths_ch, meths_de, meths_re; 
        meths_compat, consider_initial_sol, scheduler_params, lns_params, alns_params)

Create an Adaptive Large Neighborhood Search (ALNS).

Create an ALNS, i.e., LNS with `ALNSMethodSelector`` for the given solution with 
the given construction, and repair methods provided as `Vector{MHMethod}`.
If `consider_initial_sol`, consider the given solution as valid initial solution;
otherwise it is assumed to be uninitialized.
"""
function ALNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_de::Vector{MHMethod},
        meths_re::Vector{MHMethod}; 
        meths_compat::Union{Nothing, Matrix{Bool}}=nothing,
        consider_initial_sol::Bool=false, scheduler_params=SchedulerParameters(),
        lns_params=LNSParameters(), params=ALNSParameters())
    method_selector = ALNSMethodSelector(meths_de, meths_re; params)
    LNS(sol, meths_ch, meths_de, meths_re; meths_compat, consider_initial_sol,
        method_selector, scheduler_params, params=lns_params)
end

"""
    select_method(::LNS{ALNSMethodSelector}, candidates, is_destroy) :: Int

Select a method proportionally to the scores at random.
"""
function select_method(lns::LNS{ALNSMethodSelector}, 
        candidates, is_destroy::Bool) :: Int
    sel = lns.method_selector
    score_data = is_destroy ? sel.score_data_de : sel.score_data_re
    weights = [score_data[i].weight for i in candidates]
    return sample(candidates, Weights(weights))
end

"""
    update_operator_weights!(::LNS{ALNSMethodSelector})

Update operator weights at segment ends and re-initialize scores.
"""
function update_operator_weights!(lns::LNS{ALNSMethodSelector})
    sel = lns.method_selector
    iteration = lns.scheduler.iteration
    if iteration == sel.next_segment
        # update operator weights
        sel.next_segment = lns.scheduler.iteration + sel.params.segment_size
        gamma = sel.params.gamma
        for data in Iterators.flatten((sel.score_data_de, sel.score_data_re))
            if data.applied > 0
                data.weight = data.weight * (1 - gamma) + gamma * data.score / data.applied
                data.score = 0
                data.applied = 0
            end
        end
        # @show sel.score_data_de sel.score_data_re
    end
end

"""
    init_method_selector!(::LNS{ALNSMethodSelector})

Initialize method selector with current state of LNS.
"""
function init_method_selector!(lns::LNS{ALNSMethodSelector})
    sel = lns.method_selector
    sel.next_segment = lns.scheduler.iteration + sel.params.segment_size
end

"""
    update_after_destroy_and_repair_performed!(::LNS{ALNSMethodSelector}, 
        destroy, repair, case)

Update score data according to performed destroy+repair and case of result.
"""
function update_method_selector!(lns::LNS{ALNSMethodSelector}, 
        destroy::Int, repair::Int, case::Symbol,  Δ, Δ_inc)
    sel = lns.method_selector
    destroy_data = sel.score_data_de[destroy]
    repair_data = sel.score_data_re[repair]
    destroy_data.applied += 1
    repair_data.applied += 1
    score = 0
    if case == :betterThanIncumbent
        score = sel.params.sigma1
    elseif case == :notWorseThanCurrent
        score = sel.params.sigma2
    elseif case == :acceptedAlthoughWorse
        score = sel.params.sigma3
    end
    destroy_data.score += score
    repair_data.score += score
    update_operator_weights!(lns)
end
