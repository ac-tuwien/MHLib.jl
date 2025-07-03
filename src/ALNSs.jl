#     ALNSs.jl
#
# A adaptive large neighborhood search.
#
# It extends the more general scheduler module/class by distinguishing between construction
# heuristics, destroy methods and repair methods. Moreover it contain score_data which
# tracks the success of methods and a next_segment attribute.


export ALNS, ALNSParameters, ALNSMethodSelector

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
An Adaptive Large Neighborhood Search (ALNS).

# Elements
- `score_data_de`: dictionary which stores a ScoreData struct for each destroy method
- `score_data_re`: dictionary which stores a ScoreData struct for each repair method
- `next_segment`: iteration number of next segment for updating operator weights

# Configuration Parameters
- `segment_size`: size of segments for updating method weights
- `gamma`: reaction factor for updating the method weights
- `sigma1`: score for new global best solution
- `sigma2`: score for better than current solution
- `sigma3`: score for worse accepted solution
"""
mutable struct ALNSMethodSelector <: MethodSelector
    const score_data_de::Vector{ALNSScoreData}
    const score_data_re::Vector{ALNSScoreData}
    next_segment::Int
    const segment_size::Int
    const gamma::Float64
    const sigma1::Int
    const sigma2::Int
    const sigma3::Int
end

ALNSMethodSelector(meths_de::Vector{MHMethod}, meths_re::Vector{MHMethod}, args...) =
    ALNSMethodSelector([ALNSScoreData() for _ in 1:length(meths_de)], 
        [ALNSScoreData() for _ in 1:length(meths_re)], 0, args...) 


"""
    ALNS(sol::Solution, meths_ch, meths_de, meths_re; 
        segment_size::Int=100, gamma::Float64=0.025, 
        sigma1::Int=10, sigma2::Int=9, sigma3::Int=3,
        kwargs...)

Create an Adaptive Large Neighborhood Search (ALNS).

Create an ALNS, i.e., LNS with `ALNSMethodSelector` for the given solution `sol` with 
the given construction, destroy, and repair methods provided as `Vector{MHMethod}`.

# Configuration Parameters
- `segment_size`: size of segments for updating method weights
- `gamma`: reaction factor for updating the method weights
- `sigma1`: score for new global best solution
- `sigma2`: score for better than current solution
- `sigma3`: score for worse accepted solution
- `kwargs`: further configuration parameters from `LNS` and `SchedulerConfig` passed to them
"""
function ALNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_de::Vector{MHMethod},
        meths_re::Vector{MHMethod}; 
        segment_size::Int=100, gamma::Float64=0.025, 
        sigma1::Int=10, sigma2::Int=9, sigma3::Int=3,
        kwargs...)
    method_selector = ALNSMethodSelector(meths_de, meths_re, 
        segment_size, gamma, sigma1, sigma2, sigma3)
    LNS(sol, meths_ch, meths_de, meths_re; method_selector, kwargs...)
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
        sel.next_segment = lns.scheduler.iteration + sel.segment_size
        gamma = sel.gamma
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
    sel.next_segment = lns.scheduler.iteration + sel.segment_size
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
        score = sel.sigma1
    elseif case == :notWorseThanCurrent
        score = sel.sigma2
    elseif case == :acceptedAlthoughWorse
        score = sel.sigma3
    end
    destroy_data.score += score
    repair_data.score += score
    update_operator_weights!(lns)
end
