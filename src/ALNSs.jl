
module ALNSs

using MHLib
using MHLib.Schedulers
using ArgParse

export ALNS, run!

@add_arg_table! settings_cfg begin
    "--mh_alns_segment_size"
        help = "ALNS segment size"
        arg_type = Int
        default = 0 # indicates that there are no segments
    "--mh_alns_gamma"
        help = "ALNS reaction factor for updating the method weights"
        arg_type = Float64
        default = 0.0 # indicates that there is no reaction
#=
    "--mh_alns_dest_min_abs"
        help = "ALNS minimum number of elements to destroy"
        arg_type = Int
        default = 5
    "--mh_alns_dest_max_abs"
        help = "ALNS maximum number of elements to destroy"
        arg_type = Int
        default = 100
    "--mh_alns_dest_ratio"
        help = "ALNS ratio of elements to destroy"
        arg_type = Float64
        default = 0.1
=#
end


"""
Weight of a method and all data relevant to calculate the score and update the weight.

Attributes
- weight: weight to be used for selecting methods
- score: current score in current segment
- applied: number of applications in current segment
"""
mutable struct ScoreData
    weight::Float64
    score::Int64
    applied::Int64
end

ScoreData() = new(1.0, 0, 0)


"""
An adaptive large neighborhood search (ALNS).

Attributes
- scheduler: Scheduler object
- meths_ch: list of construction heuristic methods
- meths_de: list of destroy methods
- methds_re: list of repair methods
- weights_de: list of probabilities that determine how likely a destroy method is applied;
sum of the list must give 1.0
- weights_re: list of probabilities that determine how likely a repair method is applied;
sum of the list must give 1.0
"""
mutable struct ALNS
    scheduler::Scheduler
    meths_ch::Vector{MHMethod}
    meths_de::Vector{MHMethod}
    meths_re::Vector{MHMethod}
    score_data::Dict{String, ScoreData}
    next_segment::Int64
end

"""
    ALNS(solution, meths_ch, meths_de, meths_re, consider_initial_sol=false)

Create a ALNS

Create a GVNS for the given solution with the given construction,
and repair methods provided as Vector{MHMethod}.
If consider_initial_sol, consider the given solution as valid initial solution;
otherwise it is assumed to be uninitialized.
"""
function ALNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_de::Vector{MHMethod},
    meths_re::Vector{MHMethod}, consider_initial_sol::Bool=false)
    # TODO own_settings
    score_data = Dict(m.name => ScoreData() for m in vcat(meths_de, meths_re))
    ALNS(Scheduler(sol, [meths_ch; meths_de; meths_re]), meths_ch, meths_de, meths_re, score_data, 0)
end


"""
    select_method(meths, weights)

Randomly select a method from the given list with probabilities proportional to the given weights.
If weights is nothing, uniform probability is used.
"""
function select_method(meths::Vector{MHMethod}, weights::Union{Vector{Float64}, Nothing}=nothing)
    # when can weights be nothing??
    if weights == nothing
        return meths[rand(1:length(meths))]
    else
        return sample(meths, Weights(weights))
    end
end


"""
    select_method_pair(alns)

Select a destroy and repair method pair according to current weights.
"""
function select_method_pair(alns::ALNS)
    destroy = select_method(alns.meths_de, [alns.score_data[m.name]].weight for m in alns.meths_de)
    repair = select_method(alns.meths_re, [alns.score_data[m.name]].weight for m in alns.meths_re)
    return destroy, repair
end

"""
    update_operator_weights!(alns)

Update operator weights at segment ends and re-initialize scores
"""
function update_operator_weights!(alns::ALNS)
    if alns.scheduler.iteration == alns.next_segment
        # TODO: log_scores()
        alns.next_segment = alns.scheduler.iteration + settings_cfg["mh_alns_segment_size"]
        gamma = settings_cfg["mh_alns_gamma"]
        for m in vcat(alns.meths_de, alns.meths_re)
            data = alns.score_data[m.name]
            if data.applied
                data.weight = data.weight * (1 - gamma) + gamma * data.score / data.applied
                data.score = 0
                data.applied = 0
            end
        end
    end
end


"""
    update_after_destroy_and_repair_performed!(alns, destroy, repair, sol_new, sol_incumbent, sol)

Update current solution, incumbent, and all operator score data according to performed destroy+repair.
"""
function update_after_destroy_and_repair_performed!(alns::ALNS, destroy::MHMethod, repair::MHMethod,
    sol_new::Solution, sol_incumbent::Solution, sol::Solution)
    destroy_data = alns.score_data[destroy.name]
    repair_data = alns.score_data[repair.name]
    destroy_data.applied += 1
    repair_data.applied += 1
    score = 0
    if is_better(sol_new, sol_incumbent)
        #TODO: score = self.own_settings.mh_alns_sigma1
        score = 1
        copy!(sol_incumbent, sol_new)
        copy!(sol, sol_new)
    elseif is_better(sol_new, sol)
        #TODO: score = self.own_settings.mh_alns_sigma2
        score = 1
        copy!(sol, sol_new)
    #= TODO
    elif sol.is_better(sol_new) and self.metropolis_criterion(sol_new, sol):
        score = self.own_settings.mh_alns_sigma3
        # print('accepted although worse')
        sol.copy_from(sol_new)
    =#
    elseif sol_new != sol
        copy!(sol_new, sol)
    end
    destroy_data.score += score
    repair_data.score += score
end


"""
    alns(alns, solution)

Perform adaptive large neighborhood search (ALNS) on a given solution.
"""
function alns!(alns::ALNS, sol::Solution)
    alns.next_segment = alns.scheduler.iteration + settings_cfg["mh_alns_segment_size"]
    sol_incumbent = copy(sol)
    sol_new = copy(sol)
    while true
        destroy, repair = select_method_pair(alns)
        res = perform_method_pair!(alns.scheduler, destroy, repair, sol_new)
        update_after_destroy_and_repair_performed!(alns, destroy, repair, sol_new, sol_incumbent, sol)
        if res.terminate
            copy!(sol, sol_incumbent)
            return
        end
        update_operator_weights(alns)
    end
end


"""
    run(alns)

Performs the construction heuristics followed by the ALNS.
"""
function run!(alns::ALNS)
    sol = copy(alns.scheduler.incumbent)
    @assert alns.scheduler.incumbent_valid || !isempty(alns.meths_ch)
    perform_sequentially!(alns.scheduler, sol, alns.meths_ch)
    alns!(alns, sol)
end

end  # module
