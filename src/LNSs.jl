"""
    LNSs

A basic large neighborhood search.

It extends the more general scheduler module/class by distinguishing between construction
heuristics, destroy methods and repair methods.
"""
module LNSs

using MHLib
using MHLib.Schedulers
using StatsBase
using ArgParse

export LNS, get_number_to_destroy


const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--lns_init_temp_factor"
        help = "LNS factor for determining initial temperature"
        arg_type = Float64
        default = 0.0
    "--lns_temp_dec_factor"
        help = "LNS factor for decreasing the temperature"
        arg_type = Float64
        default = 0.99
end

"""
    LNSParameters

Parameters for LNS adopted from settings by default.
"""
Base.@kwdef struct LNSParameters
    init_temp_factor::Float64 = settings[:lns_init_temp_factor]
    temp_dec_factor::Float64 = settings[:lns_temp_dec_factor]
    logscores::Bool = true
end

"""
    LNS

A basic large neighborhood search.

Attributes
    - scheduler: Scheduler object
    - meths_ch: list of construction heuristic methods
    - meths_de: list of destroy methods
    - meths_re: list of repair methods
    - meths_compat: Boolean matrix indicating which repair method can be applied
        in conjunction with which destroy method
    - temperature: temperature for Metropolis criterion
    - params: LNSParameters, by default adopted from global settings
"""
mutable struct LNS
    scheduler::Scheduler
    meths_ch::Vector{MHMethod}
    meths_de::Vector{MHMethod}
    meths_re::Vector{MHMethod}
    meths_compat::Union{Nothing, Matrix{Bool}}
    temperature::Float64
    params::LNSParameters
end


"""
    LNS(sol::Solution, meths_ch, meths_de, meths_re, consider_initial_sol,
        meths_compat, params)

Create a LNS.

Create a LNS for the given solution with the given construction,
and repair methods provided as `Vector{MHMethod}`.
If `consider_initial_sol`, consider the given solution as valid initial solution;
otherwise it is assumed to be uninitialized.
"""
function LNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_de::Vector{MHMethod},
        meths_re::Vector{MHMethod}; consider_initial_sol::Bool=false, meths_compat=nothing,
        params=LNSParameters())
    temperature = obj(sol) * params.init_temp_factor + 0.000000001
    LNS(Scheduler(sol, [meths_ch; meths_de; meths_re], consider_initial_sol),
    meths_ch, meths_de, meths_re, meths_compat, temperature, params)
end


"""
    select_method_pair(lns)

Select a destroy and repair method pair according to current weights.
"""
function select_method_pair(lns::LNS)
    de_idx = rand(1:length(lns.meths_de))
    destroy = lns.meths_de[de_idx]
    if isnothing(lns.meths_compat)
        meths_re = lns.meths_re
    else
        meths_re = lns.meths_re[lns.meths_compat[de_idx, :]]
    end
    repair = rand(meths_re)
    return destroy, repair
end


"""
    metropolis_criterion(lns, sol_new, sol_current)

Apply Metropolis criterion, return true when `sol_new` should be accepted.
"""
function metropolis_criterion(lns::LNS, sol_new::Solution, sol_current::Solution)
    if is_better(sol_new, sol_current)
        return true
    end
    return rand() <= exp(-abs(obj(sol_new) - obj(sol_current)) / lns.temperature)
end


"""
    cool_down!(lns)

Apply geometric cooling.
"""
function cool_down!(lns::LNS)
    lns.temperature *= lns.params.temp_dec_factor
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
    update_after_destroy_and_repair_performed!(lns, destroy, repair, sol_new,
        sol_incumbent, sol)

Update current solution, incumbent, and all operator score data according to performed
destroy+repair.
"""
function update_after_destroy_and_repair_performed!(lns::LNS, destroy::MHMethod,
    repair::MHMethod, sol_new::Solution, sol_incumbent::Solution, sol::Solution)
    if is_better(sol_new, sol_incumbent)
        # print("better than incumbent")
        copy!(sol_incumbent, sol_new)
        copy!(sol, sol_new)
    elseif is_better(sol_new, sol)
        # print("better than current")
        copy!(sol, sol_new)
    elseif is_better(sol, sol_new) && metropolis_criterion(lns, sol_new, sol)
        # print("accepted although worse")
        copy!(sol, sol_new)
    elseif sol_new != sol
        copy!(sol_new, sol)
    end
end


"""
    lns!(lns, sol)

Perform basic large neighborhood search (LNS) on the given solution.
"""
function lns!(lns::LNS, sol::Solution)
    sol_incumbent = copy(sol)
    sol_new = copy(sol)
    while true
        destroy, repair = select_method_pair(lns)
        res = perform_method_pair!(lns.scheduler, destroy, repair, sol_new)
        update_after_destroy_and_repair_performed!(lns, destroy, repair, sol_new,
            sol_incumbent, sol) 
        if res.terminate
            copy!(sol, sol_incumbent)
            return
        end
        cool_down!(lns)
    end
end


"""
    run!(lns)

Perform the construction heuristics followed by a LNS.
"""
function MHLib.run!(lns::LNS)
    sol = copy(lns.scheduler.incumbent)
    @assert lns.scheduler.incumbent_valid || !isempty(lns.meths_ch)
    perform_sequentially!(lns.scheduler, sol, lns.meths_ch)
    lns!(lns, sol)
end

end  # module
