# LNSs.jl
#
# A basic large neighborhood search.
#
# It extends the more general scheduler module/class by distinguishing between construction
# heuristics, destroy methods and repair methods.

export LNS, LNSParameters, MethodSelector, UniformRandomMethodSelector, 
    WeightedRandomMethodSelector, destroy!, repair!, ResultCase, reinitialize!


"""
    MethodSelector

Abstract type for selecting the repair and destroy methods within the LNS.
"""
abstract type MethodSelector end

"""
    LNS{TMethodSelector <: MethodSelector, TSolution <: Solution}

A basic large neighborhood search.

Attributes
- `init_temp_factor`: factor for determining the initial temperature factor
- `temp_dec_factor`: factor for determining the temperature decay factor
- `solution`: current solution
- `scheduler`: `Scheduler`
- `meths_ch`: list of construction heuristic methods
- `meths_de`: list of destroy methods
- `meths_re`: list of repair methods
- `meths_compat`: Boolean matrix indicating which destroy method can be applied
    in conjunction with which repair method, i.e., `meths_compat[i,j]==true` indicates
    that the i-th destroy method can be applied with the j-th repair method
- `temperature`: temperature for Metropolis criterion
- `method_selector`: method selector for selecting destroy and repair methods
- `init_temp_factor`: factor for determining the initial temperature
- `temp_dec_factor`: factor for determining the temperature decay factor
"""
mutable struct LNS{TMethodSelector <: MethodSelector, TSolution <: Solution}
    solution::TSolution
    new_solution::TSolution
    scheduler::Scheduler{TSolution}
    meths_ch::Vector{MHMethod}
    meths_de::Vector{MHMethod}
    meths_re::Vector{MHMethod}
    meths_compat::Union{Nothing, Matrix{Bool}}
    temperature::Float64
    method_selector::TMethodSelector
    init_temp_factor::Float64
    temp_dec_factor::Float64
end

"""
    ResultCase

Enumeration type for type of result of method application.
"""
@enum ResultCase betterThanIncumbent notWorseThanCurrent acceptedAlthoughWorse rejected


"""
    LNS(sol::Solution, meths_ch, meths_de, meths_re;
        meths_compat=nothing, consider_initial_sol=false, 
        method_selector=UniformRandomMethodSelector(),
        init_temp_factor=0.0, temp_dec_factor=0.99, kwargs...)

Create a Large Neighborhood Search (LNS).

Create an LNS for the given solution with the given construction,
and repair methods provided as `Vector{MHMethod}`.
If `consider_initial_sol`, consider the given solution as valid initial solution;
otherwise it is assumed to be uninitialized.
Parameter `meths_compat` is either `nothing` or a Boolean matrix indicating which destroy 
method can be applied in conjunction with which repair method.
Parameter `method_selector` is the technique used for selecting the destroy and repair methods.
The `kwargs` are passed to the `SchedulerParameters` constructor and therefore can
contain any element of `SchedulerParameters` as keyword argument, e.g., `titer`, etc.
"""
function LNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_de::Vector{MHMethod},
        meths_re::Vector{MHMethod}; meths_compat::Union{Nothing, Matrix{Bool}}=nothing,
        consider_initial_sol::Bool=false, 
        method_selector::MethodSelector=UniformRandomMethodSelector(),
        init_temp_factor::Float64=0.0, temp_dec_factor::Float64=0.99, kwargs...)
    temperature = obj(sol) * init_temp_factor
    scheduler = Scheduler(sol, [meths_ch; meths_de; meths_re]; consider_initial_sol, kwargs...)
    lns = LNS{typeof(method_selector), typeof(sol)}(sol, copy(sol), scheduler, 
        meths_ch, meths_de, meths_re, meths_compat, temperature, method_selector, 
        init_temp_factor, temp_dec_factor)
    init_method_selector!(lns)
    return lns
end

"""
    reinitialize!(::LNS, sol)

Reset the LNS to the given solution with possibly a new problem instance for a new run.
"""
function reinitialize!(lns::LNS{<:MethodSelector, TSolution}, 
        sol::TSolution) where {TSolution <: Solution}
    copy!(lns.solution, sol)
    copy!(lns.new_solution, sol)
    reinitialize!(lns.scheduler, sol)
    lns.temperature = obj(sol) * lns.init_temp_factor
    init_method_selector!(lns)
end

"""
    destroy!(solution, par, result)

Scheduler method that performs destroy.
Will usually be specialized for a specific problem.
This abstract implementation just throws an exception.
"""
destroy!(s::Solution, par::Int, result::Result) =
    error("Abstract method destroy! called")

"""
    repair!(solution, par, result)

Scheduler method that performs repair.
Will usually be specialized for a specific problem.
This abstract implementation just throws an exception.
"""
repair!(s::Solution, par::Int, result::Result) =
    error("Abstract method repair! called")


"""
    metropolis_criterion(lns, sol_new, sol_current)

Apply Metropolis criterion, return true when `sol_new` should be accepted.
When the new solution has equal objective value, we also accept it.
"""
function metropolis_criterion(lns::LNS, sol_new::Solution, sol_current::Solution)
    if !is_worse(sol_new, sol_current) :: Bool
        return true
    end
    if iszero(lns.temperature)
        return false
    else
        return rand() <= exp(-abs(obj(sol_new) - obj(sol_current)) / lns.temperature)
    end
end


"""
    cool_down!(lns)

Apply geometric cooling.
"""
function cool_down!(lns::LNS)
    lns.temperature *= lns.temp_dec_factor
end


"""
    update_solution!(lns, sol_new, sol)

Update current solution and incumbent according to the result of performing a
destroy and repair method pair. Returns the case of the update.
"""
function update_solution!(lns::LNS, sol_new::Solution, sol::Solution)
    if lns.scheduler.iteration == lns.scheduler.incumbent_iteration
        # print("better than incumbent")
        copy!(sol, sol_new)
        case = betterThanIncumbent
    elseif !is_worse(sol_new, sol)
        # print("not worse than current")
        copy!(sol, sol_new)
        case = notWorseThanCurrent
    elseif is_better(sol, sol_new) && metropolis_criterion(lns, sol_new, sol)
        # print("accepted although worse")
        copy!(sol, sol_new)
        case = acceptedAlthoughWorse
    elseif sol_new != sol
        copy!(sol_new, sol)
        case = rejected
    end
    return case
end

"""
    init_method_selector!(::LNS)

Initialize the method selector.

Default implementation does nothing.
"""
init_method_selector!(::LNS) = nothing

"""
    update_method_selector!(lns, destroy, repair, case, Δ, Δ_inc)

Update the method selector according to the result of last performed method pair.

Default implementation does nothing.
"""
update_method_selector!(::LNS, destroy::Int, repair::Int, case::ResultCase, Δ, Δ_inc) = 
    nothing

"""
    lns_iteration!(lns, destroy_idx, repair_idx)

Perform one iteration of the LNS using the provided destroy and repair method indices.
"""
function lns_iteration!(lns::LNS, destroy_idx::Union{Nothing,Int}=nothing,
        repair_idx::Union{Nothing,Int}=nothing) :: Result
    destroy = isnothing(destroy_idx) ? select_method(lns, eachindex(lns.meths_de), true) : 
        destroy_idx
    repair = isnothing(repair_idx) ? select_repair_method(lns, destroy) : repair_idx
    res = perform_method_pair!(lns.scheduler, lns.meths_de[destroy], 
        lns.meths_re[repair], lns.new_solution)
    obj_new_solution = obj(lns.new_solution)
    Δ = obj_new_solution - obj(lns.solution)
    Δ_inc = obj_new_solution - obj(lns.scheduler.incumbent)
    case = update_solution!(lns, lns.new_solution, lns.solution) 
    update_method_selector!(lns, destroy, repair, case, Δ, Δ_inc)
    cool_down!(lns)
    res
end


"""
    lns!(lns, sol)

Perform basic large neighborhood search (LNS) on the given solution.
"""
function lns!(lns::LNS, sol::Solution)
    lns.solution = sol
    lns.new_solution = copy(sol)
    while true
        res = lns_iteration!(lns)
        if res.terminate
            copy!(lns.solution, lns.scheduler.incumbent)
            return
        end
    end
end



"""
    run!(lns)

Perform the construction heuristics followed by a LNS.
"""
function run!(lns::LNS)
    sol = copy(lns.scheduler.incumbent)
    @assert lns.scheduler.incumbent_valid || !isempty(lns.meths_ch)
    terminate = perform_sequentially!(lns.scheduler, sol, lns.meths_ch)
    terminate && return
    lns!(lns, sol)
end


"""
    select_repair_method(lns, destroy::Int)

Select a repair method that is compatible to the given destroy method.
"""
function select_repair_method(lns::LNS, destroy::Int)
    if isnothing(lns.meths_compat)
        repair_candidates = eachindex(lns.meths_re)
    else
        compat = view(lns.meths_compat,destroy, :)
        repair_candidates = eachindex(lns.meths_re)[compat]
    end
    return select_method(lns, repair_candidates, false)
end


"""
    UniformRandomMethodSelector

Uniformly randomly select a repair and destroy method.
"""
struct UniformRandomMethodSelector <: MethodSelector end

"""
    select_method(::LNS{UniformRandomMethodSelector})

Select a method uniformly at random.
"""
function select_method(::LNS{UniformRandomMethodSelector}, 
        candidates, is_destroy::Bool) :: Int
    return rand(candidates)
end


"""
    WeightedRandomMethodSelector

Select indices of destroy and repair methods according to constant pre-specified weights.
"""
struct WeightedRandomMethodSelector <: MethodSelector
    weights_de::Vector{Float64}
    weights_re::Vector{Float64}
end

"""
    select_method(::LNS{WeightedRandomMethodSelector}, candidates, is_destroy)

Select a method proportionally to the weights at random.
"""
function select_method(lns::LNS{WeightedRandomMethodSelector}, candidates, 
        is_destroy::Bool) :: Int
    sel = lns.method_selector
    weights = is_destroy ? sel.weights_de : sel.weights_re
    return sample(candidates, Weights(weights))
end

