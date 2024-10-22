"""
    GVNSs

A general variable neighborhood search class which can also be used for plain local search,
VND, GRASP, IG etc.

It extends the more general scheduler module/class by distinguishing between construction
heuristics, local improvement methods and shaking methods.
"""
module GVNSs

using MHLib
using MHLib.Schedulers

export GVNS, vnd!, gvns!


"""
    GVNS

A general variable neighborhood search (GVNS).

Attributes
- `scheduler`: Scheduler object
- `meths_ch`: list of construction heuristic methods
- `meths_li`: list of local improvement methods
- `meths_sh`: list of shaking methods
"""
mutable struct GVNS{TSolution <: Solution}
    scheduler::Scheduler{TSolution}
    meths_ch::Vector{MHMethod}
    meths_li::Vector{MHMethod}
    meths_sh::Vector{MHMethod}
end

"""
    GVNS{TSolution <: Solution(solution, meths_ch, meths_li, meths_sh, 
        consider_initial_sol=false)

Create a GVNS.

Create a GVNS for the given solution with the given construction,
local improvement, and shaking methods provides as `Vector{MHMethod}`.
If `consider_initial_sol` is true, consider the given solution as valid initial solution;
otherwise it is assumed to be uninitialized.
"""
function GVNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_li::Vector{MHMethod},
        meths_sh::Vector{MHMethod}; consider_initial_sol::Bool=false)
    GVNS{typeof(sol)}(Scheduler(sol, [meths_ch; meths_li; meths_sh], consider_initial_sol),
        meths_ch, meths_li, meths_sh)
end


"""
    vnd(scheduler, solution)

Perform variable neighborhood descent (VND) on given solution.
Return true if a global termination condition is fulfilled, else false.
"""
function vnd!(gvns::GVNS, sol::Solution)::Bool
    sol2 = copy(sol)
    improvement_found = true
    is_local_optimum = false
    while improvement_found && !is_local_optimum
        for m in next_method(gvns.meths_li)
            is_local_optimum = false
            res = perform_method!(gvns.scheduler, m, sol2)
            if is_better(sol2, sol)
                copy!(sol, sol2)
                res.terminate && return true
                improvement_found = true
                if res.is_local_optimum
                    is_local_optimum = true
                else
                    break
                end
            else
                res.terminate && return true
                if res.changed
                    copy!(sol2, sol)
                end
                improvement_found = false
            end
        end
    end
    false
end


"""
    gvns(gvns, solution)

Perform general variable neighborhood search (GVNS) to given solution.
"""
function gvns!(gvns::GVNS, sol::Solution)
    sol2 = copy(sol)
    use_vnd = !isempty(gvns.meths_li)
    if use_vnd && vnd!(gvns, sol2) || isempty(gvns.meths_sh)
        return
    end
    improvement_found = true
    while improvement_found
        for m in next_method(gvns.meths_sh, repeat=true)
            t_start = time()
            res = perform_method!(gvns.scheduler, m, sol2, delayed_success=use_vnd)
            terminate = res.terminate
            if !terminate && use_vnd
                terminate = vnd!(gvns, sol2)
            end
            delayed_success_update!(gvns.scheduler, m, obj(sol), t_start, sol2)
            if is_better(sol2, sol)
                copy!(sol, sol2)
                if terminate || res.terminate
                    return
                end
                improvement_found = true
                break
            else
                if terminate || res.terminate
                    return
                end
                improvement_found = false
                copy!(sol2, sol)
            end
        end
    end
end


"""
    run!(gvns)

Actually performs the construction heuristics followed by the GVNS.
"""
function MHLib.run!(gvns::GVNS)
    sol = copy(gvns.scheduler.incumbent)
    @assert gvns.scheduler.incumbent_valid || !isempty(gvns.meths_ch)
    terminate = perform_sequentially!(gvns.scheduler, sol, gvns.meths_ch)
    terminate && return
    gvns!(gvns, sol)
end


end  # module
