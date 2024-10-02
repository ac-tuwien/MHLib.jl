"""
    Schedulers

General scheduler for realizing (G)VNS, GRASP, IG and similar metaheuristics.

The module is intended for metaheuristics in which a set of methods
(or several of them) are in some way repeatedly applied to candidate solutions.
"""
module Schedulers

using ArgParse
using Random
using DataStructures  # SortedDict for method_stats
using MHLib

import Logging: AbstractLogger

export Result, MHMethod, MHMethodStatistics, Scheduler, SchedulerParameters, 
    perform_method!, next_method, update_incumbent!, check_termination, 
    perform_sequentially!, main_results, method_statistics, delayed_success_update!, 
    log_iteration, log_iteration_header, construct!, local_improve!, shaking!, 
    perform_method_pair!, reinitialize!

const settings_cfg = ArgParseSettings()

@add_arg_table! settings_cfg begin
    "--mh_titer"
        help = "maximum number of iterations (<0: turned off)"
        arg_type = Int
        default = 100
    "--mh_lnewinc"
        help = "always write iteration log if new incumbent solution"
        arg_type = Bool
        default = true
    "--mh_lfreq"
        help = "frequency of writing iteration logs (0: none, >0: number of iterations, " *
               "-1: iteration 1,2,5,10,20,..."
        arg_type = Int
        default = 0
    "--mh_tciter"
        help = "maximum number of iterations without improvement (<0: turned off)"
        arg_type = Int
        default = -1
    "--mh_ttime"
        help = "time limit [s] (<0: turned off)"
        arg_type = Float64
        default = -1.0
    "--mh_tctime"
        help = "maximum time [s] without improvement (<0: turned off)"
        arg_type = Float64
        default = -1.0
    "--mh_tobj"
        help = "objective value at which should be terminated when reached (<0: turned off)"
        arg_type = Float64
        default = -1.0
    "--mh_checkit"
        help = "call `check` for each solution after each method application"
        arg_type = Bool
        default = false
end

"""
    SchedulerParameters

Parameters for the scheduler adopted from settings by default.
"""
Base.@kwdef struct SchedulerParameters
    titer::Int = settings[:mh_titer]
    lnewinc::Bool = settings[:mh_lnewinc]
    lfreq::Int = settings[:mh_lfreq]
    tciter::Int = settings[:mh_tciter]
    ttime::Float64 = settings[:mh_ttime]
    tctime::Float64 = settings[:mh_tctime]
    tobj::Float64 = settings[:mh_tobj]
    checkit::Bool = settings[:mh_checkit]
end


"""
    Result

Data in conjunction with a method application's result.

Attributes
- `changed`: if `false`, the solution has not been changed by the method application
- `terminate`: if `true`, a termination condition has been fulfilled
- `log_info`: customized log info
"""
mutable struct Result
    changed::Bool
    terminate::Bool
    log_info::String
end

Result() = Result(true, false, "")


"""
    MHMethod

A method to be applied to a solution by the scheduler.

Attributes
- 'name`: name of the method; must be unique over all used methods
- `method`: a function called for a Solution object
- `par`: a parameter provided when calling the method
"""
struct MHMethod
    name::String
    func::Function
    par::Any
end


"""
    MHMethodStatistics

Struct that collects data on the applications of a `MHMethod`.

Attributes
- `applications`: number of applications of this method
- `netto_time`: accumulated time of all applications of this method without further costs
    (e.g., VND)
- `successes`: number of applications in which an improved solution was found
- `obj_gain`: sum of gains in the objective values over all successful applications
- `brutto_time`: accumulated time of all applications of this method including further
    costs (e.g., VND)
"""
mutable struct MHMethodStatistics
    applications::Int
    netto_time::Float64
    successes::Int
    obj_gain::Float64
    brutto_time::Float64
end

MHMethodStatistics() = MHMethodStatistics(0, 0.0, 0, 0.0, 0.0)


"""
    Scheduler{TSolution <: Solution}

Type for metaheuristics that work by iteratively applying certain methods/operations.

Attributes
- `incumbent`: incumbent solution, i.e., initial solution and always best solution so far
    encountered
- `incumbent_valid`: `true` if incumbent is a valid solution to be considered
- `incumbent_iteration`: iteration in which incumbent was found
- `incumbent_time`: time at which incumbent was found
- `methods`: vector of all `MHMethods`
- `method_stats`: dict of `MHMethodStatistics` for each `MHMethod`
- `iteration`: overall number of method applications
- `time_start`: starting time of algorithm
- `run_time`: overall runtime (set when terminating)
- `params`: parameters adopted from settings by default
"""
mutable struct Scheduler{TSolution <: Solution}
    incumbent::TSolution
    incumbent_valid::Bool
    incumbent_iteration::Int
    incumbent_time::Float64
    methods::Vector{MHMethod}
    method_stats::SortedDict{String,MHMethodStatistics}
    iteration::Int
    time_start::Float64
    run_time::Union{Float64, Missing}
    logger::AbstractLogger
    params::SchedulerParameters
end

"""
    Scheduler(solution, methods, consider_initial_sol)

Create a `MHMethod` scheduler.

Create a Scheduler for the given solution with the given methods provides as
`Vector{MHMethod}`. If `consider_initial_sol` is `true`, consider the given solution as
valid initial solution; otherwise it is assumed to be uninitialized.

"""
function Scheduler(sol::Solution, methods::Vector{MHMethod},
        consider_initial_sol::Bool=false; params::SchedulerParameters=SchedulerParameters())
    method_stats = Dict([(m.name, MHMethodStatistics()) for m in methods])
    logger = Log.get_logger(sol)
    s = Scheduler{typeof(sol)}(sol, consider_initial_sol, 0, 0.0, methods, method_stats, 0,
        time(), missing, logger, params)
    log_iteration_header(s)
    if s.incumbent_valid
        log_iteration(s, "-", NaN, sol, true, true, "")
    end
    s
end

"""
    reinitialize!(::Scheduler, solution)

Reset scheduler with given solution, which however, is not considered, for a new run.
"""
function reinitialize!(s::Scheduler{TSolution}, sol::TSolution) where {TSolution <: Solution}
    s.incumbent = sol     
    s.incumbent_valid = false
    s.incumbent_iteration = 0
    s.incumbent_time = 0.0
    s.iteration = 0
    s.time_start = time()
    s.run_time = missing
    for ms in values(s.method_stats)
        ms.applications = 0
        ms.netto_time = 0.0
        ms.successes = 0
        ms.obj_gain = 0.0
        ms.brutto_time = 0.0
    end
end

"""
    update_incumbent!(scheduler, solution, current_time)

If the given solution is better than the incumbent (or we do not have an incumbent yet)
update it.
"""
function update_incumbent!(s::Scheduler, sol::Solution, current_time::Float64)
    if !s.incumbent_valid || is_better(sol, s.incumbent)
        copy!(s.incumbent, sol)
        s.incumbent_iteration = s.iteration
        s.incumbent_time = current_time
        s.incumbent_valid = true
        return true
    end
    false
end


"""
    next_method(meths; randomize=false, repeat=false)

Generator for obtaining a next method from a given vector of methods, iterating through
all of them. `randomize`: random order, otherwise consider given order;
`repeat`: repeat infinitely, otherwise just do one pass.
"""
function next_method(meths::Vector{MHMethod}; randomize::Bool=false, repeat::Bool=false)
    if randomize
        meths = meths.copy()
    end
    function gen_methods(channel::Channel)
        while true
            if randomize
                shuffle!(meths)
            end
            for method in meths
                put!(channel, method)
            end
            if !repeat
                break
            end
        end
    end
    Channel(gen_methods)
end


"""
    perform_method!(scheduler, method, solution; delayed_success=false)::Result

Perform method on given solution and return `Results` object.

Also updates incumbent, iteration and the method's statistics in method_stats.
Furthermore checks the termination condition and eventually sets terminate in the
returned Results object. If `delayed_success`, the success is not immediately determined
and the statistics updated accordingly but at some later call of `delayed_success_update`.
"""
function perform_method!(s::Scheduler, method::MHMethod, sol::Solution;
        delayed_success=false)::Result
    res = Result()
    obj_old = obj(sol)
    t_start = time()
    method.func(sol, method.par, res)
    t_end = time()
    if s.params.checkit
        check(sol)
    end
    ms = s.method_stats[method.name]
    ms.applications += 1
    ms.netto_time += t_end - t_start
    obj_new = obj(sol)
    if !delayed_success
        ms.brutto_time += t_end - t_start
        if is_better_obj(sol, obj(sol), obj_old)
            ms.successes += 1
            ms.obj_gain += obj_new - obj_old
        end
    end
    s.iteration += 1
    new_incumbent = update_incumbent!(s, sol, t_end - s.time_start)
    terminate = check_termination(s)
    log_iteration(s, method.name, obj_old, sol, new_incumbent, terminate, res.log_info)
    if terminate
        s.run_time = time() - s.time_start
        res.terminate = true
    end
    res
end


"""
    check_termination(scheduler)

Check termination conditions and return `true` when to terminate.
"""
function check_termination(s::Scheduler)::Bool
    t = time()
    if 0 <= s.params.titer <= s.iteration ||
        0 <= s.params.tciter <= s.iteration - s.incumbent_iteration ||
        0 <= s.params.ttime <= t - s.time_start ||
        0 <= s.params.tctime::Float64 <= t - s.incumbent_time ||
        0 <= s.params.tobj && !is_worse_obj(s.incumbent, obj(s.incumbent), 
            s.params.tobj)
        return true
    end
    false
end


"""
    perform_sequentially!(scheduler, solution, methods)

Applies the given methods sequentially, finally keeping the best solution as incumbent.

Returns true if the termination condition has been fulfilled, else false.
"""
function perform_sequentially!(s::Scheduler, sol::Solution, meths::Vector{MHMethod})
    for m in next_method(meths)
        res = perform_method!(s, m, sol)
        update_incumbent!(s, sol, time() - s.time_start)
        res.terminate && return true
    end
    return false
end



"""
    delayed_success_update!(scheduler, method, obj_old, t_start, solution)

Update an earlier performed method's success information in method_stats using the
given solution, old objective value and the given thime when the application of the
method had started.
"""
function delayed_success_update!(s::Scheduler, method::MHMethod, obj_old, t_start::Float64,
    sol::Solution)
    t_end = time()
    ms = s.method_stats[method.name]
    ms.brutto_time += t_end - t_start
    obj_new = obj(sol)
    if is_better_obj(sol, obj(sol), obj_old)
        ms.successes += 1
        ms.obj_gain += obj_new - obj_old
    end
end


"""
    perform_method_pair!(scheduler, destroy, repair, sol)

Performs a destroy/repair method pair on given solution and returns `Results` structure.

Also updates incumbent, iteration and the method's statistics in method_stats.
Furthermore checks the termination condition and eventually sets terminate in the 
returned `Results` structure.
"""
function perform_method_pair!(scheduler::Scheduler, destroy::MHMethod, repair::MHMethod, 
        sol::Solution)
    res = Result()
    obj_old = obj(sol)
    t_start = time()
    destroy.func(sol, destroy.par, res)
    t_destroyed = time()
    repair.func(sol, repair.par, res)
    t_end = time()
    if scheduler.params.checkit
        check(sol)
    end                                      
    update_stats_for_method_pair!(scheduler, destroy, repair, sol, res, obj_old,
                                      t_destroyed - t_start, t_end - t_destroyed)
    return res
end

"""
    update_stats_for_method_pair!(sched, dest, repair, sol, res, obj_old, t_dest, t_repair)

Update statistics, incumbent and check termination condition after having performed a
destroy+repair.
"""
function update_stats_for_method_pair!(scheduler::Scheduler, destroy::MHMethod,
         repair::MHMethod, sol::Solution, res::Result, obj_old, t_destroy::Float64,
         t_repair::Float64)
     ms_destroy = scheduler.method_stats[destroy.name]
     ms_destroy.applications += 1
     ms_destroy.netto_time += t_destroy
     ms_destroy.brutto_time += t_destroy
     ms_repair = scheduler.method_stats[repair.name]
     ms_repair.applications += 1
     ms_repair.netto_time += t_repair
     ms_repair.brutto_time += t_repair
     obj_new = obj(sol)
     if is_better_obj(sol, obj_new, obj_old)
         ms_destroy.successes += 1
         ms_destroy.obj_gain += obj_new - obj_old
         ms_repair.successes += 1
         ms_repair.obj_gain += obj_new - obj_old
     end
     scheduler.iteration += 1
     new_incumbent = update_incumbent!(scheduler, sol, time() - scheduler.time_start)
     terminate = check_termination(scheduler)
     log_iteration(scheduler, destroy.name * "+" * repair.name, obj_old, sol, new_incumbent, terminate, res.log_info)
     if terminate
         scheduler.run_time = time() - scheduler.time_start
         res.terminate = true
     end
end


# --------------------- Diverse generic Scheduler methods -----------------------

"""
    construct!(solution, par, result)

Scheduler method that constructs a new solution.
Will usually be specialized for a specific problem.
"""
construct!(s::Solution, par::Int, result::Result) =
    initialize!(s)

"""
    local_improve!(solution, par, result)

Scheduler method that tries to locally improve the solution.
Will usually be specialized for a specific problem.
This abstract implementation just throws an exception.
"""
local_improve!(s::Solution, par::Int, result::Result) =
    error("Abstract method local_improve! called")

"""
    shaking!(solution, par, result)

Scheduler method that performs shaking.
Will usually be specialized for a specific problem.
This abstract implementation just throws an exception.
"""
shaking!(s::Solution, par::Int, result::Result) =
    error("Abstract method shaking! called")

"""
    local_improve!(bool_vector_solution, par, result)

Scheduler method that tries to locally improve the solution.
Perform one `k_flip_neighborhood_search`.
"""
local_improve!(s::BoolVectorSolution, par::Int, result::Result) =
    k_flip_neighborhood_search!(s, par, false)

"""
    shaking!(bool_vector_solution, par, result)

Scheduler method that performs shaking.
Will usually be specialized for a specific problem.
This abstract implementation just throws an exception.
"""
shaking!(s::BoolVectorSolution, par::Int, result::Result) =
    k_random_flips!(s, par)


end  # module
