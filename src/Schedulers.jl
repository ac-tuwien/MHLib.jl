# Schedulers.jl
#
# General scheduler for realizing (G)VNS, GRASP, IG and similar metaheuristics.
#
# The module is intended for metaheuristics in which a set of methods
# (or several of them) are in some way repeatedly applied to candidate solutions.

using DataStructures  # SortedDict for method_stats

export Result, MHMethod, MHMethodStatistics, Scheduler, SchedulerConfig, 
    perform_method!, next_method, update_incumbent!, check_termination, 
    perform_sequentially!, main_results, method_statistics, delayed_success_update!, 
    log_iteration, log_iteration_header, construct!, local_improve!, shaking!, 
    perform_method_pair!, reinitialize!


"""
    SchedulerConfig

Configuration parameters for `Scheduler`.

# Elements
- `checkit`: call `check` for each solution after each method application
- `consider_initial_sol`: if set, consider the given solution as valid initial solution,
    otherwise it is assumed to be uninitialized
- `log`: if true write all log information, else none
- `lnewinc`: always write iteration log if new incumbent solution
- `lfreq`: frequency of writing iteration logs (0: none, >0: number of iterations, -1: 
    iteration 1,2,5,10,20,...)
- `titer`: maximum number of iterations (<0: turned off)
- `ttime`: time limit in seconds (<0: turned off)
- `tciter`: maximum number of iterations without improvement (<0: turned off)
- `tctime`: maximum time in seconds without improvement (<0: turned off)
- `tobj`: objective value at which should be terminated when reached (<0: turned off)
"""
Base.@kwdef struct SchedulerConfig
    checkit::Bool = false
    consider_initial_sol::Bool = false
    log::Bool = true
    lnewinc::Bool = true
    lfreq::Int = 0
    titer::Int = 100
    ttime::Float64 = -1.0
    tciter::Int = -1
    tctime::Float64 = -1.0
    tobj::Float64 = -1.0
end


"""
    Result

Data in conjunction with a method application's result.

# Elements
- `changed`: if `false`, the solution has not been changed by the method application
- `is_local_optimum`: if `true`, the solution is considered a local optimum in respect to
    the applied method
- `terminate`: if `true`, a termination condition has been fulfilled
- `log_info`: customized log info
"""
mutable struct Result
    changed::Bool
    is_local_optimum::Bool
    terminate::Bool
    log_info::String
end

Result() = Result(true, false, false, "")


"""
    MHMethod

(Wrapper for) a method (function) to be applied to a solution by the scheduler.

# Elements
- 'name`: name of the method; must be unique over all used methods
- `method`: a function called for a Solution object; 
    the function must have exactly three parameters, which are the solution,
        a general parameter of arbitrary type (or `nothing`), and a `Result` structure
- `par`: a parameter provided when calling the method; can be `nothing`
"""
struct MHMethod
    name::String
    method::Function
    par::Any
end

MHMethod(name::String, method::Function) = MHMethod(name, method, nothing)


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

# Elements
- `config`: configuration parameters for the scheduler, see `SchedulerConfig`
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
"""
mutable struct Scheduler{TSolution <: Solution}
    const config::SchedulerConfig
    incumbent::TSolution
    incumbent_valid::Bool
    incumbent_iteration::Int
    incumbent_time::Float64
    const methods::Vector{MHMethod}
    const method_stats::Dict{String, MHMethodStatistics}
    iteration::Int
    time_start::Float64
    run_time::Union{Float64, Missing}
    const logger::AbstractLogger
end

"""
    Scheduler(solution, methods; kwargs...)

Create a `MHMethod` scheduler.

Create a Scheduler for the given solution with the given methods provides as
`Vector{MHMethod}`.

The `kwargs` provide various configuration parameters and are used to initialize a 
`SchedulerConfig` structure; thus see `SchedulerConfig` for the available keyword arguments
and their meaning.
"""
function Scheduler(sol::Solution, methods::Vector{MHMethod}; kwargs...)
    config = SchedulerConfig(; kwargs...)
    method_stats = Dict([(m.name, MHMethodStatistics()) for m ∈ methods])
    logger = get_logger(sol)
    sched = Scheduler(config, sol, config.consider_initial_sol, 0, 0.0, methods, 
        method_stats, 0, time(), missing, logger)
    log_iteration_header(sched)
    if sched.incumbent_valid
        log_iteration(sched, "-", NaN, sol, true, true, "")
    end
    return sched
end

"""
    reinitialize!(::Scheduler, solution)

Reset scheduler with given solution, which however, is not considered, for a new run.
"""
function reinitialize!(sched::Scheduler{TSolution}, sol::TSolution) where 
        {TSolution <: Solution}
    sched.incumbent = sol     
    sched.incumbent_valid = false
    sched.incumbent_iteration = 0
    sched.incumbent_time = 0.0
    sched.iteration = 0
    sched.time_start = time()
    sched.run_time = missing
    for ms in values(sched.method_stats)
        ms.applications = 0
        ms.netto_time = 0.0
        ms.successes = 0
        ms.obj_gain = 0.0
        ms.brutto_time = 0.0
    end
end

"""
    update_incumbent!(scheduler, solution, current_time)

If given solution is better than incumbent or we do not have an incumbent yet update it.
"""
function update_incumbent!(sched::Scheduler, sol::Solution, current_time::Float64)
    if !sched.incumbent_valid || is_better(sol, sched.incumbent)
        copy!(sched.incumbent, sol)
        sched.incumbent_iteration = sched.iteration
        sched.incumbent_time = current_time
        sched.incumbent_valid = true
        return true
    end
    false
end


"""
    next_method(meths; randomize=false, repeat=false)

Generator for obtaining a next method from a given vector of methods.
    
It iterates through all methods.

# Parameters
- `randomize`: random order, otherwise consider given order
- `repeat`: repeat infinitely, otherwise just do one pass
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
function perform_method!(sched::Scheduler, method::MHMethod, sol::Solution;
        delayed_success=false)::Result
    res = Result()
    obj_old = obj(sol)
    t_start = time()
    method.method(sol, method.par, res)
    t_end = time()
    if sched.config.checkit
        check(sol)
    end
    ms = sched.method_stats[method.name]
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
    sched.iteration += 1
    new_incumbent = update_incumbent!(sched, sol, t_end - sched.time_start)
    terminate = check_termination(sched)
    log_iteration(sched, method.name, obj_old, sol, new_incumbent, terminate, res.log_info)
    if terminate
        sched.run_time = time() - sched.time_start
        res.terminate = true
    end
    res
end


"""
    check_termination(scheduler)

Check termination conditions and return `true` when to terminate.
"""
function check_termination(sched::Scheduler)::Bool
    t = time()
    config = sched.config
    if 0 <= config.titer <= sched.iteration ||
        0 <= config.tciter <= sched.iteration - sched.incumbent_iteration ||
        0 <= config.ttime <= t - sched.time_start ||
        0 <= config.tctime::Float64 <= t - sched.incumbent_time ||
        0 <= config.tobj && !is_worse_obj(sched.incumbent, obj(sched.incumbent), 
            config.tobj)
        return true
    end
    false
end


"""
    perform_sequentially!(scheduler, solution, methods)

Applies the given methods sequentially, finally keeping the best solution as incumbent.

Returns true if the termination condition has been fulfilled, else false.
"""
function perform_sequentially!(sched::Scheduler, sol::Solution, meths::Vector{MHMethod})
    for m in next_method(meths)
        res = perform_method!(sched, m, sol)
        update_incumbent!(sched, sol, time() - sched.time_start)
        res.terminate && return true
    end
    return false
end



"""
    delayed_success_update!(scheduler, method, obj_old, t_start, solution)

Update an earlier performed method's success information in method_stats.

Uses the given solution, old objective value and the given thime when the application of the
method had started.
"""
function delayed_success_update!(sched::Scheduler, method::MHMethod, obj_old, 
        t_start::Float64, sol::Solution)
    t_end = time()
    ms = sched.method_stats[method.name]
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
function perform_method_pair!(sched::Scheduler, destroy::MHMethod, repair::MHMethod, 
        sol::Solution)
    res = Result()
    obj_old = obj(sol)
    t_start = time()
    destroy.method(sol, destroy.par, res)
    t_destroyed = time()
    repair.method(sol, repair.par, res)
    t_end = time()
    if sched.config.checkit
        check(sol)
    end                                      
    update_stats_for_method_pair!(sched, destroy, repair, sol, res, obj_old,
                                      t_destroyed - t_start, t_end - t_destroyed)
    return res
end

"""
    update_stats_for_method_pair!(sched, dest, repair, sol, res, obj_old, t_dest, t_repair)

Update statistics, incumbent, and check termination condition.

To be applied after having performed a destroy+repair.
"""
function update_stats_for_method_pair!(sched::Scheduler, destroy::MHMethod,
         repair::MHMethod, sol::Solution, res::Result, obj_old, t_destroy::Float64,
         t_repair::Float64)
     ms_destroy = sched.method_stats[destroy.name]
     ms_destroy.applications += 1
     ms_destroy.netto_time += t_destroy
     ms_destroy.brutto_time += t_destroy
     ms_repair = sched.method_stats[repair.name]
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
     sched.iteration += 1
     new_incumbent = update_incumbent!(sched, sol, time() - sched.time_start)
     terminate = check_termination(sched)
     log_iteration(sched, destroy.name * "+" * repair.name, obj_old, sol, new_incumbent, terminate, res.log_info)
     if terminate
         sched.run_time = time() - sched.time_start
         res.terminate = true
     end
end


# ------------- Diverse generic scheduler methods (MHMethod functions)m--------------------

"""
    construct!(::Solution, par, result)

`MHMethod` that constructs a new solution.

Will usually be specialized for a specific problem.
"""
construct!(sol::Solution, ::Nothing, ::Result) = initialize!(sol)

"""
    local_improve!(::Solution, par, result)

`MHMethod` that tries to locally improve the solution.

To be specialized for a specific problem.
"""
function local_improve! end

"""
    shaking!(::Solution, par, result)

`MHMethod` that performs shaking.

To be specialized for a specific problem.
"""
function shaking! end

"""
    local_improve!(::BoolVectorSolution, par, result)

`MHMethod` that tries to locally improve the solution.

Perform one `k_flip_neighborhood_search`.
"""
local_improve!(s::BoolVectorSolution, par::Int, ::Result) =
    k_flip_neighborhood_search!(s, par, false)

"""
    shaking!(::BoolVectorSolution, k, result)

`MHethod` that performs shaking by flipping `k` random bits.
"""
shaking!(s::BoolVectorSolution, k::Int, ::Result) = k_random_flips!(s, k)

