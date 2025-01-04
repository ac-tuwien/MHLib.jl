#     Log.jl

# Custom logging functionallity for MHLib.  
    
# Implements a basic logger type and required methods.  
# Provides a method to return a simple logger and allows users to overload this method to 
# incoperate custom user defined loggers. Additionally exports differnt methods to 
# output optimization results and interation data.

# The logging process works as follows.

# - If the user does not set the `--ofile` argument than no log file is created. 
#     Print statements are still sent to stdout as is normal
# - If the user does set the `--ofile` argument then by default a simple logger is 
#     created with the `get_logger()` function. This logger is an instance of the 
#     MHLogger type that simply writes the message directly to a file with 
#     no additional information.
# - If the user overloads the `get_logger` method than any logger can be returned. 
#     This allows for the user to create their own custom loggers that save the solution
#     data in different ways.

using Logging
using Printf

export  log_iteration_header, log_iteration, method_statistics, main_results


# Define log level constants

"""
    IterLevel

LogLevel for iteration data.
"""
const IterLevel = LogLevel(1)

"""
    StatsLevel

LogLevel for optimization statistics data.
"""
const StatsLevel = LogLevel(2)

"""
    SummaryLevel

LogLevel for optimization results summary.
"""
const SummaryLevel = LogLevel(3)

"""
    HeaderLevel

LogLevel for iteration header string.
"""
const HeaderLevel = LogLevel(4)


# --------------------- Message Logger ---------------------

"""
    MHLogger

Custom logger type for MHLib.
"""
struct MHLogger <: AbstractLogger
    io::IO                          # Output IO Stream
    levels::Vector{LogLevel}        # Use empty vector to enable all levels
end

# Constructors
MHLogger(io::IO) = MHLogger(io, Vector{LogLevel}())
MHLogger(file::String) = MHLogger(open(file, "w"), Vector{LogLevel}())
MHLogger(file::String, lvls::Vector{LogLevel}) = MHLogger(open(file,"w"), lvls)

# Required Logging Methods
Logging.min_enabled_level(logger::MHLogger) =
    isempty(logger.levels) ? Logging.BelowMinLevel : minimum(logger.levels)

Logging.shouldlog(logger::MHLogger, level, _module, group, id) =
    isempty(logger.levels) ? true : in(level, logger.levels)

Logging.catch_exceptions(logger::MHLogger) = true

function Logging.handle_message(logger::MHLogger, lvl, msg, _mod, group, id, file, line;
        kwargs...)
    println(logger.io, msg) 
    flush(logger.io)
end


# --------------------- Logger generation method -----------------------

"""
    get_logger(::Solution)

Returns `logger <: AbstractLogger` to log output to.  

If the argument `--ofile` is set output will be saved to the file specified 
as well as to `stdout`.  Additionally users can overload the `get_logger` 
method with their own deffinition to customize the logging output. 
"""
function get_logger(::Solution)
    if settings[:ofile] == ""
        logger = NullLogger()
    else
        logger = MHLogger(settings[:ofile])
    end
    return logger
end


# --------------------- Output logging methods ---------------------
"""
    log_iteration_header(scheduler)

Write iteration log header.
"""
function log_iteration_header(sched::Scheduler)
    s = "I       iter             best          obj_old          obj_new" *
        "        time              method info"
    println(s)
    with_logger(sched.logger) do 
        @logmsg HeaderLevel s
    end 
end


function is_logarithmic_number(x::Int)::Bool
    EPS = 1e-12
    LOG10_2 = log10(2)
    LOG10_5 = log10(5)
    lr = log10(x) % 1
    abs(lr) < EPS || abs(lr-LOG10_2) < EPS || abs(lr-LOG10_5) < EPS
end


"""
    log_iteration(scheduler, method_name, obj_old, new_sol, new_incumbent, in_any_case,
        log_info)

Writes iteration log info.

A line is written if in_any_case is set or in dependence of
`params.lfreq` and `params.lnewinc`.
`method_name`: name of applied method or "-" (if initially given solution);
`obj_old`: objective value before applying last operator;
`param new_sol`: newly created solution;
`new_incumbent`: true if the method yielded a new incumbent solution;
`in_any_case`: turns filtering of iteration logs off;
`log_info`: customize log info optionally added if not ""
"""
function log_iteration(sched::Scheduler, method_name::String, obj_old, new_sol::Solution,
        new_incumbent::Bool, in_any_case::Bool, log_info::String="")
    log = in_any_case || new_incumbent && sched.params.lnewinc
    if !log
        lfreq = sched.params.lfreq
        if lfreq > 0 && sched.iteration % lfreq == 0
            log = true
        elseif lfreq < 0 && is_logarithmic_number(sched.iteration)
            log = true
        end
    end
    if log
        s = @sprintf("I %10d %16.5f %16.5f %16.5f%12.4f%20s %s",
            sched.iteration, obj(sched.incumbent), obj_old, obj(new_sol),
            time()-sched.time_start, method_name, log_info)
        println(s)
        with_logger(sched.logger) do 
            @logmsg IterLevel s iter=sched.iteration best_obj=obj(sched.incumbent) prev_obj=obj_old cur_obj=obj(new_sol) time=time()-sched.time_start method=method_name info=log_info cur_sol=new_sol
        end
    end
end

"""
    sdiv(x::Real, y::Real)

Safe division: return x/y if y!=0 and nan otherwise.
"""
sdiv(x::Real, y::Real) = (y == 0) ? NaN : x/y

"""
    method_statistics(s::Scheduler)

Write overall statistics.
"""
function method_statistics(s::Scheduler)

    if s.run_time === missing
        s.run_time = time() - s.time_start
    end

    total_applications = 0
    total_netto_time = 0.0
    total_successes = 0
    total_brutto_time = 0.0
    total_obj_gain = 0.0

    for ms in values(s.method_stats)
        total_applications += ms.applications
        total_netto_time += ms.netto_time
        total_successes += ms.successes
        total_brutto_time += ms.brutto_time
        total_obj_gain += ms.obj_gain
    end

    res = "\nMethod statistics\n" *
          "S  method    iter  succ  succ-rate%  tot-obj-gain  avg-obj-gain  rel-succ%  net-time  " *
          "net-time%  brut-time  brut-time%\n"

    for key in keys(s.method_stats)
        e = s.method_stats[key]
        temp = ("S  " * key * "       ")[1:11]
        res *=@sprintf("%s%6d%6d%12.5f%14.5f%14.5f%11.5f%10.5f%11.5f%11.5f%12.5f",
          temp, e.applications, e.successes, sdiv(e.successes, e.applications) * 100,
          e.obj_gain, sdiv(e.obj_gain, e.applications),
          sdiv(e.successes, total_successes) * 100,
          e.netto_time, sdiv(e.netto_time, s.run_time) * 100,
          e.brutto_time, sdiv(e.brutto_time, s.run_time) * 100)
        res *= "\n"
    end

    temp = ("S  SUM/AVG       ")[1:11]
    res *=@sprintf("%s%6d%6d%12.5f%14.5f%14.5f%11.5f%10.5f%11.5f%11.5f%12.5f",
      temp, total_applications, total_successes, sdiv(total_successes, total_applications) * 100,
      total_obj_gain, sdiv(total_obj_gain, total_applications),
      sdiv(sdiv(total_successes, length(s.method_stats)), total_successes) * 100,
      total_netto_time, sdiv(total_netto_time, s.run_time) * 100,
      total_brutto_time, sdiv(total_brutto_time, s.run_time) * 100)
    res *= "\n"

    println(res)
    with_logger(s.logger) do 
        @logmsg StatsLevel res
    end
end


"""
    main_results(scheduler)

Print main results.
"""
function main_results(s::Scheduler)
    str = "T best solution: $(s.incumbent)\nT best obj: $(obj(s.incumbent))\n" *
        "T best iteration: $(s.incumbent_iteration)\n" *
        "T total iterations: $(s.iteration)\n" *
        @sprintf("T best time [s]: %.3f\n", s.incumbent_time) *
        @sprintf("T total time [s]: %.4f\n", s.run_time)
    print(str)
    with_logger(s.logger) do 
        @logmsg SummaryLevel str
    end
    check(s.incumbent)
end
