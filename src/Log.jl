"""
    Log

Custom logging functionallity of MHLib.  Implements a basic logger type and methods to generate differnt
loggers based on the input arguments.  Also provides functionallity to incoperate custom user defined loggers. 
"""

module Log

# Includes
using ArgParse
using MHLib
using Logging, LoggingExtras


# export  ?


# Define Log Level Constants
"""
LogLevel for iteration data
"""
const IterLevel = LogLevel(1)
"""
LogLevel for optimization statistics data
"""
const StatsLevel = LogLevel(2)
"""
LogLevel for optimization results summary
"""
const SummaryLevel = LogLevel(3)
"""
LogLevel for iteration header string
"""
const HeaderLevel = LogLevel(4)

"""
Standard ArgParseSettings always used.
"""
const settings_cfg = ArgParseSettings()
@add_arg_table! settings_cfg begin
    "--log_file"
    help = "File to log outputs. Input filename or \"None\" to dissable"
    arg_type = String
    default = "None"
end

# --------------------- Message Logger ---------------------
struct MHLogger <: AbstractLogger
    io::IO                          # Output IO Stream
    levels::Vector{LogLevel}        # Use empty vector to enable all levels
end

# Constructors
MHLogger(io::IO) = MHLogger(io, Vector{LogLevel}())
MHLogger(file::String) = MHLogger(open(file, "w"), Vector{LogLevel}())
MHLogger(file::String, lvls::Vector{LogLevel}) = MHLogger(open(file,"w"), lvls)

# Required Logging Methods
function Logging.min_enabled_level(logger::MHLogger)
    return isempty(logger.levels) ? Logging.BelowMinLevel : minimum(logger.levels)
end

function Logging.shouldlog(logger::MHLogger, level, _module, group, id) 
    return isempty(logger.levels) ? true : in(level, logger.levels)
 end

Logging.catch_exceptions(logger::MHLogger) = true

function Logging.handle_message(logger::MHLogger,
    lvl, msg, _mod, group, id, file, line;
    kwargs...)

    # Writes message to IO stream and then flushes the stream
    println(logger.io, msg) 
    flush(logger.io)
end


# --------------------- Methods -----------------------
"""
    get_logger(::Solution)

Returns `logger<:AbstractLogger` to log output to.  If the argument `--log_file` is set outputs will be
saved to the file specified as well as to `stdout`.  Additionally users can overload the `get_logger` 
method with their own deffinition to custom the logging output. 
"""
function get_logger(sol::Solution)
    if settings[:log_file] == "None"
        logger = NullLogger()
    else
        logger = MHLogger(settings[:log_file])
    end
    return logger
end


end