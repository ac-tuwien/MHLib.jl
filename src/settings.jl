#=
Managing global settings via decentralized specifications.
=#

using ArgParse
using Random

export settings_cfg, settings, parse_settings!, add_arg_table!,
    get_settings_as_string, seed_random_generator, settings_new_default_value


"""
ArgParseSettings used to add further arguments in modules.
"""
const settings_cfg = ArgParseSettings()


"""
Dictionary with all parameters and their values.
"""
const settings = Dict{Symbol,Any}()

@add_arg_table! settings_cfg begin
    "--seed"
        help = "random seed, 0: initialize randomly"
        arg_type = Int
        default = 0
end


"""
    settings_new_default_value(name, value)

Set a new default value for a registered parameter.
"""
function settings_new_default_value(name::String, value)
    fields = settings_cfg.args_table.fields
    p = findfirst(x -> x.dest_name==name, fields)
    fields[p].default = value
end



"""
    parse_settings!(;args=ARGS)

Parses the arguments and stores initializing settings correspondingly,
and seed the random number generator if `settings[:seed] != 0`.
"""
function parse_settings!(args = ARGS)
    settings_cfg.fromfile_prefix_chars = Set('@')
    settings_cfg.autofix_names = true
    s = parse_args(args, settings_cfg, as_symbols=true)
    merge!(settings, s)
    seed_random_generator()
    settings
end


"""
    get_settings_as_string()

Get all parameters and their values as descriptive multi-line string.
"""
function get_settings_as_string()
    s = "Settings:\n"
    for (par, value) in settings
        s *= "--$par=$value\n"
    end
    s
end


"""
    seed_random_generator!(seed=-1)
Initialize random number generators with settings.seed.
If zero, a random seed is generated, if -1 `settings[:seed]` is used.
"""
function seed_random_generator(seed::Int=-1)
    if seed == -1
        seed = settings[:seed]
    end
    if seed == 0
        seed = rand(1:typemax(Int32))
        settings[:seed] = seed
    end
    Random.seed!(seed)
end
