"""
    MAXSAT

MAXSAT demo problem.

The goal is to maximize the number of clauses satisfied in a boolean function given in
conjunctive normal form.
"""

using Random
using StatsBase
using MHLib

export MAXSATInstance, MAXSATSolution, destroy!, repair!, solve_maxsat, maxsat_settings_cfg

"""
A MAXSAT problem instance.

The goal is to maximize the number of clauses satisfied in a boolean function given in
conjunctive normal form.

Attributes
- `n`: number of variables, i.e., size of incidence vector
- `m`: number of clauses
- `clauses`: vector of clauses, where each clause is represented by a
    vector of integers;
    a positive integer v refers to the v-th variable, while a negative
    integer v refers to the negated form of the v-th variable
- `variable_usage`: vector containing for each variable a vector with the
    indices of the clauses in which the variable appears;
    needed for efficient incremental evaluation
"""
struct MAXSATInstance
    n::Int
    m::Int
    clauses::Vector{Vector{Int}}
    variable_usage::Vector{Vector{Int}}
end

"""
    MAXSATInstance(file_name)

Read a MAXSATInstance from the given file.
"""
function MAXSATInstance(file_name::String)
    local n::Int
    local m::Int
    clauses = Vector{Vector{Int}}()
    local variable_usage::Vector{Vector{Int}}
    open(file_name) do f
        for line in eachline(f)
            if line[1] == 'c'
                # ignore comments
                continue
            end
            fields = split(line)
            if length(fields) == 4 && fields[1] == "p" && fields[2] == "cnf"
                # parse header line with n and m
                n = parse(Int, fields[3])
                m = parse(Int, fields[4])
                variable_usage = [Vector{Int}() for _ in 1:n]
            elseif length(fields) >= 1
                # parse clause
                clause = [parse(Int, s) for s in fields[1:length(fields)-1]]
                push!(clauses, clause)
                for v in clause
                    push!(variable_usage[abs(v)], length(clauses))
                end
            end
        end
    end
    @assert n > 0 && m > 0
    @assert length(clauses) == m
    MAXSATInstance(n, m, clauses, variable_usage)
end


"""
    MAXSATSolution

A concrete solution type to solve the MAXSAT problem.
"""
mutable struct MAXSATSolution <: BoolVectorSolution
    inst::MAXSATInstance
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Bool}
    destroyed::Vector{Int}
end

"""
    MAXSATSolution(maxsat_instance)

Create a solution object for the given `MAXSATInstance`.
"""
MAXSATSolution(inst::MAXSATInstance) =
    MAXSATSolution(inst, -1, false, Vector{Bool}(undef, inst.n), [])

function Base.copy!(s1::MAXSATSolution, s2::MAXSATSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    copy!(s1.x, s2.x)
    copy!(s1.destroyed, s2.destroyed)
end

Base.copy(s::MAXSATSolution) =
    MAXSATSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x), copy(s.destroyed))

Base.show(io::IO, s::MAXSATSolution) =
    println(io, s.x)


"""
    calc_objective(maxsat_solution)

Count the number of satisfied clauses.
"""
function MHLib.calc_objective(s::MAXSATSolution)::Int
    satisfied = 0
    for clause in s.inst.clauses
        for v in clause
            if s.x[abs(v)] == (v > 0)
                satisfied += 1
                break
            end
        end
    end
    return satisfied
end


function MHLib.flip_variable!(s::MAXSATSolution, pos::Int)::Int
    obj(s)
    s.x[pos] = val = !s.x[pos]
    for clause in s.inst.variable_usage[pos]
        fulfilled_by_other = false
        val_fulfills_now = false
        for v in s.inst.clauses[clause]
            if v == 0 break end
            if abs(v) == pos
                val_fulfills_now = (v>0 ? val : !val)
            elseif s.x[abs(v)] == (v>0)
                fulfilled_by_other = true
                break  # clause fulfilled by other variable, no change
            end
        end
        if !fulfilled_by_other
            s.obj_val += (val_fulfills_now ? 1 : -1)
        end
    end
    return s.obj_val
end


"""
    destroy(maxsat_solution, par, result)

`MHMethod` that selects `3 * par` positions uniformly at random for removal.

Selected positions are stored with the solution in list `self.destroyed`.
"""
function MHLib.destroy!(sol::MAXSATSolution, par::Int, ::Result)
    x = sol.x
    num = get_number_to_destroy(sol, length(x); min_abs=3par, max_abs=3par)
    sol.destroyed = sample(1:length(x), num, replace=false)
    invalidate!(sol)
end


"""
    repair!(::MAXSATSolution, ::Nothing, result)

`MHMethod`that assigns new random values to all positions in `sol.destroyed`.
"""
function MHLib.repair!(sol::MAXSATSolution, ::Nothing, ::Result)
    @assert !isempty(sol.destroyed)
    x = sol.x
    for p in sol.destroyed
        x[p] = rand(0:1)
    end
    empty!(sol.destroyed)
    invalidate!(sol)
end

# -------------------------------------------------------------------------------

# We define a new problem-specific parameter alg, which is used in the demo program
const maxsat_settings_cfg = ArgParseSettings()
@add_arg_table! maxsat_settings_cfg begin
    "--alg"
        help = "Algorithm to apply (gvns, lns, weighted-lns, alns)"
        arg_type = String
        default = "alns"
end


function solve_maxsat(args=ARGS)
    println("MAXSAT Demo version $(git_version())\nARGS: ", args)
    args isa AbstractString && (args = split(args))

    # set some new default values for parameters and parse all relevant arguments
    settings_new_default_value!(MHLib.settings_cfg, "ifile", 
        "MHLibDemos/data/maxsat-adv1.cnf")
    settings_new_default_value!(MHLib.scheduler_settings_cfg, "mh_titer", 1000)
    parse_settings!([MHLib.scheduler_settings_cfg,
        MHLib.lns_settings_cfg, MHLib.alns_settings_cfg, maxsat_settings_cfg], args)
    println(get_settings_as_string())
    
    inst = MAXSATInstance(settings[:ifile])
    sol = MAXSATSolution(inst)
    println(sol)

    # Depending on the parameter `alg`, we create the respective algorithm object
    if settings[:alg] === "lns"
        alg = LNS(sol, [MHMethod("construct", construct!)],
            [MHMethod("de", destroy!, 1)],
            [MHMethod("re", repair!)];
            meths_compat = [true;;])
    elseif settings[:alg] === "weighted-lns"
        num_de = 5
        method_selector = WeightedRandomMethodSelector(num_re:-1:1, 1:1)
        alg = LNS(sol, [MHMethod("construct", construct!)],
            [MHMethod("de$i", destroy!, i) for i in 1:num_de],
            [MHMethod("re", repair!, nothing)];
            method_selector)
    elseif settings[:alg] === "alns"
        num_de = 5
        alg = ALNS(sol, [MHMethod("construct", construct!)],
            [MHMethod("de$i", destroy!, i) for i in 1:num_de],
            [MHMethod("re", repair!)])
    elseif settings[:alg] === "gvns"
        alg = GVNS(sol, [MHMethod("con", construct!)],
            [MHMethod("li1", local_improve!, 1)],
            [MHMethod("sh$i", shaking!, i) for i in 1:5])
    else
        error("Invalid parameter alg: $(settings[:alg])")
    end
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end

# To run from REPL, use MHLibDemos and call `solve_maxsat(<args>)` where `<args>` is 
# a single string or list of strings being passed as arguments for setting global 
# parameters, e.g. `solve_maxsat("--seed=1 --mh_titer=120")`.
# `@<filename>` may be used to read arguments from a configuration file <filename>
# Run with profiler:
# @profview solve_maxsat(args)