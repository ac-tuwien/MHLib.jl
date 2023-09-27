"""
    MKP

Demo problem: multi-dimensional knapsack problem (MKP).

Given are a set of n items, m resources, and a capacity for each resource.
Each item has a price and requires from each resource a certain amount.
Find a subset of the items with maximum total price that does not exceed the resources'
capacities.
"""

using ArgParse
using MHLib

export MKPInstance, MKPSolution, solve_mkp

"""
    MKPInstance

Instance oof a multidimensional knapsack problem.

- `n`: number of elements
- `m`: number of resources
- `p: vector of prizes of elements
- `r`: resource consumption values of each each element
- `b`: capacities of resources
- `r_min`: minimum resource consumption value of any element
- `obj_opt`: optimal solution value (if known)
"""
struct MKPInstance
    n::Int
    m::Int
    p::Vector{Int}
    r::Array{Int, 2}
    b::Vector{Int}
    r_min::Int
    obj_opt::Float64
end

"""
    MKPInstance(file_name)

Read MKP instance from file.
"""
function MKPInstance(file_name::String)
    local n::Int
    local m::Int
    local p::Vector{Int}
    local r::Array{Int,2}
    local b::Vector{Int}
    local r_min::Int
    local obj_opt::Float64
    all_values = Vector{Int}()

    open(file_name) do f
        for line in eachline(f)
            for word in split(line)
                push!(all_values, parse(Int,word))
            end
        end
        n = all_values[1]
        m = all_values[2]
        if length(all_values) != 3+n+m*n+m
            error("Invalid number of values in MKP instance file $(file_name)")
        end
        obj_opt = all_values[3]
        p = Vector{Int}(all_values[4:4+n-1])
        r = reshape(Vector{Int}(all_values[4+n:4+n+m*n-1]),(m,n))
        b = Vector{Int}(all_values[4+n+m*n:4+n+m*n+m-1])
        r_min = min(minimum(r),1)
    end
    MKPInstance(n, m, p, r, b, r_min, obj_opt)
end

"""
    MKPSolution

Solution to an MKP Instance represented as a SubsetVectorSolution.

Attributes in addition to those needed by `SubsetVectorSolution`
- `y`: consumed amounts of the resources
"""
mutable struct MKPSolution <: SubsetVectorSolution{Int}
    inst::MKPInstance
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Int}
    y::Vector{Int}
    all_elements::Set{Int}
    sel::Int
end

MKPSolution(inst::MKPInstance) =
    MKPSolution(inst, -1, false, collect(1:inst.n), zeros(inst.m), Set{Int}(1:inst.n), 0)

function Base.copy!(s1::MKPSolution, s2::MKPSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    copy!(s1.x, s2.x)
    copy!(s1.y, s2.y)
    s1.all_elements = Set(s2.all_elements)
    s1.sel = s2.sel
end

Base.copy(s::MKPSolution) =
    MKPSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x), copy(s.y),
        copy(s.all_elements), s.sel)

Base.show(io::IO, s::MKPSolution) =
    println(io, s.x)

MHLib.calc_objective(s::MKPSolution) =
    s.sel > 0 ? sum(s.inst.p[s.x[1:s.sel]]) : 0


"""
    calc_y!(mkp_solution)

Calculate consumed amounts of resources for current solution.
"""
function calc_y!(s::MKPSolution)
    if s.sel > 0
        s.y = vec(sum(s.inst.r[begin:end, s.x[1:s.sel]], dims=2))
    end
    return 0
end

function MHLib.check(s::MKPSolution; kwargs...)
    invoke(check, Tuple{SubsetVectorSolution}, s; kwargs...)
    y_old = s.y
    calc_y!(s)
    if any(y_old .!= s.y)
        error("Solution had invalid y values: $(s.y) $(y_old)")
    end
    if any(s.y .> s.inst.b)
        error("Solution exceeds capacity limits: $(self.y) $(s.inst.b)")
    end
end

function clear!(s::MKPSolution)
    fill!(s.y, 0)
    invoke(clear, Tuple{SubsetVectorSolution}, s)
end

"""
    construct!(mkp_solution, par, result)

Construct new solution by random initialization.
"""
function MHLib.Schedulers.construct!(s::MKPSolution, par::Int, result::Result)
    initialize!(s)
end

"""
    local_improve!(mkp_solution, par, result)

Perform two-exchange local search followed by random fill.
"""
function MHLib.Schedulers.local_improve!(s::MKPSolution, par::Int, result::Result)
    if !two_exchange_random_fill_neighborhood_search!(s, false)
        result.changed = false
    end
end

"""
    shaking!(mkp_solution, par, result)

Perform shaking by removing `par` randomly selected elements followed ba a random fill.
"""
function MHLib.Schedulers.shaking!(s::MKPSolution, par::Int, result::Result)
    remove_some!(s, par)
    fillup!(s)
end

"""
    may_be_extendible(mkp_solution)

Quick check if the solution may be extended by adding further elements.
"""
MHLib.SubsetVectorSolutions.may_be_extendible(s::MKPSolution) =
    all((s.y .+ s.inst.r_min) .<= s.inst.b) && s.sel < length(s.x)

function MHLib.SubsetVectorSolutions.element_removed_delta_eval!(s::MKPSolution; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    elem = s.x[s.sel+1]
    s.y .-= s.inst.r[:, elem]
    if update_obj_val
        s.obj_val -= s.inst.p[elem]
    end
    return true
end

function MHLib.SubsetVectorSolutions.element_added_delta_eval!(s::MKPSolution; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    elem = s.x[s.sel]
    y_new = s.y .+ s.inst.r[:, elem]
    feasible = all(y_new .<= s.inst.b)
    if allow_infeasible || feasible
        # accept
        s.y = y_new
        if update_obj_val
            s.obj_val += s.inst.p[elem]
        end
        return feasible
    end
    # revert
    s.sel -= 1
    return false
end

# -------------------------------------------------------------------------------

function solve_mkp(args=ARGS)
    println("MKP Demo version $(git_version())\nARGS: ", args)

    # set some new default values for parameters and parse all relevant arguments
    settings_new_default_value!(MHLib.settings_cfg, "ifile", "data/mknapcb5-01.txt")
    settings_new_default_value!(MHLib.Schedulers.settings_cfg, "mh_titer", 5000)
    parse_settings!([MHLib.Schedulers.settings_cfg], args)
    println(get_settings_as_string())

    inst = MKPInstance(settings[:ifile])
    sol = MKPSolution(inst)
    # initialize!(sol)
    # check(sol)
    # println(sol)

    # we apply a variable neighborhood search:
    alg = GVNS(sol, [MHMethod("con", construct!, 0)],
        [MHMethod("li1", local_improve!, 1)],
        [MHMethod("sh1", shaking!, 1), MHMethod("sh2", shaking!, 2),
            MHMethod("sh3", shaking!, 3)], 
        consider_initial_sol = true)
    run!(alg)
    method_statistics(alg.scheduler)
    main_results(alg.scheduler)
    check(sol)
    return sol
end

# To run from REPL, use MHLibDemos and call `solve_mkp(<args>)` where `<args>` is 
# a list of strings being passed as arguments for setting global parameters.
# `@<filename>` may be used to read arguments from a configuration file <filename>

# Run with profiler:
# @profview solve_mkp(args)