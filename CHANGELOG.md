
# Changelog of MHLib.jl

Major changes in releases:

## Version 0.3.0
- Usage of `settings` completely removed in MHLib and replaced by normal function parameters; although settings.jl remains, its usage is mostly discouraged, as this mechanism corresponds to global variables and comes with respective problems; see also the new [remark in README.md](README.md) concerning the recommendation of using `Revise.jl` for development and testing instead of frequently restarting Julia to execute an optimization run.
- Automated seeding of random number generation when parsing `settings` also removed.
- `MHLibDemos` adapted accordingly; new main functions `solve_*()` at bottom of problem-specific files, independent scripts removed
- Package dependencies updated
- Tuning: SMAC3 example removed, irace examples simplified

## Version 0.2.1
- Added templates for tuning with irace, which is now the preferred tuning tool
- `clear!` renamed to `Base.empty!`
- Unit tests use now `TestItems.jl`, which has support integrated in VSCode
- All submodules within MHLib removed
- MaxSat demo added directly i to be able to perform basic tests of all algorithms
- Other tests for different problems moved to `MHLibDemos`
- Docstrings improved
- New setting `mh_log` for turning of logging completely

## Version 0.1.15
- `Tuning`renamed to `tuning`and small fixes in it, updates for Julia 1.11

## Version 0.1.14
- `Results` from `Scheduler` is extended with field `is_local_optimum` to be able to 
    indicate if the solution is a local optimum in respect to the current method and it
    therefore does not make sense to apply this method again; used within VND.

## Version 0.1.13
- The logging possibilities were extended by the new `Logging` 
- In the Python code of `Tuning`, package `julia` is replaced by the newer `juliacall`

## Version 0.1.12
- Fix in `GVNS` and `LNS`: The termination condition has not been considered directly after
    performing the initial construction method(s).
    
## Version 0.1.11
- Fix in `GVNS`: Empty local search method list led to an infinite loop 
    [issue #5](/../../issues/5)

## Version 0.1.10
- subdirectory `Tuning` added with examples how to use SMAC3 for tuning parameters 
    algorithms implemented in Julia
- the LNS variants now also accept new equally good solutions as new incumbents
    with which to continue

## Version 0.1.9
- bug fix in `MISPSolution.clear!` and `MKPSolution.clear!`, MKP.jl polished

## Version 0.1.8
- all demo applications refined and moved into an own subpackage `MHLibDemos`
- all symbols of submodules are now also re-exported by the main `MHLib` module
- LNS/ALNS refactored, `MethodSelector` introduced to generalize LNS
- all structures adapted to use type parameters instead of abstract types for elements

## Version 0.1.6, 0.1.7
- used packages updated for Julia 1.9

## Version 0.1.5
- switched from outdated `LightGraphs.jl` to `Graphs.jl`
- minor documentation and code polishing

## Version 0.1.4
- `k_random_flips updated to do selection without replacement, i.e., indeed flip 
    exactly `k` bits
- hack for seeing symbols in `testrun.jl` in VSCode; separate `CHANGELOG.md`

## Version 0.1.3
- bug fix in 2-opt neighborhood search of `PermutationSolution`

## Version 0.1.2
- `PermutationSolution` and `SubsetVectorSolution` put into own module
- some cleaning in MKP
- most import statements replaced by qualified function definitions

## Version 0.1.1
- `GraphColoring`, `PermutationSolution`, `SubsetVectorSolution`, `TSP`, `MKP`, 
    and `MISP` demos added
- statistics output of method applications in scheduler
- polishing of docstrings and code

## Version 0.1.0
- Initial version
