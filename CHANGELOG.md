
# Changelog of MHLib.jl

Major changes in releases:

## Upcoming

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
