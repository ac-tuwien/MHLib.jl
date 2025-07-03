# `MHLib.jl` - A Toolbox for Metaheuristics and Hybrid Combinatorial Optimization Methods

![](https://github.com/ac-tuwien/MHLib.jl/actions/workflows/test_MHLib.yml/badge.svg)
![](https://github.com/ac-tuwien/MHLib.jl/actions/workflows/test_MHLibDemos.yml/badge.svg)
[![codecov](https://codecov.io/github/ac-tuwien/MHLib.jl/graph/badge.svg?token=E2SPCIZ5RN)](https://codecov.io/github/ac-tuwien/MHLib.jl)

_This project is still in development, any feedback is much appreciated!_

`MHLib.jl` is a collection of types and functions  in 
[Julia](https://julialang.org/) supporting
the effective implementation of metaheuristics and certain hybrid optimization approaches
for solving primarily combinatorial optimization problems.

![ ](mh.png)

Julia `MHLib.jl` emerged from the
[Python `mhlib`](https://github.com/ac-tuwien/pymhlib) and the older
[C++ `mhlib`](https://bitbucket.org/ads-tuwien/mhlib) to which it has certain similarities
but also many differences.

The main purpose of the library is to support rapid prototyping and teaching as well
as efficient implementations due to Julia's highly effective just-in-time-compilation.

`MHLib.jl` is developed primarily by the
[Algorithms and Complexity Group of TU Wien](https://www.ac.tuwien.ac.at),
Vienna, Austria, since 2020.

### Contributors:
- [Günther Raidl](https://www.ac.tuwien.ac.at/raidl) (primarily responsible)
- Nikolaus Frohner
- Thomas Jatschka
- Fabio Oberweger
- James Mulhern

## Installation

Major versions of `MHLib.jl` can be installed from the Julia REPL via

    ] add MHLib

Development versions are available at https://github.com/ac-tuwien/MHLib.jl and can be
installed via

    ] add https://github.com/ac-tuwien/MHLib.jl.git

## Recommended First-Time Usage

- Make a copy of the `MHLibDemos` subpackage, [see below](#MHLibDemos)
- rename it,
- choose one of the demo problems as a template,
- and adapt it to your own problem and solution approach.

## Major Components

The main file `src/MHLib.jl` provides the following types to represent candidate solutions and various functions for them:
- `Solution`:
    An abstract type that represents a candidate solution to an optimization problem.
- `VectorSolution`:
    An abstract solution encoded by a vector of some user-provided type.
- `BoolVectorSolution`:
    An abstract solution encoded by a boolean vector.
- `PermutationSolution`:
    An abstract solution representing permutations of a fixed number of elements.
- `SubsetVectorSolution`:
    A solution that is an arbitrarily large subset of a given set
    represented in vector form. The front part represents the selected
    elements, the back part optionally the unselected ones.

Moreover, the main file provides:
- `git_version()`:
    Function returning the abbreviated git version string of the current project. It is good practice to write this information also to a file wit log-information of an optimization run in order to be able to later possibly reproduce results with the same program version.

# Configuration/Parametrization

*Remark:* Earlier MHLib versions relied heavily on an extensible `settings` mechanism for various configuration parameters that corresponded to a global dictionary that is compiled from command line arguments provided when starting Julia. With version 0.3.0, `MHLib` replaced this mechanism by classical keyword arguments with default values in functions or constructors of structs. Partly, they are stored in separate `...Config` structs, partly directly in the respective structs representing certain metaheuristics.
In applications using `MHLib`, we also do not recommended to make use of such a global variable-based configuration parameter dictionary approach and/or the heavy use of Julia command line arguments (`ARGS`) anymore. 
Typically in Julia development, one better does not restart/call the whole Julia framework independently for each optimization/test run. It is usually much more efficient to use [Revise](https://github.com/timholy/Revise.jl) to automatically update the code in a continuously running Julia REPL session and to call individual optimization runs/tests from therein - directly with the keyword arguments to be used. 
Also when performing larger tests over many optimization runs in a batched fashion, it is often much simpler to do this directly in Julia instead of using a shell script that calls Julia many times. This partly also holds when using a compute cluster: better aggregate multiple runs - in particular when each run is relatively short - into fewer Julia processes to avoid/reduce Julia startup overhead.

Thus, for the various configuration parameters of the divers metaheuristics realized in `MHLib`, see the respective functions and structs doc-strings.

# Files (Modules):

- `Schedulers`, type `Scheduler`:
    A an abstract framework for single trajectory metaheuristics that rely on iteratively
    applying certain methods to a current solution.
    Modules like `GVNSs` and `LNSs` extend this type towards more specific metaheuristics.
- `GVNSs`, type `GVNSs`:
    A framework for local search, iterated local search, (general) variable neighborhood
    search, GRASP, etc.
- `LNSs`, type `LNS`:
    A framework for different variants of large neighborhood search (LNS).
    The selection of the destroy and repair methods is done in an extensible way by
    means of the abstract type `MethodSelector` and derived types in order to realize 
    different LNS variants.
- `ALNSs`, type `ALNS`:
    Adaptive large neighborhood search (ALNS). It is realized via `LNS` and `ALNSMethodSelector`.
- `OneMax`:
    A trivial test problem to which the above algorithms are applied in the unit tests in `test`.

## MHLibDemos

For demonstration purposes subdirectory [`MHLibDemos`](MHLibDemos/README.md) provides a Julia package (not separately registered at JuliaHub), with basic implementations for the following classical combinatorial optimization problems, to which some of MHLib's metaheuristics are applied:

- `GraphColoring`: graph coloring problem based on `VectorSolution`
- `MAXSAT`: maximum satisfiability problem based on `BinaryVectorSolution`
- `TSP`: traveling salesperson problem based on `PermutationSolution`
- `MKP`: multi-constrained knapsack problem based on `SubsetVectorSolution`
- `MISP`: maximum independent set problem based on `SubsetVectorSolution`

It is recommended to take the `MHLibDemos` package with one of the demos as template for 
developing MHLib-based metaheuristics for your own problem. Remember to activate this package's own Environment in order to run the demos.

Further smaller usage examples can also be found in the test directory of the MHLibDemos package.

## Parameter Tuning

Subdirectory `Tuning` contains examples on how [`irace`](https://github.com/MLopez-Ibanez/irace) can specifically be used for tuning
algorithms implemented in Julia. See [Tuning/README.md](Tuning/README.md) for details.

## News

See [CHANGELOG.md](CHANGELOG.md)

