## `MHLib.jl` - A Toolbox for Metaheuristics and Hybrid Optimization Methods

Build status:
[![Build Status](https://travis-ci.com/ac-tuwien/MHLib.jl.svg?branch=master)](https://travis-ci.com/ac-tuwien/MHLib.jl)

_This project is still in early development, any feedback is much appreciated!_

`MHLib.jl` is a collection of modules, types, and functions  in Julia 1.5+ supporting
the effective implementation of metaheuristics and certain hybrid optimization approaches
for solving primarily  combinatorial optimization problems.

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

#### Contributors:
- [Günther Raidl](https://www.ac.tuwien.ac.at/raidl) (primarily responsible)
- Thomas Jatschka
- Fabio Oberweger

### Installation

Major versions of `pymhlib` can be installed from the Julia REPL via

    ] add MHLib

Development versions are available at https://github.com/ac-tuwien/MHLib.jl and can be
installed via

    ] add https://github.com/ac-tuwien/MHLib.jl.git

### Major Components

Note that `MHLib.jl` is still far behind the capabilities of the Python `pymhlib`.

The main module provides the following types for candidate solutions and various
functions for them:
- `Solution`:
    An abstract type that represents a candidate solution to an optimization problem.
- `VectorSolution`:
    An abstract solution encoded by a vector of some user-provided type.
- `BoolVectorSolution`:
    An abstract solution encoded by a boolean vector.
_ `SubsetVectorSolution`:
    A solution that is an arbitrary cardinality subset of a given set
    of integers represented in vector form. The front part represents the selected
    elements, the back part the unselected ones.

Moreover, the main module provides:
- `git_version()`:
    Function returning the abbreviated git version string of the current project.
- `settings`:
    Global settings that can be defined independently per module in a distributed
    way, while values for these parameters can be provided as program arguments or in
    configuration files. Most `pyhmlib` modules rely on this mechanism for their external
    parameters.

Further modules:

- `Schedulers`, type `Scheduler`:
    A an abstract framework for single trajectory metaheuristics that rely on iteratively
    applying certain methods to a current solution.
    Modules like `GVNSs` and `ALNSs` extend this type towards
    more specific metaheuristics.
- `GVNSs`, type `GVNSs`:
    A framework for local search, iterated local search, (general) variable neighborhood
    search, GRASP, etc.
- `ALNSs`, type `ALNS`:
    A framework for adaptive large neighborhood search (ALNS).


#### Demos

For demonstration purposes, simple metaheuristic approaches are provided in the `demos`
subdirectory for the following well-known combinatorial optimization problems.
They can be statet by respective scripts in the `bin` folder.

It is recommended to take such a demo as template or solving your own problem.

- `OneMax`: basic test problem in which the goal is to set all digits in a binary
    string to `true`
- `MAXSAT`: maximum satisfiability problem based on `BinaryVectorSolution`
- `MKP`: multi-constrained knapsack problem based on `SubsetVectorSolution`


### Changelog

Major changes over major releases:

#### Version 0.1.1
- `SubsetVectorSolution` and `MKP` added

#### Version 0.1.0
- Initial version
