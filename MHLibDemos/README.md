# `MHLibDemos.jl` - Demo Applications for MHLib

[![Build Status](https://github.com/ac-tuwien/MHLib.jl/workflows/CI/badge.svg)](https://github.com/ac-tuwien/MHLib.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov.io](http://codecov.io/github/ac-tuwien/MHLib.jl/coverage.svg?branch=master)](http://codecov.io/github/ac-tuwien/MHLib.jl?branch=master)

_This project is still in early development, any feedback is much appreciated!_

[`MHLib.jl`](https://github.com/ac-tuwien/MHLib.jl) is a collection of modules, types, and functions  in Julia 1.8+ supporting
the effective implementation of metaheuristics and certain hybrid optimization approaches
for solving primarily  combinatorial optimization problems.

The current package `MHLibDemos.jl` provides demo applications for `MHLib.jl` that can be used as templates for your own applications.

`MHLib.jl` and `MHLibDemos.jl` is developed primarily by the
[Algorithms and Complexity Group of TU Wien](https://www.ac.tuwien.ac.at),
Vienna, Austria, since 2020.

### Contributors:

- [Günther Raidl](https://www.ac.tuwien.ac.at/raidl) (primarily responsible)
- Nikolaus Frohner
- Thomas Jatschka
- Fabio Oberweger

## Installation

Major versions of `MHLib.jl` can be installed from the Julia REPL via

    ] add MHLib

The associated package `MHLibDemos.jl`, which provides diverse demos for solving classical combinatorial optimization problems with `MHLib.jl`, can be installed via

    ] add MHLibDemos

Development versions of both packages are available at https://github.com/ac-tuwien/MHLib.jl and can be
installed via

    ] add https://github.com/ac-tuwien/MHLib.jl.git

and

    ] add https://github.com/ac-tuwien/MHLib.jl.git#master:MHLibDemos


## Further information

See the README.md file of the main package [MHLib.jl](https://github.com/ac-tuwien/MHLib.jl).