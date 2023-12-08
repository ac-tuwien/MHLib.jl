#!/usr/bin/env julia
"""
    demo_tsp.jl

Standalone demo program for solving the OneMax Problem.
"""

# switch to MHLibDemos directory and activate its environment
cd(@__DIR__()*"/..")
using Pkg; Pkg.activate(".") 

using MHLibDemos

# Command line arguments are parsed and used to set global parameters
# use `@<filename>` to read parameters from configuration file `<filename>`
# alternatively, they may also be provided here in a string
solve_onemax("--onemax_n=90")
