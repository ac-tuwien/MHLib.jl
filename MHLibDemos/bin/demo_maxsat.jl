#!/usr/bin/env julia
"""
    demo_maxsat.jl

Standalone demo program for solving the MAXSAT problem.
"""

# switch to MHLibDemos directory and activate its environment
cd(@__DIR__()*"/..")
using Pkg; Pkg.activate(".") 

using MHLibDemos

# Command line arguments are parsed and used to set global parameters
# use `@<filename>` to read parameters from configuration file `<filename>`
# alternatively, they may also be provided here as a string
solve_maxsat("--seed=8 --mh_ttime=3")
