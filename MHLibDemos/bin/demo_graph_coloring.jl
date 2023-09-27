#!/usr/bin/env julia
"""
    demo_graph_coloring.jl

Standalone demo program for solving the Graph Coloring Problem.
"""

# switch to MHLibDemos directory and activate its environment
cd(@__DIR__()*"/..")
using Pkg; Pkg.activate(".") 

using MHLibDemos

# Command line arguments are parsed and used to set global parameters
# use `@<filename>` to read parameters from configuration file `<filename>`
# alternatively, they may also be provided here to the call as vector of strings
solve_graph_coloring()
