# Unit tests

# always performed in the test directory within the test environment

using TestItemRunner

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using MHLibDemos

@run_package_tests;
