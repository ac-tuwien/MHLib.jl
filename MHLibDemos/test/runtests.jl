# runtests.jl
#
# Unit tests for MHLibDemos.
#
# Always performed in the test directory within the test environment.

using TestItemRunner

using MHLibDemos

@run_package_tests;
