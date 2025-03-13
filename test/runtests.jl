# Unit tests

# always performed in the test directory within the test environment

using TestItemRunner

@run_package_tests filter=ti->(!endswith(ti.filename, "MHLibDemos/test/tests.jl"));

