# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`MHLib.jl` is a Julia toolbox of types and functions supporting rapid, efficient
implementation of metaheuristics and hybrid combinatorial optimization methods
(local search, GVNS/VNS/GRASP/ILS, LNS, ALNS). It is developed by the Algorithms and
Complexity Group, TU Wien. `MHLibDemos` (in the `MHLibDemos/` subdirectory) is a
separate, non-registered demo package with reference implementations of classical
combinatorial problems built on top of `MHLib`.

The repo is a Julia "workspace" (root `Project.toml` has `[workspace] projects =
["docs", "MHLibDemos"]`, Julia ≥1.12) rather than a monorepo package manager setup —
`docs` and `MHLibDemos` are separate projects that share the resolved dependency graph
with the root package.

## Commands

Run from the repo root unless noted.

- Instantiate/build the main package: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Run the `MHLib` test suite: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Run the `MHLibDemos` test suite (from `MHLibDemos/`, with `MHLib` dev'd in):
  `julia --project=MHLibDemos -e 'using Pkg; Pkg.develop(path="."); Pkg.test()'`
- Run a single test item: tests use `TestItems.jl`/`TestItemRunner.jl` — open
  `test/tests.jl` (or `MHLibDemos/test/tests.jl`) in the Julia VS Code extension and
  run/debug an individual `@testitem` block, or filter in `runtests.jl`'s
  `@run_package_tests filter=...` call.
- Build docs locally: `julia --project=docs docs/make.jl`
- Interactive development: prefer a long-running REPL with
  [Revise.jl](https://github.com/timholy/Revise.jl) over repeated `julia` invocations —
  this is the project's documented recommended workflow (see README "Configuration"
  section) since MHLib does not use a global settings/CLI-args mechanism.

CI (`.github/workflows/test_MHLib.yml`, `test_MHLibDemos.yml`) runs each package's
tests under `--project=monorepo` after `pkg"dev ."` (and `pkg"dev ./MHLibDemos ."` for
demos), against the current Julia release on Ubuntu, and uploads coverage to Codecov.

## Architecture

### Solution type hierarchy (`src/MHLib.jl`)

Everything centers on the abstract type `Solution`, with a chain of increasingly
specific abstract subtypes each adding required fields/behavior that concrete problem
solutions must implement:

- `Solution` — requires `obj_val`, `obj_val_valid::Bool`, `calc_objective`,
  `initialize!`, `to_maximize`, `copy!`, `copy`. `obj(s)` is the standard accessor:
  it lazily (re-)computes and caches `obj_val` when `obj_val_valid` is false;
  `invalidate!(s)` must be called after any mutation so `obj` recomputes.
- `VectorSolution{T}` — adds `x::AbstractVector{T}` and `destroyed` (positions
  touched by destroy+repair operators, e.g. in LNS).
- `BoolVectorSolution <: VectorSolution{Bool}` — adds bit-flip neighborhood search
  (`k_random_flips!`, `k_flip_neighborhood_search!`, `flip_variable!` — the latter is
  the extension point for incremental objective evaluation on a flip).
- `PermutationSolution` (`src/PermutationSolutions.jl`) and `SubsetVectorSolution`
  (`src/SubsetVectorSolutions.jl`, front part = selected elements, back part =
  optionally the unselected ones) are further concrete-ish abstract types with their
  own neighborhood/move operators.

Comparisons (`is_better`, `is_worse`, `is_better_obj`, `is_worse_obj`) all dispatch
through `to_maximize(::Type)`, so a concrete solution type only needs to override
`to_maximize` to flip between minimization and maximization — comparison logic itself
is never duplicated per problem.

### Metaheuristic frameworks (layered on `Scheduler`)

`Schedulers.jl` defines the common engine: an `MHMethod` wraps a `(scheduler, sol,
par) -> Result` function with a name and a parameter; `Scheduler` iterates methods
against a solution, tracks the incumbent, logs iterations, and checks termination
criteria (`titer`, `ttime`, `tciter`, `tctime`, `tobj`, all in `SchedulerConfig`).
Concrete algorithm types wrap a `Scheduler` and organize methods into named phases:

- `GVNSs.jl` → `GVNS`: construction, local-improvement, and shaking method lists
  (implements local search / VND / GVNS / GRASP / ILS depending on which lists are
  populated).
- `LNSs.jl` → `LNS`: construction, destroy, and repair method lists; method choice at
  each iteration is delegated to a pluggable `MethodSelector` (e.g.
  `UniformRandomMethodSelector`) plus an optional `meths_compat` matrix restricting
  which destroy/repair method pairs may be combined; includes simulated-annealing-like
  acceptance via `init_temp`/`temp_dec_factor`.
- `ALNSs.jl` → `ALNS`: built on `LNS` with `ALNSMethodSelector`, adaptively reweighting
  destroy/repair methods using a segment-based scoring scheme (`segment_size`, `gamma`,
  `sigma1/2/3`).

All algorithm keyword parameters are plain keyword arguments / `...Config` structs with
defaults — there is deliberately **no** global settings dictionary or CLI-args (`ARGS`)
based configuration mechanism (an earlier design used one; see the README
"Configuration/Parametrization" section). New code should follow the same convention.

`Log.jl` provides the iteration/statistics logging used by `Scheduler` (custom
`LogLevel`s `IterLevel`/`StatsLevel`, overridable via `get_logger`).

`OneMax.jl` implements a trivial `OneMaxSolution` (bit-vector) problem used purely as
the target of the unit tests.

### Tests are embedded in source, then included by both package and test runner

Unusually, `test/tests.jl` (and `MHLibDemos/test/tests.jl`) is `include`d directly at
the bottom of `src/MHLib.jl` (`MHLibDemos/src/MHLibDemos.jl`) via a relative path,
*and* separately picked up by `test/runtests.jl` via `TestItemRunner`'s
`@run_package_tests`. `@testitem`/`@testsnippet` blocks (from `TestItems.jl`) are
inert unless run through `TestItemRunner`/the VS Code Julia test extension — including
the file from the module body just makes the test code get syntax-checked as part of
normal compilation. When adding tests, put them in `test/tests.jl` as `@testitem`
blocks, not in `test/runtests.jl`, and don't assume `include`d test code executes at
module load time.

### MHLibDemos

Each demo problem (`GraphColoring`, `MAXSAT`, `MISP`, `MKP`, `TSP` — files directly in
`MHLibDemos/src/`) is a self-contained problem instance + solution type built on one of
MHLib's abstract solution types (see README "Major Components" for the mapping), plus
construction/local-improvement/destroy/repair method functions to plug into `GVNS`,
`LNS`, or `ALNS`. `MHLibDemos.jl` also provides the shared `create_or_read_simple_graph`
helper (reads a DIMACS-ish graph file format, or generates a random `SimpleGraph` for
names like `gnm-n-m`). New problem implementations should be modeled on these demos;
the README explicitly recommends copying `MHLibDemos`, renaming it, and adapting one
demo as a starting template for a new project (rather than depending on `MHLibDemos`
directly).

### Hyperparameter tuning (`tuning/`)

Examples of using [irace](https://github.com/MLopez-Ibanez/irace) to tune Julia-side
metaheuristic parameters: `tuning/irace-classical` spawns a fresh `julia main.jl`
process per evaluation; `tuning/irace-with-julia-server` instead keeps a persistent
`julia-server.jl` process listening on a socket to avoid per-evaluation Julia startup
cost — prefer the latter pattern when the per-run algorithm cost is small relative to
Julia startup time.
