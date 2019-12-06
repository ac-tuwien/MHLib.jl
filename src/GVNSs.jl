#=
A general variable neighborhood search class which can also be used for plain local search,
VND, GRASP, IG etc.

It extends the more general scheduler module/class by distinguishing between construction
heuristics, local improvement methods and shaking methods.
=#

module GVNSs

using MHLib
using MHLib.Schedulers


"""GVNS

A general variable neighborhood search (GVNS).

Attributes
    - scheduler: Scheduler object
    - meths_ch: list of construction heuristic methods
    - meths_li: list of local improvement methods
    - meths_sh: list of shaking methods
"""
mutable struct GVNS
    scheduler::Scheduler
    meths_ch::Vector{MHMethod}
    meths_li::Vector{MHMethod}
    meths_sh::Vector{MHMethod}
end

"""GVNS(sol, meths_ch, meths_li, mehts_sh, consider_initial_sol=false)

Parameters
    - sol: solution to be improved
    - meths_ch: list of construction heuristic methods
    - meths_li: list of local improvement methods
    - meths_sh: list of shaking methods
    - own_settings: optional dictionary with specific settings
    - consider_initial_sol: if true consider sol as valid solution that should be
        improved upon; otherwise
    sol is considered just a possibly uninitialized of invalid solution template
"""
function GVNS(sol::Solution, meths_ch::Vector{MHMethod}, meths_li::Vector{MHMethod},
    meths_sh::Vector{MHMethod}; consider_initial_sol::Bool=false)
    # TODO own_settings
    GVNS(Scheduler(sol, [meths_ch; meths_li; meths_sh]),
        consider_initial_sol=consider_initial_sol)
end

#=
    def vnd(self, sol: Solution) -> bool:
        """Perform variable neighborhood descent (VND) on given solution.

        :returns: true if a global termination condition is fulfilled, else False.
        """
        sol2 = sol.copy()
        while True:
            for m in self.next_method(self.meths_li):
                res = self.perform_method(m, sol2)
                if sol2.is_better(sol):
                    sol.copy_from(sol2)
                    if res.terminate:
                        return True
                    break
                else:
                    if res.terminate:
                        return True
                    if res.changed:
                        sol2.copy_from(sol)
            else:  # local optimum reached
                return False

    def gvns(self, sol: Solution):
        """Perform general variable neighborhood search (GVNS) to given solution."""
        sol2 = sol.copy()
        if self.vnd(sol2) or not self.meths_sh:
            return
        use_vnd = bool(self.meths_li)
        while True:
            for m in self.next_method(self.meths_sh, repeat=True):
                t_start = time.process_time()
                res = self.perform_method(m, sol2, delayed_success=use_vnd)
                terminate = res.terminate
                if not terminate and use_vnd:
                    terminate = self.vnd(sol2)
                self.delayed_success_update(m, sol.obj(), t_start, sol2)
                if sol2.is_better(sol):
                    sol.copy_from(sol2)
                    if terminate or res.terminate:
                        return
                    break
                else:
                    if terminate or res.terminate:
                        return
                    sol2.copy_from(sol)
            else:
                break

    def run(self) -> None:
        """Actually performs the construction heuristics followed by the GVNS."""
        sol = self.incumbent.copy()
        assert self.incumbent_valid or self.meths_ch
        self.perform_sequentially(sol, self.meths_ch)
        self.gvns(sol)

=#

end  # module
