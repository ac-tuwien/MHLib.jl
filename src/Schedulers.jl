#=Schedulers

General scheduler for realizing (G)VNS, GRASP, IG and similar metaheuristics.

The module is intended for metaheuristics in which a set of methods
(or several of them) are in some way repeatedly applied to candidate solutions.
=#
module Schedulers

using ArgParse
using Printf
using MHLib
# import MHLib: @add_arg_table, settings, settings_cfg, Solution, obj

export Result, MHMethod, MHMethodStatistics, Scheduler, perform_method!,
    next_method, update_incumbent!, check_termination, perform_sequentially!,
    main_results

@add_arg_table settings_cfg begin
    "--mh_titer"
        help = "maximum number of iterations (<0: turned off)"
        arg_type = Int
        default = 100
end
#= TODO
parser = get_settings_parser()
parser.add_argument("--mh_titer", type=int, default=100, help='maximum number of iterations (<0: turned off)')
parser.add_argument("--mh_tciter", type=int, default=-1,
                    help='maximum number of iterations without improvement (<0: turned off)')
parser.add_argument("--mh_ttime", type=int, default=-1, help='time limit [s] (<0: turned off)')
parser.add_argument("--mh_tctime", type=int, default=-1, help='maximum time [s] without improvement (<0: turned off)')
parser.add_argument("--mh_tobj", type=float, default=-1,
                    help='objective value at which should be terminated when reached (<0: turned off)')
add_bool_arg(parser, "mh_lnewinc", default=true, help='write iteration log if new incumbent solution')
parser.add_argument("--mh_lfreq", type=int, default=0,
                    help='frequency of writing iteration logs (0: none, >0: number of iterations, '
                         '-1: iteration 1,2,5,10,20,...')
add_bool_arg(parser, "mh_checkit", default=false, help='call check() for each solution after each method application')
parser.add_argument("--mh_workers", type=int, default=4, help='number of worker processes when using multiprocessing')
=#


"""
Data in conjunction with a method application's result.

Attributes
- `changed`: if false, the solution has not been changed by the method application
- `terminate`: if true, a termination condition has been fulfilled
- `log_info`: customized log info
"""
mutable struct Result
    changed::Bool
    terminate::Bool
    log_info::String
end

Result() = Result(true, false, "")


"""
A method to be applied to a solution by the scheduler.

Attributes
- 'name`: name of the method; must be unique over all used methods
- `method`: a function called for a Solution object
- `par`: a parameter provided when calling the method
"""
struct MHMethod
    name::String
    func::Function
    par::Int
end


"""
Struct that collects data on the applications of a `MHMethod`.

Attributes
- `applications`: number of applications of this method
- `netto_time`: accumulated time of all applications of this method without further costs
    (e.g., VND)
- `successes`: number of applications in which an improved solution was found
- `obj_gain`: sum of gains in the objective values over all successful applications
- `brutto_time`: accumulated time of all applications of this method including further
    costs (e.g., VND)
"""
mutable struct MHMethodStatistics
    applications::Int
    netto_time::Float64
    successes::Int
    obj_gain::Float64
    brutto_time::Float64
end

MHMethodStatistics() = MHMethodStatistics(0,0.0,0,0.0,0.0)


"""
Struct for metaheuristics that work by iteratively applying certain methods/operations.

Attributes
- `incumbent`: incumbent solution, i.e., initial solution and always best solution so far
    encountered
- `incumbent_valid`: true if incumbent is a valid solution to be considered
- `incumbent_iteration`: iteration in which incumbent was found
- `incumbent_time`: time at which incumbent was found
- `methods`: vector of all MHMethods
- `method_stats`: dict of MHMethodStatistics for each MHMethod
- `iteration`: overall number of method applications
- `time_start`: starting time of algorithm
- `run_time`: overall runtime (set when terminating)
- `own_settings`: own settings object with possibly individualized parameter values
"""
mutable struct Scheduler
    incumbent::Solution
    incumbent_valid::Bool
    incumbent_iteration::Int
    incumbent_time::Float64
    methods::Vector{MHMethod}
    method_stats::Dict{String,MHMethodStatistics}
    iteration::Int
    time_start::Float64
    run_time::Float64
    # logger = logging.getLogger("pymhlib") TODO
    # iter_logger = logging.getLogger("pymhlib_iter")
end


"""
    Scheduler(solution, methods, consider_initial_sol)

Create a `MHMethod` scheduler.

Create a Scheduler for the given solution with the given methods provides as
`Vector{MHMethod}`. If `consider_initial_sol`, consider the given solution as
valid initial solution; otherwise it is assumed to be uninitialized.

"""
function Scheduler(sol::Solution, methods::Vector{MHMethod}, consider_initial_sol=false)
    method_stats = Dict([(m.name, MHMethodStatistics()) for m in methods])
    return Scheduler(sol, consider_initial_sol, 0, 0.0, methods, method_stats, 0,
        time(), 0.0)
    # TODO self.log_iteration_header()
    # if self.incumbent_valid:
    #    self.log_iteration('-', float('NaN'), sol, true, true, None)
    #    self.own_settings = OwnSettings(own_settings) if own_settings else settings
end


"""
    update_incumbent!(scheduler, solution, current_time)

If the given solution is better than the incumbent (or we do not have an incumbent yet)
update it.
"""
function update_incumbent!(s::Scheduler, sol::Solution, current_time::Float64)
    if !s.incumbent_valid || is_better(sol, s.incumbent)
        copy!(s.incumbent, sol)
        s.incumbent_iteration = s.iteration
        s.incumbent_time = current_time
        s.incumbent_valid = true
        return true
    end
    false
end


"""
    next_method(meths; randomize=false, repeat=false)

Generator for obtaining a next method from a given vector of methods, iterating through
all of them. `randomize`: random order, otherwise consider given order;
`repeat`: repeat infinitely, otherwise just do one pass.
"""
function next_method(meths::Vector{MHMethod}; randomize::Bool=false, repeat::Bool=false)
    if randomize
        meths = meths.copy()
    end
    function gen_methods(channel::Channel)
        while true
            # if randomize TODO
            #     random.shuffle(meths)
            # end
            for method in meths
                put!(channel, method)
            end
            if !repeat
                break
            end
        end
    end
    Channel(gen_methods)
end


"""
    perform_method!(scheduler, method, solution; delayed_success=false)::Result

Perform method on given solution and return Results object.

Also updates incumbent, iteration and the method's statistics in method_stats.
Furthermore checks the termination condition and eventually sets terminate in the
returned Results object. If `delayed_success`, the success is not immediately determined
and the statistics updated accordingly but at some later call of `delayed_success_update`.
"""
function perform_method!(s::Scheduler, method::MHMethod, sol::Solution;
    delayed_success=false)::Result
    res = Result()
    obj_old = obj(sol)
    t_start = time()
    method.func(sol, method.par, res)
    t_end = time()
    # if __debug__ and self.own_settings.mh_checkit: TODO
    #     sol.check()
    ms = s.method_stats[method.name]
    ms.applications += 1
    ms.netto_time += t_end - t_start
    obj_new = obj(sol)
    if !delayed_success
        ms.brutto_time += t_end - t_start
        if is_better_obj(sol, obj(sol), obj_old)
            ms.successes += 1
            ms.obj_gain += obj_new - obj_old
        end
    end
    s.iteration += 1
    new_incumbent = update_incumbent!(s, sol, t_end - s.time_start)
    terminate = check_termination(s)
    # TODO self.log_iteration(method.name, obj_old, sol, new_incumbent, terminate, res.log_info)
    if terminate
        s.run_time = time() - s.time_start
        res.terminate = true
    end
    res
end


"""
    check_termination(scheduler)

Check termination conditions and return true when to terminate.
"""
function check_termination(s::Scheduler)::Bool
    t = time()
    if 0 <= settings[:mh_titer] <= s.iteration
            # TODO or \
            #0 <= self.own_settings.mh_tciter <= self.iteration - self.incumbent_iteration or \
            #0 <= self.own_settings.mh_ttime <= t - self.time_start or \
            #0 <= self.own_settings.mh_tctime <= t - self.incumbent_time or \
            #0 <= self.own_settings.mh_tobj and not self.incumbent.is_worse_obj(self.incumbent.obj(),
            #                                                                   self.own_settings.mh_tobj):
        return true
    end
    false
end


"""
    perform_sequentially!(scheduler, solution, methods)

Applies the given methods sequentially, finally keeping the best solution as
incumbent.
"""
function perform_sequentially!(s::Scheduler, sol::Solution, meths::Vector{Method})
    for m in s.next_method(meths)
        res = self.perform_method!(m, sol)
        if res.terminate
            break
        end
        self.update_incumbent!(sol, time() - s.time_start)
    end
end


"""
    main_results(scheduler)
    
Print main results.
"""
function main_results(s::Scheduler)
    str = "T best solution: $(s.incumbent)\nT best obj: $(obj(s.incumbent))\n" *
        "T best iteration: $(s.incumbent_iteration)\n" *
        "T total iterations: $(s.iteration)\n" *
        @sprintf("T best time [s]: %.3f\n", s.incumbent_time) *
        @sprintf("T total time [s]: %.4f\n", s.run_time)
    # TODO self.logger.info(LogLevel.indent(s))
    print(str)
    check(s.incumbent)
end

#=
    def perform_method_pair(self, destroy: MHMethod, repair: MHMethod, sol: Solution) -> Result:
        """Performs a destroy/repair method pair on given solution and returns Results object.

        Also updates incumbent, iteration and the method's statistics in method_stats.
        Furthermore checks the termination condition and eventually sets terminate in the returned Results object.

        :param destroy: destroy destroy method to be performed
        :param repair: repair destroy method to be performed
        :param sol: solution to which the method is applied
        :returns: Results object
        """
        res = Result()
        obj_old = sol.obj()
        t_start = time.process_time()
        destroy.func(sol, destroy.par, res)
        t_destroyed = time.process_time()
        repair.func(sol, repair.par, res)
        t_end = time.process_time()
        self.update_stats_for_method_pair(destroy, repair, sol, res, obj_old,
                                          t_destroyed - t_start, t_end - t_destroyed)
        return res


    def update_stats_for_method_pair(self, destroy: MHMethod, repair: MHMethod, sol: Solution, res: Result, obj_old: TObj,
                                     t_destroy: float, t_repair: float):
        """Update statistics, incumbent and check termination condition after having performed a destroy+repair."""
        if __debug__ and self.own_settings.mh_checkit:
            sol.check()
        ms_destroy = self.method_stats[destroy.name]
        ms_destroy.applications += 1
        ms_destroy.netto_time += t_destroy
        ms_destroy.brutto_time += t_destroy
        ms_repair = self.method_stats[repair.name]
        ms_repair.applications += 1
        ms_repair.netto_time += t_repair
        ms_repair.brutto_time += t_repair
        obj_new = sol.obj()
        if sol.is_better_obj(sol.obj(), obj_old):
            ms_destroy.successes += 1
            ms_destroy.obj_gain += obj_new - obj_old
            ms_repair.successes += 1
            ms_repair.obj_gain += obj_new - obj_old
        self.iteration += 1
        new_incumbent = self.update_incumbent(sol, time.process_time() - self.time_start)
        terminate = self.check_termination()
        self.log_iteration(destroy.name+'+'+repair.name, obj_old, sol, new_incumbent,
                           terminate, res.log_info)
        if terminate:
            self.run_time = time.process_time() - self.time_start
            res.terminate = true

    def delayed_success_update(self, method: MHMethod, obj_old: TObj, t_start: TObj, sol: Solution):
        """Update an earlier performed method's success information in method_stats.

        :param method: earlier performed method
        :param obj_old: objective value of solution with which to compare to determine success
        :param t_start: time when the application of method dad started
        :param sol: current solution considered the final result of the method
        """
        t_end = time.process_time()
        ms = self.method_stats[method.name]
        ms.brutto_time += t_end - t_start
        obj_new = sol.obj()
        if sol.is_better_obj(sol.obj(), obj_old):
            ms.successes += 1
            ms.obj_gain += obj_new - obj_old

    def log_iteration_header(self):
        """Write iteration log header."""
        s = f"I {'iteration':>10} {'best':>16} {'obj_old':>16} {'obj_new':>16} "

        if self.population is not None:
            s += f"{'pop_obj_avg':>16} {'pop_obj_std':>16} "

        s += f"{'time':>12} {'method':<20} info"
        self.iter_logger.info(s)

    @staticmethod
    def is_logarithmic_number(x: int):
        const eps = 1e-12  # epsilon value for is_logarithmic_number()
        const log10_2 = log10(2)
        const log10_5 = log10(5)

        lr = log10(x) % 1
        return abs(lr) < Scheduler.eps or abs(lr-Scheduler.log10_2) < Scheduler.eps or \
            abs(lr-Scheduler.log10_5) < Scheduler.eps

    def log_iteration(self, method_name: str, obj_old: TObj, new_sol: Solution, new_incumbent: bool, in_any_case: bool,
                      log_info: Optional[str]):
        """Writes iteration log info.

        A line is written if in_any_case is set or in dependence of settings.mh_lfreq and settings.mh_lnewinc.

        :param method_name: name of applied method or '-' (if initially given solution)
        :param obj_old: objective value before applying last operator
        :param new_sol: newly created solution
        :param new_incumbent: true if the method yielded a new incumbent solution
        :param in_any_case: turns filtering of iteration logs off
        :param log_info: customize log info optionally added if not None
        """
        log = in_any_case or new_incumbent and self.own_settings.mh_lnewinc
        if not log:
            lfreq = self.own_settings.mh_lfreq
            if lfreq > 0 and self.iteration % lfreq == 0:
                log = true
            elif lfreq < 0 and self.is_logarithmic_number(self.iteration):
                log = true
        if log:
            s = f"I {self.iteration:>10d} {self.incumbent.obj():16.5f} {obj_old:16.6f} {new_sol.obj():16.5f} "

            if self.population is not None:
                s += f"{self.population.obj_avg():16.6f} {self.population.obj_std():16.5f} "

            s += f"{time.process_time()-self.time_start:12.4f} " \
                f"{method_name:<20} {log_info if log_info is not None else ''}"

            self.iter_logger.info(s)


    @staticmethod
    def sdiv(x, y):
        """Safe division: return x/y if y!=0 and nan otherwise."""
        if y == 0:
            return float('nan')
        else:
            return x/y

    def method_statistics(self):
        """Write overall statistics."""
        if not self.run_time:
            self.run_time = time.process_time() - self.time_start
        s = f"Method statistics:\n"
        s += f"S  method    iter   succ succ-rate%    tot-obj-gain    avg-obj-gain rel-succ%  net-time  " \
             f"net-time%  brut-time  brut-time%\n"

        total_applications = 0
        total_netto_time = 0.0
        total_successes = 0
        total_brutto_time = 0.0
        total_obj_gain = 0.0
        for ms in self.method_stats.values():
            total_applications += ms.applications
            total_netto_time += ms.netto_time
            total_successes += ms.successes
            total_brutto_time += ms.brutto_time
            total_obj_gain += ms.obj_gain

        for name, ms in self.method_stats.items():
            s += f"S {name:>7} {ms.applications:7d} {ms.successes:6d} " \
                 f"{self.sdiv(ms.successes, ms.applications)*100:10.4f} " \
                 f"{ms.obj_gain:15.5f} {self.sdiv(ms.obj_gain, ms.applications):15.5f} " \
                 f"{self.sdiv(ms.successes, total_successes)*100:9.4f} " \
                 f"{ms.netto_time:9.4f} {self.sdiv(ms.netto_time, self.run_time)*100:10.4f} " \
                 f"{ms.brutto_time:10.4f} {self.sdiv(ms.brutto_time, self.run_time)*100:11.4f}\n"
        s += f"S {'SUM/AVG':>7} {total_applications:7d} {total_successes:6d} " \
             f"{self.sdiv(total_successes, total_applications)*100:10.4f} " \
             f"{total_obj_gain:15.5f} {self.sdiv(total_obj_gain, total_applications):15.5f} " \
             f"{self.sdiv(self.sdiv(total_successes, len(self.method_stats)), total_successes)*100:9.4f} " \
             f"{total_netto_time:9.4f} {self.sdiv(total_netto_time, self.run_time)*100:10.4f} " \
             f"{total_brutto_time:10.4f} {self.sdiv(total_brutto_time, self.run_time)*100:11.4f}\n"
        self.logger.info(LogLevel.indent(s))




=#

end  # module
