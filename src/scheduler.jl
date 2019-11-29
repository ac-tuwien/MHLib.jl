#=
General scheduler for realizing (G)VNS, GRASP, IG and similar metaheuristics.

The module is intended for metaheuristics in which a set of methods
(or several of them) are in some way repeatedly applied to candidate solutions.
=#

module Scheduler

using ArgParse
import MHLib: @add_arg_table, settings_cfg

@add_arg_table settings_cfg begin
    "--mh_titer"
        help = "maximum number of iterations (<0: turned off)"
        arg_type = Int
        default = 100
end

#=
parser = get_settings_parser()
parser.add_argument("--mh_titer", type=int, default=100, help='maximum number of iterations (<0: turned off)')
parser.add_argument("--mh_tciter", type=int, default=-1,
                    help='maximum number of iterations without improvement (<0: turned off)')
parser.add_argument("--mh_ttime", type=int, default=-1, help='time limit [s] (<0: turned off)')
parser.add_argument("--mh_tctime", type=int, default=-1, help='maximum time [s] without improvement (<0: turned off)')
parser.add_argument("--mh_tobj", type=float, default=-1,
                    help='objective value at which should be terminated when reached (<0: turned off)')
add_bool_arg(parser, "mh_lnewinc", default=True, help='write iteration log if new incumbent solution')
parser.add_argument("--mh_lfreq", type=int, default=0,
                    help='frequency of writing iteration logs (0: none, >0: number of iterations, '
                         '-1: iteration 1,2,5,10,20,...')
add_bool_arg(parser, "mh_checkit", default=False, help='call check() for each solution after each method application')
parser.add_argument("--mh_workers", type=int, default=4, help='number of worker processes when using multiprocessing')


class Result:
    """Data in conjunction with a method application's result.

    Attributes
        - changed: if false, the solution has not been changed by the method application
        - terminate: if true, a termination condition has been fulfilled
        - log_info: customized log info
    """
    __slots__ = ('changed', 'terminate', 'log_info')

    def __init__(self):
        self.changed = True
        self.terminate = False
        self.log_info = None

    def __repr__(self):
        return f"(changed={self.changed}, terminate={self.terminate}, log_info={self.log_info})"


@dataclass
class Method:
    """A method to be applied by the scheduler.

    Attributes
        - name: name of the method; must be unique over all used methods
        - method: a function called for a Solution object
        - par: a parameter provided when calling the method
    """
    __slots__ = ('name', 'func', 'par')
    name: str
    func: Callable[[Solution, Any, Result], None]
    par: Any


@dataclass
class MethodStatistics:
    """Class that collects data on the applications of a Method.

    Attributes
        - applications: number of applications of this method
        - netto_time: accumulated time of all applications of this method without further costs (e.g., VND)
        - successes: number of applications in which an improved solution was found
        - obj_gain: sum of gains in the objective values over all successful applications
        - brutto_time: accumulated time of all applications of this method including further costs (e.g., VND)
    """
    applications: int = 0
    netto_time: float = 0.0
    successes: int = 0
    obj_gain: float = 0.0
    brutto_time: float = 0.0


class Scheduler(ABC):
    """Abstract class for metaheuristics that work by iteratively applying certain operators.

    Attributes
        - incumbent: incumbent solution, i.e., initial solution and always best solution so far encountered
        - incumbent_valid: True if incumbent is a valid solution to be considered
        - incumbent_iteration: iteration in which incumbent was found
        - incumbent_time: time at which incumbent was found
        - population: only used in derived population-based metaheurstics and here for logging, otherwise None
        - methods: list of all Methods
        - method_stats: dict of MethodStatistics for each Method
        - iteration: overall number of method applications
        - time_start: starting time of algorithm
        - run_time: overall runtime (set when terminating)
        - logger: pymhlib's logger for logging general info
        - iter_logger: pymhlib's logger for logging iteration info
        - own_settings: own settings object with possibly individualized parameter values
    """
    eps = 1e-12  # epsilon value for is_logarithmic_number()
    log10_2 = log10(2)  # log10(2)
    log10_5 = log10(5)  # log10(5)

    def __init__(self, sol: Solution, methods: List[Method], own_settings: dict = None, consider_initial_sol=False,
                 population=None):
        """
        :param sol: template/initial solution
        :param methods: list of scheduler methods to apply
        :param own_settings: an own settings object for locally valid settings that override the global ones
        :param consider_initial_sol: if true consider sol as valid solution that should be improved upon; otherwise
            sol is considered just a possibly uninitialized of invalid solution template
        :param population: optional population object used in derived population-based metaheuristic
        """
        self.incumbent = sol
        self.incumbent_valid = consider_initial_sol
        self.incumbent_iteration = 0
        self.incumbent_time = 0.0
        self.population = population
        self.methods = methods
        self.method_stats = {method.name: MethodStatistics() for method in methods}
        self.iteration = 0
        self.time_start = time.process_time()
        self.run_time = None
        self.logger = logging.getLogger("pymhlib")
        self.iter_logger = logging.getLogger("pymhlib_iter")
        self.log_iteration_header()
        if self.incumbent_valid:
            self.log_iteration('-', float('NaN'), sol, True, True, None)
        self.own_settings = OwnSettings(own_settings) if own_settings else settings

    def update_incumbent(self, sol, current_time):
        """If the given solution is better than incumbent (or we do not have an incumbent yet) update it."""
        if not self.incumbent_valid or sol.is_better(self.incumbent):
            self.incumbent.copy_from(sol)
            self.incumbent_iteration = self.iteration
            self.incumbent_time = current_time
            self.incumbent_valid = True
            return True

    @staticmethod
    def next_method(meths: List, *, randomize: bool = False, repeat: bool = False):
        """Generator for obtaining a next method from a given list of methods, iterating through all methods.

        :param meths: List of methods
        :param randomize: random order, otherwise consider given order
        :param repeat: repeat infinitely, otherwise just do one pass
        """
        if randomize:
            meths = meths.copy()
        while True:
            if randomize:
                random.shuffle(meths)
            for method in meths:
                yield method
            if not repeat:
                break

    def perform_method(self, method: Method, sol: Solution, delayed_success=False) -> Result:
        """Performs method on given solution and returns Results object.

        Also updates incumbent, iteration and the method's statistics in method_stats.
        Furthermore checks the termination condition and eventually sets terminate in the returned Results object.

        :param method: method to be performed
        :param sol: solution to which the method is applied
        :param delayed_success: if set the success is not immediately determined and updated but at some later
                call of delayed_success_update()
        :returns: Results object
        """
        res = Result()
        obj_old = sol.obj()
        t_start = time.process_time()
        method.func(sol, method.par, res)
        t_end = time.process_time()
        if __debug__ and self.own_settings.mh_checkit:
            sol.check()
        ms = self.method_stats[method.name]
        ms.applications += 1
        ms.netto_time += t_end - t_start
        obj_new = sol.obj()
        if not delayed_success:
            ms.brutto_time += t_end - t_start
            if sol.is_better_obj(sol.obj(), obj_old):
                ms.successes += 1
                ms.obj_gain += obj_new - obj_old
        self.iteration += 1
        new_incumbent = self.update_incumbent(sol, t_end - self.time_start)
        terminate = self.check_termination()
        self.log_iteration(method.name, obj_old, sol, new_incumbent, terminate, res.log_info)
        if terminate:
            self.run_time = time.process_time() - self.time_start
            res.terminate = True
        return res

    def perform_method_pair(self, destroy: Method, repair: Method, sol: Solution) -> Result:
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

    def perform_methods(self, methods: List[Method], sol: Solution) -> Result:
        """Performs all methods on given solution and returns Results object.

        Also updates incumbent, iteration and the method's statistics in method_stats.
        Furthermore checks the termination condition and eventually sets terminate in the returned Results object.

        :param methods: list of methods to perform
        :param sol: solution to which the method is applied
        :returns: Results object
        """
        res = Result()
        obj_old = sol.obj()
        method_name = ""
        for method in methods:
            if method_name != "":
                method_name += "+"
            method_name += method.name

            method.func(sol, method.par, res)
            if res.terminate:
                break
        t_end = time.process_time()

        self.iteration += 1
        new_incumbent = self.update_incumbent(sol, t_end - self.time_start)
        terminate = self.check_termination()
        self.log_iteration(method_name, obj_old, sol, new_incumbent, terminate, res.log_info)
        if terminate:
            self.run_time = time.process_time() - self.time_start
            res.terminate = True

        return res

    def update_stats_for_method_pair(self, destroy: Method, repair: Method, sol: Solution, res: Result, obj_old: TObj,
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
            res.terminate = True

    def delayed_success_update(self, method: Method, obj_old: TObj, t_start: TObj, sol: Solution):
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

    def check_termination(self):
        """Check termination conditions and return True when to terminate."""
        t = time.process_time()
        if 0 <= self.own_settings.mh_titer <= self.iteration or \
                0 <= self.own_settings.mh_tciter <= self.iteration - self.incumbent_iteration or \
                0 <= self.own_settings.mh_ttime <= t - self.time_start or \
                0 <= self.own_settings.mh_tctime <= t - self.incumbent_time or \
                0 <= self.own_settings.mh_tobj and not self.incumbent.is_worse_obj(self.incumbent.obj(),
                                                                                   self.own_settings.mh_tobj):
            return True

    def log_iteration_header(self):
        """Write iteration log header."""
        s = f"I {'iteration':>10} {'best':>16} {'obj_old':>16} {'obj_new':>16} "

        if self.population is not None:
            s += f"{'pop_obj_avg':>16} {'pop_obj_std':>16} "

        s += f"{'time':>12} {'method':<20} info"
        self.iter_logger.info(s)

    @staticmethod
    def is_logarithmic_number(x: int):
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
                log = True
            elif lfreq < 0 and self.is_logarithmic_number(self.iteration):
                log = True
        if log:
            s = f"I {self.iteration:>10d} {self.incumbent.obj():16.5f} {obj_old:16.6f} {new_sol.obj():16.5f} "

            if self.population is not None:
                s += f"{self.population.obj_avg():16.6f} {self.population.obj_std():16.5f} "

            s += f"{time.process_time()-self.time_start:12.4f} " \
                f"{method_name:<20} {log_info if log_info is not None else ''}"

            self.iter_logger.info(s)

    @abstractmethod
    def run(self):
        """Actually performs the optimization."""
        pass

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

    def main_results(self):
        """Write main results to logger."""
        s = f"T best solution: {self.incumbent}\nT best obj: {self.incumbent.obj()}\n" \
            f"T best iteration: {self.incumbent_iteration}\n" \
            f"T total iterations: {self.iteration}\n" \
            f"T best time [s]: {self.incumbent_time:.3f}\n" \
            f"T total time [s]: {self.run_time:.4f}\n"

        if self.population is not None:
            s += f"T population obj avg: {self.population.obj_avg()}\n" \
                f"T population obj std: {self.population.obj_std()}\n"

        self.logger.info(LogLevel.indent(s))
        self.incumbent.check()

    def perform_sequentially(self, sol: Solution, meths: List[Method]):
        """Applies the given list of methods sequentially, finally keeping the best solution as incumbent."""
        for m in self.next_method(meths):
            res = self.perform_method(m, sol)
            if res.terminate:
                break
            self.update_incumbent(sol, time.process_time())

=#

end  # module
