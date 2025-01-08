# Module that starts a Julia subprocess and interfaces with it via stdin/stdout
# to, e.g., evaluate a function `f` many times.
# Most meaningful when function calls take considerable time so that parallelization
# makes sense but times for starting an independent Julia process for each function 
# call would still dominate a single function call.

import subprocess
from ConfigSpace import Configuration

# start Julia subprocess and load relevant code
julia = subprocess.Popen(["julia"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
julia.stdin.write(b'include("../julia-function-to-tune.jl")\n')
julia.stdin.flush()
reply = julia.stdout.readline()
print(reply, "Julia process started ................................")


# exemplary wrapper for Julia function to optimize
def f(config: Configuration, instance, seed: int=0) -> float:
    cmd = f'f(\"{instance}\", {seed}, {config["x"]}, {config["y"]}, \"{config["z"]}\")\n'
    print("cmd: ", cmd.encode())
    julia.stdin.write(cmd.encode())
    julia.stdin.flush()
    reply = julia.stdout.readline()
    res = float(reply)
    print(cmd, "-> ", res)
    return res
