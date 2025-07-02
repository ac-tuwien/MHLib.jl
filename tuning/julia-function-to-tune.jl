# The function for which to tune the parameters, i.e., minimize the returned value

using Random

"""
    f(inst::AbstractString, seed::Int, x::Float64, y::Int, z::String)::Float64

Demo function to tune, which returns a `Float64` value that should be minimized.

Parameters:
- `inst::AbstractString`: Assumed problem instance to be solved.
- `seed::Int`: Random seed for reproducibility.
- `x::Float64`: A floating-point parameter.
- `y::Int`: An integer parameter.
- `z::AbstractString`: A string parameter that can influence the result as well.
"""
function f(inst::AbstractString, seed::Int, x::Float64, y::Int, z::AbstractString)::Float64
    # load problem instance `inst` as needed

    Random.seed!(seed)

    # just some busy waiting to simulate a time-consuming calculation
    value=3
    for i in 1:1000000
        value += 1e-6 * sin(value)
    end

    # just some non-deterministic calculation involving the parameters to get a result
    result = (x + 2y - 2.5)^2 + rand()
    if z == "opt2"
        result += 100.0
    end

    return result
end
