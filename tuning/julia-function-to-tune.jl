# The function for which to tune the parameters, i.e., minimize the returned value

"""
    f(instance::AbstractString, seed::Int, x::Float64, y::Int, z::String)::Float64

Demo function to tune with SMAC3 in different ways.
"""
function f(instance::AbstractString, seed::Int, x::Float64, y::Int, z::AbstractString)::Float64
    # just some busy waiting:
    xx=3
    for i in 1:10000000
        xx = xx + 1e-6 *sin(xx)
    end

    # just some non-deterministic calculation involving the parameters to get a result
    result = (x + 2y - 2.5)^2 + rand()
    if z == "opt2"
        result += 100.0
    end

    return result
end
