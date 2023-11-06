# The function for which to tune the parameters, i.e., minimize

"""
    f(instance::AbstractString, x::Float64, y::Int, z::String)::Float64

Demo function to tune with SMAC3 in different ways.
"""
function f(instance::AbstractString, x::Float64, y::Int, z::String)::Float64
    # just some busy waiting:
    xx=3
    for i in 1:10000000
        xx = xx + 1e-6 *sin(xx)
    end

    result = (x + 2y - 2.5)^2
    
    if z == "opt2"
        result += 100.0
    end

    return result
end
