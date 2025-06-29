"""
    module MHLibDemos

A package demonstrating the use of MHLib for various optimization problems.

Can also be used as templates for new projects using MHLib.
"""
module MHLibDemos

using Graphs
using MHLib

export create_or_read_simple_graph

"""
    create_or_read_simple_graph(name::AbstractString)

Read a simple unweighted graph from the specified file or create random G_n,m graph 
with n nodes and m edges if name is `gnm-n-m`.

File format:
- ``c <comments>    #`` ignored
- ``p <name> <number of nodes> <number of edges>``
- ``e <node_1> <node_2>    #`` for each edge, nodes are labeled in 1...number of nodes
"""
function create_or_read_simple_graph(name::AbstractString) :: SimpleGraph{Int}
    if startswith(name, "gnm-")
        # create random G_n,m graph
        par = split(name, '-')
        n = parse(Int, par[2])
        m = parse(Int, par[3])
        return SimpleGraph(n, m)
    else  # read from file
        graph =  SimpleGraph()
        for line in eachline(name)
            flag = line[1]
            if flag == 'p'
                split_line = split(line)
                n = parse(Int, split_line[3])
                m = parse(Int, split_line[4])
                graph = SimpleGraph(n)
            elseif flag == 'e'
                split_line = split(line)
                u = parse(Int, split_line[2])
                v = parse(Int, split_line[3])
                @assert add_edge!(graph, u, v)
            end
        end
        @assert nv(graph) > 0
        return graph
    end
end


include("MAXSAT.jl")
include("GraphColoring.jl")
include("MISP.jl")
include("MKP.jl")
include("TSP.jl")

include("../test/tests.jl")

end  # module
