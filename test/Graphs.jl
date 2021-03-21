"""
    Graphs

Module containing utility functions for demos that use graphs.
"""
module Graphs

using LightGraphs

export create_or_read_simple_graph

"""
    create_or_read_simple_graph(name::AbstractString)

Read a simple unweighted graph from the specified file or create random G_n,m graph if
name is `gnm-n-m`.

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

end  # module
