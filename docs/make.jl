using Documenter
using MHLib
using MHLibDemos

makedocs(
    sitename = "MHLib.jl",
    modules = [MHLib, MHLibDemos],
    pages = [
        "Home" => "index.md",
        "MHLibDemos" => "MHLibDemos.md",
    ]
)
