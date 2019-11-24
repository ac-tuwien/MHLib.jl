#!/usr/local/bin/julia

using MHLib
using MHLib.OneMax

println(ARGS)
parse_settings!()
println(get_settings_as_string())

s1 = OneMaxSolution{5}()
initialize!(s1)
s2 = OneMaxSolution{5}()
initialize!(s2)
s3 = copy(s1)
initialize!(s3)
copy!(s1,s3)
println("$s1, $(obj(s1))\n$s2, $(obj(s2))\n$s3, $(obj(s3))")
