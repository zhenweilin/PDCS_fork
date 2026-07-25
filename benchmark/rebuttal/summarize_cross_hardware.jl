#!/usr/bin/env julia

include("common.jl")
using .RebuttalCommon

h100=abspath(option("h100",nothing))
a100=abspath(option("a100",nothing))
output=abspath(option("output","benchmark/results/rebuttal/cross_hardware.csv"))
function rows(path,label)
    lines=readlines(path); names=replace.(strip.(split(lines[1],',')),"\"" => "")
    idx=Dict(x=>i for (i,x) in pairs(names)); out=Any[]
    for line in lines[2:end]
        f=replace.(strip.(split(line,',')),"\"" => "")
        push!(out,(label,f[idx["cone_count"]],f[idx["cone_dimension"]],
                   f[idx["strategy"]],f[idx["median_ms"]],
                   f[idx["paired_ratio"]],f[idx["fastest_strategy"]]))
    end
    out
end
write_csv(output,("hardware","cone_count","cone_dimension","strategy",
 "median_ms","normalized_ratio","fastest_strategy"),
 vcat(rows(h100,"H100-sm90"),rows(a100,"A100-sm80")))
