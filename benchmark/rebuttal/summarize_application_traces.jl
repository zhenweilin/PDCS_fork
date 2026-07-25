#!/usr/bin/env julia

using Statistics
include("common.jl")
using .RebuttalCommon

root=abspath(option("root","benchmark/results/rebuttal/application"))
output=abspath(option("output",joinpath(root,"application_trace_summary.csv")))
records=Any[]
for file in filter(f->endswith(f,"_trace.csv"),readdir(root;join=true))
    lines=readlines(file); length(lines)>1 || continue
    names=replace.(strip.(split(lines[1],',')),"\"" => "")
    idx=Dict(x=>i for (i,x) in pairs(names))
    required=("application","instance_size","projection_time_ms","solver_iteration_time_ms")
    all(haskey(idx,x) for x in required) || continue
    rows=[replace.(strip.(split(x,',')),"\"" => "") for x in lines[2:end]]
    p=parse.(Float64,getindex.(rows,idx["projection_time_ms"]))
    s=parse.(Float64,getindex.(rows,idx["solver_iteration_time_ms"]))
    push!(records,(rows[1][idx["application"]],rows[1][idx["instance_size"]],
          length(rows),sum(p),sum(s),sum(p)/sum(s),median(p),maximum(p)))
end
write_csv(output,("application","instance_size","iterations","projection_ms",
 "solver_iteration_ms","projection_fraction","projection_median_ms",
 "projection_max_ms"),records)
