#!/usr/bin/env julia

using Statistics
include("common.jl")
using .RebuttalCommon

root=abspath(option("root","benchmark/results/rebuttal"))
output=abspath(option("output",joinpath(root,"root_profile_summary.csv")))
iterations_output=abspath(option("iterations-output",
 joinpath(root,"root_iterations_summary.csv")))
rows=Any[]
iteration_rows=Any[]
for (dir,_,files) in walkdir(root), file in files
    endswith(file,".csv") || continue
    file in ("manifest.csv","root_profile_summary.csv") && continue
    path=joinpath(dir,file); lines=readlines(path)
    length(lines)>=2 || continue
    names=split(lines[1],','); pos=Dict(strip(x)=>i for (i,x) in pairs(names))
    if haskey(pos,"bisection_iterations")
        parsed=[replace.(strip.(split(line,',')),"\"" => "")
                for line in lines[2:end] if !isempty(strip(line))]
        total=[sum(parse(Int,f[pos[name]]) for name in
                   ("interval_expansion_iterations","newton_attempts",
                    "bisection_iterations")) for f in parsed]
        branches=parse.(Int,getindex.(parsed,pos["branch_code"]))
        residuals=parse.(Float64,getindex.(parsed,pos["final_residual"]))
        failures=count(f->parse(Int,f[pos["max_iter_reached"]])!=0,parsed)
        spreads=Float64[]; efficiencies=Float64[]
        for first in 1:32:length(total)
            group=total[first:min(first+31,end)]
            push!(spreads,maximum(group)-minimum(group))
            maximum(group)>0 && push!(efficiencies,sum(group)/(length(group)*maximum(group)))
        end
        finite_residuals=filter(isfinite,residuals)
        push!(iteration_rows,(relpath(path,root),
          parsed[1][pos["kind"]],parsed[1][pos["strategy"]],
          parsed[1][pos["seed"]],length(total),median(total),quantile(total,.9),
          quantile(total,.99),maximum(total),median(spreads),
          isempty(efficiencies) ? 1.0 : median(efficiencies),
          count(==(0),branches),count(==(1),branches),count(==(2),branches),
          count(==(3),branches),count(==(4),branches),count(==(-1),branches),
          failures,isempty(finite_residuals) ? NaN : maximum(abs,finite_residuals)))
        continue
    end
    haskey(pos,"median_ms") || haskey(pos,"runtime_ms") || continue
    for line in lines[2:end]
        f=replace.(strip.(split(line,',')),"\"" => "")
        status=get(pos,"status",0)>0 ? f[pos["status"]] : "unknown"
        runtime=haskey(pos,"median_ms") ? f[pos["median_ms"]] :
                haskey(pos,"runtime_ms") ? f[pos["runtime_ms"]] : ""
        case_name=haskey(pos,"case") ? f[pos["case"]] : ""
        strategy=haskey(pos,"strategy") ? f[pos["strategy"]] : ""
        seed=haskey(pos,"seed") ? f[pos["seed"]] : ""
        push!(rows,(relpath(path,root),case_name,strategy,seed,runtime,status))
    end
end
write_csv(output,("source","case","strategy","seed","runtime_ms","status"),rows)
write_csv(iterations_output,("source","kind","strategy","seed","cones",
 "iteration_p50","iteration_p90","iteration_p99","iteration_max",
 "warp_spread_median","modeled_active_lane_efficiency_median",
 "branch_feasible","branch_polar","branch_positive_root","branch_negative_root",
 "branch_near_zero","branch_unavailable","max_iter_failures",
 "max_abs_final_residual"),iteration_rows)
@info "Root profiles summarized" output rows=length(rows)
