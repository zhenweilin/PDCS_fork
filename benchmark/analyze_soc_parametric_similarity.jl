#!/usr/bin/env julia

using Statistics

function option(name,default)
    for (i,x) in pairs(ARGS)
        x=="--$name" && i<length(ARGS) && return ARGS[i+1]
        startswith(x,"--$name=") && return split(x,'=';limit=2)[2]
    end
    default
end
parse_seeds(x)=parse.(Int,split(x,','))
root=abspath(option("root","benchmark/results/rebuttal/soc_divergence_2x2/parametric"))
expected=parse_seeds(option("seeds",join(2026:2035,',')))
output=abspath(option("output",joinpath(root,"parametric_summary.csv")))

function rows(path)
    lines=readlines(path); header=Symbol.(split(lines[1],','))
    [NamedTuple{Tuple(header)}(Tuple(split(x,','))) for x in lines[2:end] if !isempty(x)]
end

result=Any[]
for delta_dir in sort(filter(isdir,readdir(root;join=true)))
    startswith(basename(delta_dir),"delta_") || continue
    seed_values=Int[]; thread=Float64[]; warp=Float64[]
    bisection=Float64[]; spread=Float64[]
    for seed in expected
        dir=joinpath(delta_dir,"seed$seed")
        timing=joinpath(dir,"timings_seed_summary.csv")
        correctness=joinpath(dir,"correctness.csv")
        diagnostic=joinpath(dir,"root_work_summary.csv")
        isfile(timing) && isfile(correctness) && isfile(diagnostic) ||
          error("incomplete parametric seed directory: $dir")
        any(r->r.status=="PASS",rows(correctness)) ||
          error("parametric correctness failed: $dir")
        tr=rows(timing)
        t=only(filter(r->r.strategy=="threadWise" && r.layout=="grouped",tr))
        w=only(filter(r->r.strategy=="warpWise" && r.layout=="grouped",tr))
        diag=rows(diagnostic)
        td=only(filter(r->r.strategy=="threadWise",diag))
        push!(seed_values,seed)
        push!(thread,parse(Float64,t.median_ms))
        push!(warp,parse(Float64,w.median_ms))
        push!(bisection,parse(Float64,td.bisection_median))
        push!(spread,parse(Float64,td.within_warp_spread_median))
    end
    seed_values==expected || error("parametric seed set incomplete in $delta_dir")
    ratio=thread./warp
    delta=replace(basename(delta_dir),"delta_"=>"","p"=>".")
    push!(result,(delta,length(expected),mean(thread),median(thread),
      mean(warp),median(warp),exp(mean(log.(ratio))),median(bisection),
      maximum(bisection),median(spread)))
end
mkpath(dirname(output))
open(output,"w") do io
    println(io,"delta,seeds,thread_mean_ms,thread_median_ms,warp_mean_ms,warp_median_ms,thread_warp_geomean_ratio,bisection_median_across_seeds,bisection_max_seed_median,within_warp_spread_median")
    foreach(r->println(io,join(r,',')),result)
end
println("PARAMETRIC_ANALYSIS_COMPLETE rows=$(length(result)) output=$output")
