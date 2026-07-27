#!/usr/bin/env julia

using Random
using Statistics
using Printf

function option(name,default)
    for (i,a) in pairs(ARGS)
        a=="--$name" && i<length(ARGS) && return ARGS[i+1]
        startswith(a,"--$name=") && return split(a,'=';limit=2)[2]
    end
    default
end
const ROOT=abspath(option("root","benchmark/results/rebuttal/soc_parametric_similarity"))
const SEEDS=parse.(Int,split(option("seeds",join(2026:2035,',')),','))
const BOOTSTRAP=parse(Int,option("bootstrap","10000"))
const DELTAS=(0.0,1e-4,1e-3,1e-2)

function readrows(path)
    isfile(path) || error("missing required artifact: $path")
    lines=readlines(path); isempty(lines) && error("empty CSV: $path")
    names=Symbol.(split(lines[1],','))
    [NamedTuple{Tuple(names)}(Tuple(split(line,','))) for line in lines[2:end]
     if !isempty(strip(line))]
end
function writerows(path,header,rows)
    mkpath(dirname(path))
    open(path,"w") do io
        println(io,join(header,','))
        for r in rows println(io,join(r,',')) end
    end
end
f64(x)=parse(Float64,x)
i64(x)=parse(Int,x)
label(d)=replace(string(d),'.'=>'p')

timing=Dict{Tuple{Int,Float64,Symbol},Float64}()
diagnostic=Any[]
warpdiagnostic=Any[]
for seed in SEEDS
    dir=joinpath(ROOT,"seeds","seed_$seed")
    correctness=readrows(joinpath(dir,"correctness","cpu_gpu.csv"))
    length(correctness)==4 || error("expected four correctness rows for seed $seed")
    all(r->r.status=="PASS",correctness) || error("correctness failure for seed $seed")
    for r in readrows(joinpath(dir,"timing","seed_level.csv"))
        i64(r.launches)==10 || error("expected ten launches: seed $seed")
        timing[(seed,f64(r.delta),Symbol(r.strategy))]=f64(r.median_ms)
    end
    append!(diagnostic,readrows(joinpath(dir,"diagnostic","per_cone_summary.csv")))
    append!(warpdiagnostic,readrows(joinpath(dir,"diagnostic","warp_root_work.csv")))
end
length(timing)==length(SEEDS)*8 || error("incomplete timing matrix")

seed_rows=Any[]
for seed in SEEDS, delta in DELTAS
    t=timing[(seed,delta,:threadWise)]
    w=timing[(seed,delta,:warpWise)]
    push!(seed_rows,(seed,delta,t,w,t/w,w/t))
end
writerows(joinpath(ROOT,"analysis","seed_level_ratios.csv"),
  ("seed","delta","thread_median_ms","warp_median_ms","thread_over_warp",
   "thread_speedup"),seed_rows)

rng=MersenneTwister(20260727)
draw_indices=[rand(rng,1:length(SEEDS),length(SEEDS)) for _ in 1:BOOTSTRAP]
effects=Any[]; bootrows=Any[]
function ci(values)
    s=sort(values)
    (s[clamp(floor(Int,.025length(s)),1,length(s))],
     s[clamp(ceil(Int,.975length(s)),1,length(s))])
end
for delta in DELTAS
    tv=[timing[(s,delta,:threadWise)] for s in SEEDS]
    wv=[timing[(s,delta,:warpWise)] for s in SEEDS]
    logratio=log.(tv./wv)
    draws=[exp(mean(logratio[idx])) for idx in draw_indices]
    lo,hi=ci(draws); gm=exp(mean(logratio))
    push!(effects,("strategy_ratio",delta,"threadWise/warpWise",gm,lo,hi,
                   mean(tv),median(tv),std(tv),minimum(tv),maximum(tv),
                   mean(wv),median(wv),std(wv),minimum(wv),maximum(wv)))
    for (draw,value) in pairs(draws)
        push!(bootrows,("strategy_ratio",delta,draw,value))
    end
end
for strategy in (:threadWise,:warpWise)
    logratio=log.([timing[(s,1e-2,strategy)]/timing[(s,0.0,strategy)]
                   for s in SEEDS])
    draws=[exp(mean(logratio[idx])) for idx in draw_indices]
    lo,hi=ci(draws); gm=exp(mean(logratio))
    push!(effects,("endpoint_ratio",NaN,"$(strategy):delta_0p01/delta_0",
                   gm,lo,hi,fill(NaN,10)...))
    for (draw,value) in pairs(draws)
        push!(bootrows,("endpoint_$(strategy)",NaN,draw,value))
    end
end
writerows(joinpath(ROOT,"analysis","effect_estimates.csv"),
  ("effect","delta","comparison","geometric_mean","ci95_low","ci95_high",
   "thread_mean_ms","thread_median_ms","thread_std_ms","thread_min_ms",
   "thread_max_ms","warp_mean_ms","warp_median_ms","warp_std_ms",
   "warp_min_ms","warp_max_ms"),effects)
writerows(joinpath(ROOT,"analysis","bootstrap_draws.csv"),
  ("effect","delta","draw","ratio"),bootrows)

# Diagnostic tables: aggregate seed-level summary rows without treating cones
# or warps as inferential observations.
diag_rows=Any[]
for delta in DELTAS, strategy in (:threadWise,:warpWise),
    metric in ("expansion","bisection","oracle")
    rows=filter(r->f64(r.delta)==delta && Symbol(r.strategy)==strategy &&
                   r.metric==metric,diagnostic)
    length(rows)==length(SEEDS) || error("incomplete diagnostic summary")
    push!(diag_rows,(delta,strategy,metric,
      median(f64.(getproperty.(rows,:median))),
      median(f64.(getproperty.(rows,:q90))),
      maximum(f64.(getproperty.(rows,:maximum)))))
end
writerows(joinpath(ROOT,"analysis","root_work_publication.csv"),
  ("delta","strategy","metric","median_of_seed_medians",
   "median_of_seed_p90","maximum_across_seeds"),diag_rows)

warp_rows=Any[]
for delta in DELTAS, strategy in (:threadWise,:warpWise),
    metric in ("expansion","bisection","oracle")
    rows=filter(r->f64(r.delta)==delta && Symbol(r.strategy)==strategy &&
                   r.metric==metric,warpdiagnostic)
    length(rows)==length(SEEDS) || error("incomplete within-warp summary")
    push!(warp_rows,(delta,strategy,metric,
      median(f64.(getproperty.(rows,:spread_median))),
      median(f64.(getproperty.(rows,:spread_p90))),
      maximum(f64.(getproperty.(rows,:spread_max))),
      median(f64.(getproperty.(rows,:modeled_efficiency_median))),
      median(f64.(getproperty.(rows,:modeled_efficiency_p10))),
      minimum(f64.(getproperty.(rows,:modeled_efficiency_min)))))
end
writerows(joinpath(ROOT,"analysis","warp_root_work_publication.csv"),
  ("delta","strategy","metric","median_seed_spread_median",
   "median_seed_spread_p90","maximum_spread","median_modeled_efficiency",
   "median_modeled_efficiency_p10","minimum_modeled_efficiency"),warp_rows)

function utilization_rows()
    raw=joinpath(ROOT,"utilization","raw")
    isdir(raw) || return Any[]
    output=Any[]
    for seed in SEEDS, delta in (0.0,1e-2), strategy in (:threadWise,:warpWise)
        stem="seed$(seed)_delta$(delta==0 ? "0" : "0p01")_$(strategy)"
        gpu=joinpath(raw,stem*".gpu.csv")
        app=joinpath(raw,stem*".application.log")
        isfile(gpu) && isfile(app) || error("missing utilization cell $stem")
        lines=filter(!isempty,strip.(readlines(gpu)))
        length(lines)>=35 || error("fewer than 35 utilization samples in $stem")
        window=lines[6:min(35,length(lines))]
        length(window)>=30 || error("fewer than 30 publication samples in $stem")
        fields=[strip.(split(line,',')) for line in window]
        util=parse.(Float64,getindex.(fields,3))
        mem=parse.(Float64,getindex.(fields,4))
        log=read(app,String)
        m=match(r"UTILIZATION_COMPLETE .*launches=(\d+).*elapsed_seconds=([0-9.eE+-]+).*restore_seconds=([0-9.eE+-]+).*projection_seconds=([0-9.eE+-]+)",log)
        m===nothing && error("missing duration ledger in $app")
        launches=parse(Int,m.captures[1]); wall=parse(Float64,m.captures[2])
        restore=parse(Float64,m.captures[3]); projection=parse(Float64,m.captures[4])
        gap=max(0,wall-restore-projection)
        discrepancy=abs(projection+restore+gap-wall)/wall
        discrepancy<=.01 || error("time ledger mismatch in $stem")
        push!(output,(seed,delta,strategy,length(window),mean(util),median(util),
          quantile(util,.10),quantile(util,.90),maximum(util),
          count(==(0.0),util)/length(util),mean(mem),launches,launches/wall,
          projection/wall,restore/wall,gap/wall,fields[1][2],discrepancy))
    end
    output
end
util=utilization_rows()
if !isempty(util)
    writerows(joinpath(ROOT,"utilization","aligned_seed_level.csv"),
      ("seed","delta","strategy","samples","mean_gpu_utilization",
       "median_gpu_utilization","p10_gpu_utilization","p90_gpu_utilization",
       "peak_gpu_utilization","zero_fraction","mean_memory_utilization",
       "launches","launches_per_second","projection_time_fraction",
       "restore_time_fraction","host_gap_fraction","gpu_uuid",
       "ledger_relative_error"),util)
end

open(joinpath(ROOT,"analysis","publication_tables.md"),"w") do io
    println(io,"# R1-3 Parametric-Similarity Results\n")
    println(io,"Statistical unit: ten workload seeds (2026–2035). Cell times are means of seed medians; ratios are paired geometric means with 10,000 seed-block bootstrap intervals.\n")
    println(io,"## Timing and strategy ratios\n")
    println(io,"| delta | thread mean (ms) | warp mean (ms) | T/W | 95% CI | thread speedup |")
    println(io,"|--:|--:|--:|--:|:--|--:|")
    for r in effects
        r[1]=="strategy_ratio" || continue
        @printf(io,"| %.4g | %.6f | %.6f | %.6f | [%.6f, %.6f] | %.3f× |\n",
          r[2],r[7],r[12],r[4],r[5],r[6],1/r[4])
    end
    println(io,"\n## Thread-wise natural-warp oracle-work summary\n")
    println(io,"The efficiency columns are work-count-based modeled lane-efficiency proxies, not hardware active-lane measurements.\n")
    println(io,"| delta | spread median | spread P90 | spread maximum | modeled efficiency median | modeled efficiency P10 |")
    println(io,"|--:|--:|--:|--:|--:|--:|")
    for r in warp_rows
        r[2]===:threadWise && r[3]=="oracle" || continue
        @printf(io,"| %.4g | %.3f | %.3f | %.0f | %.6f | %.6f |\n",
          r[1],r[4],r[5],r[6],r[7],r[8])
    end
    if !isempty(util)
        println(io,"\n## Endpoint GPU utilization\n")
        println(io,"Source: `nvidia-smi utilization.gpu`, one-second sampling, first five samples discarded, next thirty retained. Values below are means across ten seed-level summaries.\n")
        println(io,"| delta | strategy | mean | median | P10 | P90 | zero fraction | launches/second |")
        println(io,"|--:|:--|--:|--:|--:|--:|--:|--:|")
        for delta in (0.0,1e-2), strategy in (:threadWise,:warpWise)
            rows=filter(r->r[2]==delta && r[3]==strategy,util)
            @printf(io,"| %.4g | %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.4f | %.2f |\n",
              delta,strategy,mean(getindex.(rows,5)),mean(getindex.(rows,6)),
              mean(getindex.(rows,7)),mean(getindex.(rows,8)),
              mean(getindex.(rows,10)),mean(getindex.(rows,13)))
        end
    end
end
println("PARAMETRIC_SIMILARITY_ANALYSIS_COMPLETE root=$ROOT")
