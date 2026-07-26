#!/usr/bin/env julia

using Random
using Statistics
using Printf

function option(name, default)
    for (i,x) in pairs(ARGS)
        x == "--$name" && i < length(ARGS) && return ARGS[i+1]
        startswith(x,"--$name=") && return split(x,'=';limit=2)[2]
    end
    default
end

const ROOT = abspath(option("root",
    "benchmark/results/rebuttal/soc_divergence_2x2"))
function parse_seeds(value)
    if occursin(':',value)
        a,b=parse.(Int,split(value,':';limit=2))
        return collect(a:b)
    end
    parse.(Int,split(value,','))
end
const EXPECTED = parse_seeds(option("seeds", join(2026:2045,',')))
const BOOTSTRAPS = parse(Int, option("bootstrap", "10000"))
const OUTPUT = abspath(option("output", ROOT))

function csv_rows(path)
    lines=readlines(path); isempty(lines) && return NamedTuple[]
    names=Symbol.(split(lines[1],','))
    [NamedTuple{Tuple(names)}(Tuple(split(line,','))) for line in lines[2:end] if !isempty(line)]
end

function find_named(name)
    sort(filter(p -> basename(p)==name,
      [joinpath(dir,file) for (dir,_,files) in walkdir(ROOT) for file in files]))
end

timing_files=find_named("timings_seed_summary.csv")
correctness_files=find_named("correctness.csv")
isempty(timing_files) && error("no timings_seed_summary.csv below $ROOT")

correct=Set{Tuple{String,Int}}()
for path in correctness_files, row in csv_rows(path)
    row.status=="PASS" && push!(correct,(row.experiment,parse(Int,row.seed)))
end

cells=Dict{Tuple{String,Int,String,String},Float64}()
for path in timing_files, row in csv_rows(path)
    key=(row.experiment,parse(Int,row.seed),row.strategy,row.layout)
    haskey(cells,key) && error("duplicate timing cell $key")
    cells[key]=parse(Float64,row.median_ms)
end

experiments=sort(unique(k[1] for k in keys(cells)))
seed_rows=Any[]
effect_rows=Any[]

function percentile(v,p)
    x=sort(v); x[clamp(ceil(Int,p*length(x)),1,length(x))]
end

function summarize_ratio(experiment,name,values)
    logs=log.(values); n=length(logs)
    estimate=exp(mean(logs))
    se=n>1 ? std(logs)/sqrt(n) : NaN
    # The confirmatory design fixes n=20. Values below cover df=19; for a
    # diagnostic non-20 run the normal critical value is clearly labeled.
    onecrit=n==20 ? 1.729132812 : 1.644853627
    twocrit=n==20 ? 2.093024054 : 1.959963985
    tlo=exp(mean(logs)-twocrit*se); thi=exp(mean(logs)+twocrit*se)
    rng=MersenneTwister(20260726 + sum(codeunits(experiment*name)))
    boot=Vector{Float64}(undef,BOOTSTRAPS)
    for b in eachindex(boot)
        boot[b]=exp(mean(logs[rand(rng,1:n,n)]))
    end
    blo,bhi=percentile(boot,0.025),percentile(boot,0.975)
    one_lo=exp(mean(logs)-onecrit*se); one_hi=exp(mean(logs)+onecrit*se)
    decision = one_hi < 1.05 ? "RULE_OUT_5_PERCENT_PENALTY" :
               one_lo > 1.05 ? "MATERIAL_PENALTY" : "INCONCLUSIVE"
    (experiment,name,n,estimate,tlo,thi,blo,bhi,one_lo,one_hi,decision)
end

for experiment in experiments
    observed=sort(unique(k[2] for k in keys(cells) if k[1]==experiment))
    observed==EXPECTED ||
        error("$experiment seed block incomplete: expected $(EXPECTED), got $observed")
    for seed in observed
        (experiment,seed) in correct ||
            error("missing passing correctness row for $experiment seed $seed")
        getcell(strategy,layout)=get(cells,(experiment,seed,strategy,layout)) do
            error("missing cell $experiment seed=$seed $strategy/$layout")
        end
        tg=getcell("threadWise","grouped")
        ti=getcell("threadWise","interleaved")
        wg=getcell("warpWise","grouped")
        wi=getcell("warpWise","interleaved")
        a=ti/wi; ag=tg/wg; rt=ti/tg; rw=wi/wg; theta=rt/rw
        push!(seed_rows,(experiment,seed,tg,ti,wg,wi,a,ag,rt,rw,theta))
    end
    rows=[r for r in seed_rows if r[1]==experiment]
    for (name,index) in (("A",7),("A_grouped",8),("R_T",9),("R_W",10),("Theta",11))
        push!(effect_rows,summarize_ratio(experiment,name,[r[index] for r in rows]))
    end
end

mkpath(OUTPUT)
open(joinpath(OUTPUT,"seed_effects.csv"),"w") do io
    println(io,"experiment,seed,T_thread_grouped_ms,T_thread_interleaved_ms,T_warp_grouped_ms,T_warp_interleaved_ms,A,A_grouped,R_T,R_W,Theta")
    foreach(r->println(io,join(r,',')),seed_rows)
end
open(joinpath(OUTPUT,"effect_estimates.csv"),"w") do io
    println(io,"experiment,estimand,seeds,geometric_mean,t_ci_low,t_ci_high,bootstrap_low,bootstrap_high,one_sided_low,one_sided_high,decision_5pct")
    foreach(r->println(io,join(r,',')),effect_rows)
end
open(joinpath(OUTPUT,"publication_tables.md"),"w") do io
    println(io,"# SOC warp-divergence 2×2 results\n")
    println(io,"Publication timing uses uninstrumented kernels. The workload seed is the independent unit.\n")
    println(io,"| Experiment | Estimand | Seeds | Geometric mean | Paired t 95% CI | Seed bootstrap 95% CI | One-sided 95% bounds | 5% decision |")
    println(io,"|:--|:--|--:|--:|:--|:--|:--|:--|")
    for r in effect_rows
        @printf(io,"| %s | %s | %d | %.6f | [%.6f, %.6f] | [%.6f, %.6f] | [%.6f, %.6f] | %s |\n",
          r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11])
    end
end
println("ANALYSIS_COMPLETE output=$OUTPUT")
