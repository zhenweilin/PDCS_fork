#!/usr/bin/env julia

using Random
using Statistics
using Printf
include("common.jl")
using .RebuttalCommon

raw = abspath(option("raw","benchmark/results/rebuttal/strategy_map/timings_raw.csv"))
output = abspath(option("output","benchmark/results/rebuttal/strategy_map/strategy_map.csv"))
heuristic = abspath(option("heuristic","benchmark/results/rebuttal/strategy_map/heuristic.csv"))
bootstrap_n = parse(Int,option("bootstrap","10000"))

unquote(x)=replace(strip(x),r"^\"|\"$"=>"","\"\"" => "\"")
lines=readlines(raw); names=unquote.(split(lines[1],','))
idx=Dict(x=>i for (i,x) in pairs(names))
rows=[unquote.(split(x,',')) for x in lines[2:end] if !isempty(strip(x))]
valid=filter(r->r[idx["status"]]=="PASS",rows)
cells=sort(unique((parse(Int,r[idx["cone_count"]]),parse(Int,r[idx["cone_dimension"]])) for r in valid))
summary=Any[]; heuristic_rows=Any[]
rng=MersenneTwister(2026)
for (count,dimension) in cells
    selected=filter(r->parse(Int,r[idx["cone_count"]])==count &&
                       parse(Int,r[idx["cone_dimension"]])==dimension,valid)
    by_strategy=Dict{String,Dict{Int,Float64}}()
    for r in selected
        get!(by_strategy,r[idx["strategy"]],Dict{Int,Float64}())[
            parse(Int,r[idx["seed"]])] = parse(Float64,r[idx["runtime_ms"]])
    end
    isempty(by_strategy) && continue
    medians=Dict(s=>median(collect(values(v))) for (s,v) in by_strategy)
    winner=first(sort(collect(keys(medians));by=s->medians[s])); oracle=medians[winner]
    near=join(sort([s for (s,m) in medians if m <= 1.05oracle]),";")
    for strategy in sort(collect(keys(by_strategy)))
        common=sort(collect(intersect(keys(by_strategy[strategy]),keys(by_strategy[winner]))))
        ratios=[by_strategy[strategy][s]/by_strategy[winner][s] for s in common]
        boots=Float64[]
        if !isempty(ratios)
            for _ in 1:bootstrap_n
                push!(boots,median(rand(rng,ratios,length(ratios))))
            end
        end
        values_s=collect(values(by_strategy[strategy]))
        push!(summary,(count,dimension,strategy,length(values_s),mean(values_s),
              median(values_s),length(values_s)>1 ? std(values_s) : 0.0,
              median(ratios),quantile(boots,.025),quantile(boots,.975),
              winner,near))
    end
    push!(heuristic_rows,(count,dimension,winner,oracle,near))
end
write_csv(output,("cone_count","cone_dimension","strategy","trials","mean_ms",
 "median_ms","std_ms","paired_ratio","ratio_ci_low","ratio_ci_high",
 "fastest_strategy","within_5_percent"),summary)
write_csv(heuristic,("cone_count","cone_dimension","selected_strategy",
 "oracle_median_ms","within_5_percent"),heuristic_rows)
@info "Strategy map summarized" output heuristic
