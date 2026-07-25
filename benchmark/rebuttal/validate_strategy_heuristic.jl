#!/usr/bin/env julia

using Statistics
include("common.jl")
using .RebuttalCommon

function table(path)
    lines=readlines(path); names=replace.(strip.(split(lines[1],',')),"\"" => "")
    idx=Dict(x=>i for (i,x) in pairs(names))
    [Dict(names[i]=>replace(strip(f[i]),"\"" => "") for i in eachindex(names))
     for line in lines[2:end] if !isempty(strip(line))
     for f in (split(line,','),)]
end
heuristic_path=abspath(option("heuristic","benchmark/results/rebuttal/strategy_map/heuristic.csv"))
validation_path=abspath(option("validation","benchmark/results/rebuttal/strategy_validation/strategy_map.csv"))
output=abspath(option("output","benchmark/results/rebuttal/strategy_validation/heuristic_validation.csv"))
h=table(heuristic_path); v=table(validation_path)
cells=unique((parse(Int,r["cone_count"]),parse(Int,r["cone_dimension"])) for r in v)
rows=Any[]
for (count,dimension) in sort(cells)
    nearest=argmin(r->abs(log(count/parse(Int,r["cone_count"])))+
                      abs(log(dimension/parse(Int,r["cone_dimension"]))),h)
    selected=nearest["selected_strategy"]
    candidates=filter(r->parse(Int,r["cone_count"])==count &&
                         parse(Int,r["cone_dimension"])==dimension,v)
    med=Dict(r["strategy"]=>parse(Float64,r["median_ms"]) for r in candidates)
    isempty(med) && continue
    fastest=first(sort(collect(keys(med));by=s->med[s]))
    slowdown=get(med,selected,Inf)/med[fastest]
    push!(rows,(count,dimension,selected,fastest,slowdown,
                slowdown<=1.10,slowdown<=1.25))
end
write_csv(output,("cone_count","cone_dimension","selected_strategy",
 "oracle_strategy","slowdown","within_10_percent","within_25_percent"),rows)
pass90=count(r->r[6],rows)>=ceil(.9length(rows))
pass25=all(r->r[7],rows)
@info "Heuristic validation complete" output cells=length(rows) pass90 pass25
(pass90 && pass25) || exit(2)
