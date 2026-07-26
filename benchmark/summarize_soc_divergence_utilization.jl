#!/usr/bin/env julia

using Statistics

function option(name, default)
    for (i,x) in pairs(ARGS)
        x=="--$name" && i<length(ARGS) && return ARGS[i+1]
        startswith(x,"--$name=") && return split(x,'=';limit=2)[2]
    end
    default
end
root=abspath(option("root","benchmark/results/rebuttal/soc_divergence_2x2"))
output=abspath(option("output",joinpath(root,"utilization_summary.csv")))

function quant(v,p)
    s=sort(v); s[clamp(round(Int,1+p*(length(s)-1)),1,length(s))]
end

rows=Any[]
for (dir,_,files) in walkdir(root), file in files
    endswith(file,".dmon.txt") || continue
    path=joinpath(dir,file)
    values=Float64[]
    for line in eachline(path)
        startswith(strip(line),"#") && continue
        fields=split(strip(line))
        length(fields)>=3 || continue
        # With `-s pucvmet`, dmon emits gpu, power, temperature, SM busy, ...
        value=tryparse(Float64,replace(fields[4],"-"=>""))
        value===nothing || push!(values,value)
    end
    stem=replace(file,".dmon.txt"=>"")
    parts=split(stem,'_')
    length(parts)>=4 || continue
    if parts[1]=="parametric" && length(parts)>=5
        experiment="parametric"
        delta=replace(parts[3],"p"=>".")
        layout="grouped"; strategy=parts[4]
        seed=parse(Int,replace(parts[5],"seed"=>""))
    else
        experiment=parts[1]; delta=""
        layout=parts[2]; strategy=parts[3]
        seed=parse(Int,replace(parts[4],"seed"=>""))
    end
    app=joinpath(dir,stem*".application.log")
    text=isfile(app) ? read(app,String) : ""
    launch_match=match(r"launches=(\d+)",text)
    elapsed_match=match(r"elapsed_seconds=([0-9.eE+-]+)",text)
    launches=launch_match===nothing ? 0 : parse(Int,launch_match.captures[1])
    elapsed=elapsed_match===nothing ? NaN : parse(Float64,elapsed_match.captures[1])
    isempty(values) && push!(values,NaN)
    push!(rows,(experiment,delta,layout,strategy,seed,length(values),mean(values),
      median(values),quant(values,0.10),quant(values,0.90),maximum(values),
      launches,elapsed,path))
end
mkpath(dirname(output))
open(output,"w") do io
    println(io,"experiment,delta,layout,strategy,seed,samples,sm_busy_mean,sm_busy_median,sm_busy_p10,sm_busy_p90,sm_busy_peak,launches,elapsed_seconds,source")
    foreach(r->println(io,join(r,',')),rows)
end
println("UTILIZATION_SUMMARY_COMPLETE rows=$(length(rows)) output=$output")
