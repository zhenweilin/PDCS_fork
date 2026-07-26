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
ledger_root=abspath(option("ledger-root",root))

function quant(v,p)
    s=sort(v); s[clamp(round(Int,1+p*(length(s)-1)),1,length(s))]
end

rows=Any[]
for (dir,_,files) in walkdir(root), file in files
    (endswith(file,".dmon.txt") || endswith(file,".gpu.csv")) || continue
    path=joinpath(dir,file)
    values=Float64[]
    memory=Float64[]; power=Float64[]; clocks=Float64[]; temperatures=Float64[]
    if endswith(file,".gpu.csv")
        samples=[strip.(split(line,',')) for line in eachline(path) if !isempty(strip(line))]
        # The monitor is started immediately before START. Exclude exactly the
        # first five one-second stabilization samples and retain at most the
        # following 30 publication-window samples.
        samples=length(samples)>5 ? samples[6:min(end,35)] : Vector{Vector{SubString{String}}}()
        for fields in samples
            length(fields)>=10 || continue
            parsed=tryparse.(Float64,fields[4:10])
            any(isnothing,parsed) && continue
            push!(values,parsed[1]); push!(memory,parsed[2])
            push!(power,parsed[4]); push!(clocks,parsed[5])
            push!(temperatures,parsed[7])
        end
    else
        for line in eachline(path)
            startswith(strip(line),"#") && continue
            fields=split(strip(line))
            length(fields)>=4 || continue
            value=tryparse(Float64,replace(fields[4],"-"=>""))
            value===nothing || push!(values,value)
        end
    end
    stem=replace(replace(file,".dmon.txt"=>""),".gpu.csv"=>"")
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
    ledger=joinpath(ledger_root,experiment,"seed$seed",
      "duration_$(experiment)_$(layout)_$(strategy)_seed$(seed)_ledger.csv")
    zero_fraction=count(==(0.0),values)/length(values)
    metric(v)=isempty(v) ? NaN : mean(v)
    push!(rows,(experiment,delta,layout,strategy,seed,length(values),mean(values),
      median(values),quant(values,0.10),quant(values,0.90),maximum(values),
      zero_fraction,metric(memory),metric(power),metric(clocks),
      metric(temperatures),launches,elapsed,path,
      isfile(ledger) ? ledger : ""))
end
mkpath(dirname(output))
open(output,"w") do io
    println(io,"experiment,delta,layout,strategy,seed,samples,sm_busy_mean,sm_busy_median,sm_busy_p10,sm_busy_p90,sm_busy_peak,zero_sm_fraction,memory_busy_mean,power_mean_w,sm_clock_mean_mhz,temperature_mean_c,launches,elapsed_seconds,source,time_ledger")
    foreach(r->println(io,join(r,',')),rows)
end
println("UTILIZATION_SUMMARY_COMPLETE rows=$(length(rows)) output=$output")
