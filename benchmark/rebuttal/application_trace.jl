#!/usr/bin/env julia

include("common.jl")
using .RebuttalCommon

manifest=abspath(option("manifest","benchmark/rebuttal/application_manifest.csv"))
output=abspath(option("output","benchmark/results/rebuttal/application"))
isfile(manifest) || error("application manifest not found: $manifest")
unquote(x)=replace(strip(x),r"^\"|\"$"=>"","\"\"" => "\"")
lines=readlines(manifest); names=unquote.(split(lines[1],','))
idx=Dict(x=>i for (i,x) in pairs(names))
for required in ("application","instance_size","sequence_index","command")
    haskey(idx,required) || error("manifest missing $required")
end
mkpath(output); rows=Any[]
for line in lines[2:end]
    isempty(strip(line)) && continue
    f=unquote.(split(line,','))
    app=f[idx["application"]]; size=f[idx["instance_size"]]
    seq=f[idx["sequence_index"]]; command=f[idx["command"]]
    stem="$(app)_$(replace(size,r\"[^A-Za-z0-9]+\"=>\"_\"))_seq$(seq)"
    trace=joinpath(output,stem*"_trace.csv")
    log=joinpath(output,stem*".log")
    env=copy(ENV)
    env["PDCS_APPLICATION"]=app
    env["PDCS_INSTANCE_SIZE"]=size
    env["PDCS_SEQUENCE_INDEX"]=seq
    env["PDCS_APPLICATION_TRACE_OUTPUT"]=trace
    started=time(); status=0
    open(log,"w") do io
        try
            run(pipeline(setenv(`bash -lc $command`,env),stdout=io,stderr=io))
        catch
            status=1
        end
    end
    trace_status=isfile(trace) ? "present" : "missing"
    push!(rows,(app,size,seq,command,status,time()-started,trace,trace_status,log))
end
write_csv(joinpath(output,"manifest.csv"),
 ("application","instance_size","sequence_index","command","exit_status",
  "wall_seconds","trace","trace_status","log"),rows)
