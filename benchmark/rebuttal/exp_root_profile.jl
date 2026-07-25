#!/usr/bin/env julia

include("common.jl")
using .RebuttalCommon

cases=split(option("cases","similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved"),',')
strategies=split(option("strategies","threadWise,warpWise,blockWise"),',')
seeds=int_list(option("seeds","2026,2027,2028,2029,2030,2031,2032,2033,2034,2035"))
sigma_x=float_list(option("sigma-x","0.1,0.5,1,2,5,10"))
sigma_d=float_list(option("sigma-d","0.1,0.5,1,2,5,10"))
count=parse(Int,option("cone-count","1048576"))
trials=parse(Int,option("trials","1"))
output=abspath(option("output","benchmark/results/rebuttal/exp_root"))
dry_run=flag("dry-run")
harness=normpath(joinpath(@__DIR__,"..","exp_warp_profile.jl"))
mkpath(output); manifest=Any[]
settings=unique(vcat([(x,1.0,"input_sweep") for x in sigma_x],
                     [(1.0,d,"diagonal_sweep") for d in sigma_d]))
for (sx,sd,sweep) in settings, case_name in cases, strategy in strategies, seed in seeds
    stem="exp_$(sweep)_x$(sx)_d$(sd)_$(case_name)_$(strategy)_seed$(seed)"
    destination=joinpath(output,stem*".csv")
    command=`$(Base.julia_cmd()) --project=$(normpath(joinpath(@__DIR__,"../.."))) $harness --case $case_name --strategy $strategy --cone-count $count --sigma $sx --diagonal-sigma $sd --seed $seed --trials $trials --output $destination`
    started=time(); status=0; note=dry_run ? "dry_run" : ""
    if dry_run
        println(command)
    else
        try run(command) catch e; status=1; note=sprint(showerror,e) end
    end
    push!(manifest,(stem,sweep,sx,sd,case_name,strategy,seed,status,
                    time()-started,destination,note))
end
write_csv(joinpath(output,"manifest.csv"),
 ("run","sweep","input_sigma","diagonal_sigma","case","strategy","seed",
  "exit_status","wall_seconds","output","note"),manifest)
