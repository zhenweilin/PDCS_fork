#!/usr/bin/env julia

include("common.jl")
using .RebuttalCommon

cases = split(option("cases","uniform,similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved"),',')
strategies = split(option("strategies","threadWise,warpWise,blockWise"),',')
dimensions = int_list(option("dimensions","10,32,128,256"))
seeds = int_list(option("seeds","2026,2027,2028,2029,2030,2031,2032,2033,2034,2035"))
count = parse(Int,option("cone-count","262144"))
trials = parse(Int,option("trials","1"))
sigma = parse(Float64,option("hetero-sigma","2.0"))
warm_modes = split(option("warm-modes","cold,warm"),',')
output = abspath(option("output","benchmark/results/rebuttal/soc_root"))
dry_run = flag("dry-run")
harness = normpath(joinpath(@__DIR__,"..","rescaled_soc_warp_profile.jl"))
mkpath(output)
manifest=Any[]
for dimension in dimensions, case_name in cases, strategy in strategies,
    seed in seeds, warm in warm_modes
    stem="soc_$(case_name)_$(strategy)_d$(dimension)_$(warm)_seed$(seed)"
    destination=joinpath(output,stem*".csv")
    command=`$(Base.julia_cmd()) --project=$(normpath(joinpath(@__DIR__,"../.."))) $harness --case $case_name --strategy $strategy --cone-count $count --cone-dimension $dimension --seed $seed --trials $trials --hetero-sigma $sigma --check --output $destination`
    warm=="warm" && (command=`$command --warm-start`)
    started=time(); status=0; note=dry_run ? "dry_run" : ""
    if dry_run
        println(command)
    else
        try run(command) catch e; status=1; note=sprint(showerror,e) end
    end
    push!(manifest,(stem,case_name,strategy,dimension,warm,seed,status,
                    time()-started,destination,note))
end
write_csv(joinpath(output,"manifest.csv"),
 ("run","case","strategy","dimension","warm_mode","seed","exit_status",
  "wall_seconds","output","note"),manifest)
