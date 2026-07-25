#!/usr/bin/env julia

using CUDA
using PDCS: PDCS_GPU
include("common.jl")
include("generate_soc_cases.jl")
include("generate_exp_cases.jl")
include("diagnostic_root_profile.jl")
using .RebuttalCommon
using .GenerateSOCCases
using .GenerateExpCases
using .DiagnosticRootProfile

kind=option("kind","soc")
strategy=Symbol(option("strategy","threadWise"))
count=parse(Int,option("cone-count","1024"))
dimension=parse(Int,option("cone-dimension","10"))
seed=parse(Int,option("seed","2026"))
sigma_x=parse(Float64,option("sigma","1.0"))
sigma_d=parse(Float64,option("diagonal-sigma","1.0"))
output=abspath(option("output","benchmark/results/rebuttal/root_iterations_raw.csv"))
sample=parse(Int,option("sample","0"))

if kind=="soc"
    case=rescaled_soc(count,dimension;sigma_x,sigma_d,seed)
    input=case.x; scale=case.diagonal; type_code=Int64(22)
elseif kind=="exp"
    case=random_exp(count;sigma_x,sigma_d,seed)
    input=case.x; scale=case.diagonal; dimension=3; type_code=Int64(27)
else
    error("--kind must be soc or exp")
end
total=length(input)
starts=CuArray(Int64.(0:dimension:total-dimension))
sizes=CuArray(fill(Int64(dimension),count))
types=CuArray(fill(type_code,count))
x=CuArray(input); original=copy(x)
zero=CUDA.zeros(Float64,total); scale_gpu=CuArray(scale)
scale2=CuArray(scale.^2); scale_x=CUDA.zeros(Float64,total)
temp=CUDA.zeros(Float64,total); warm=CUDA.zeros(Float64,count)

args=(x,zero,zero,scale_gpu,scale2,scale_x,temp,warm,starts,sizes,Int64(count),types)
records=profile_project!(strategy,args...)
diagnostic_output=Array(x)
copyto!(x,original); fill!(warm,0); fill!(scale_x,0); fill!(temp,0)
if strategy===:threadWise
    PDCS_GPU.threadWise_block_proj(args...)
elseif strategy===:warpWise
    PDCS_GPU.warpWise_block_proj(args...)
else
    PDCS_GPU.blockWise_block_proj(args...)
end
CUDA.synchronize()
production_output=Array(x)
instrumentation_error=maximum(abs.(diagnostic_output.-production_output))
instrumentation_error<=5e-8 ||
    error("diagnostic kernel changed output: error=$instrumentation_error")

indices=sample>0 ? unique(round.(Int,range(1,count;length=min(sample,count)))) : 1:count
rows=Any[]
for i in indices
    r=records[i]
    push!(rows,(kind,strategy,count,dimension,seed,i,r.branch_code,
      r.interval_expansion_iterations,r.newton_attempts,r.newton_accepts,
      r.bisection_iterations,r.oracle_evaluations,r.warm_start_attempted,
      r.warm_start_accepted,r.max_iter_reached,r.final_residual,
      instrumentation_error))
end
write_csv(output,("kind","strategy","cone_count","cone_dimension","seed","cone",
 "branch_code","interval_expansion_iterations","newton_attempts","newton_accepts",
 "bisection_iterations","oracle_evaluations","warm_start_attempted",
 "warm_start_accepted","max_iter_reached","final_residual",
 "instrumentation_max_error"),rows)
@info "Diagnostic records written" output rows=length(rows) instrumentation_error
