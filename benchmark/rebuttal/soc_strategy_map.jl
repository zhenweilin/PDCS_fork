#!/usr/bin/env julia

using CUDA
using PDCS: PDCS_GPU
using LinearAlgebra
using Printf

include("common.jl")
include("generate_soc_cases.jl")
include("validate_projection.jl")
include("timing.jl")
using .RebuttalCommon
using .GenerateSOCCases
using .ValidateProjection
using .RebuttalTiming

counts = int_list(option("cone-counts", "3,10,100,1000,10000,100000,1000000"))
dimensions = int_list(option("dimensions", "10,32,100,500,2000,10000,50000"))
strategies = symbol_list(option("strategies", "gridWise,blockWise,warpWise,threadWise"))
trials = parse(Int, option("trials", "10"))
sigma = parse(Float64, option("sigma", "2.0"))
base_seed = parse(Int, option("seed", "2026"))
timeout_s = parse(Float64, option("timeout", "15"))
memory_fraction = parse(Float64, option("memory-fraction", "0.70"))
max_gridwise = parse(Int, option("max-gridwise-cones", "1000000"))
output = abspath(option("output", "benchmark/results/rebuttal/strategy_map/timings_raw.csv"))

all(in(STRATEGIES), strategies) || error("unknown strategy")
CUDA.functional() || error("CUDA is not functional")

function buffers(input, count, dimension)
    total = length(input)
    starts = Int64.(0:dimension:total-dimension)
    sizes = fill(Int64(dimension), count)
    types = fill(Int64(20), count)
    zero = CUDA.zeros(Float64, total)
    (; x=CuArray(input), original=CuArray(input), zero,
       warm=CUDA.zeros(Float64, count), starts, starts_gpu=CuArray(starts),
       sizes, sizes_gpu=CuArray(sizes), types, types_gpu=CuArray(types))
end

function project!(strategy, b, count)
    args = (b.x,b.zero,b.zero,b.zero,b.zero,b.zero,b.zero,b.warm)
    if strategy === :gridWise
        PDCS_GPU.gridWise_block_proj(args...,b.starts,b.sizes_gpu,b.sizes,Int64(count),b.types)
    elseif strategy === :blockWise
        PDCS_GPU.blockWise_block_proj(args...,b.starts_gpu,b.sizes_gpu,Int64(count),b.types_gpu)
    elseif strategy === :warpWise
        PDCS_GPU.warpWise_block_proj(args...,b.starts_gpu,b.sizes_gpu,Int64(count),b.types_gpu)
    else
        PDCS_GPU.threadWise_block_proj(args...,b.starts_gpu,b.sizes_gpu,Int64(count),b.types_gpu)
    end
end

header = ("cell","cone_count","cone_dimension","total_scalars","input_sigma",
          "strategy","trial","seed","runtime_ms","input_bandwidth_gbps",
          "max_error","finite","feasible","status","note")
rows = Any[]
cell = 0
for count in counts, dimension in dimensions
    cell += 1
    total = count * dimension
    required = 4total*sizeof(Float64) + 3count*sizeof(Int64) + 512*1024^2
    free = Int(CUDA.free_memory())
    if required > memory_fraction * free
        for strategy in strategies
            push!(rows,(cell,count,dimension,total,sigma,strategy,0,base_seed,
                        "","","","","","memory_excluded",
                        "required=$required;free=$free"))
        end
        continue
    end
    for trial in 0:trials-1
        seed = base_seed + 10_000cell + trial
        input = ordinary_soc(count,dimension;sigma,seed)
        for strategy in strategies
            if strategy === :gridWise && count > max_gridwise
                push!(rows,(cell,count,dimension,total,sigma,strategy,trial,seed,
                            "","","","","","skipped","grid-wise count limit"))
                continue
            end
            b = buffers(input,count,dimension)
            project!(strategy,b,count); CUDA.synchronize()
            copyto!(b.x,b.original); CUDA.synchronize()
            runtime = cuda_event_time(() -> project!(strategy,b,count))
            CUDA.synchronize()
            sample_count = min(count,1024)
            sample_input = Vector{Float64}(undef,sample_count*dimension)
            sample_output = similar(sample_input)
            for j in 1:sample_count
                src=(j-1)*dimension+1:j*dimension
                dst=(j-1)*dimension+1:j*dimension
                sample_input[dst] .= @view input[src]
                sample_output[dst] .= Array(@view b.x[src])
            end
            check = validate_soc(sample_input,sample_output,sample_count,dimension)
            status = check.status == "PASS" ?
                     (runtime > 1000timeout_s ? "timeout" : "PASS") : "FAIL"
            push!(rows,(cell,count,dimension,total,sigma,strategy,trial,seed,
                        @sprintf("%.9g",runtime),
                        @sprintf("%.9g",bandwidth_lower_bound(total,runtime)),
                        @sprintf("%.9g",check.max_error),check.finite,check.feasible,
                        status,status=="timeout" ? "completed beyond timeout threshold" : ""))
            b=nothing; CUDA.reclaim()
        end
    end
end
write_csv(output,header,rows)
@info "SOC strategy map complete" output rows=length(rows)
