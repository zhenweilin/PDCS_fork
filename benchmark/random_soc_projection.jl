#!/usr/bin/env julia

# GPU-only timing sweep for random, equal-dimensional second-order cones.
# Example:
#   PDCS_SOC_DIMS=4,32,256,2048 PDCS_SOC_COUNTS=4,256,4096 \
#     julia --project=. benchmark/random_soc_projection.jl > soc_gpu.csv

using CUDA
using PDCS: PDCS_GPU
using Random
using Statistics
using Printf

CUDA.functional() || error("A functional CUDA device is required")
# `gridWise_block_proj` creates its module-owned cuBLAS handle lazily.

parse_list(name, default) = parse.(Int, split(get(ENV, name, default), ','))
const DIMS = parse_list("PDCS_SOC_DIMS", "4,16,64,256,2048")
const COUNTS = parse_list("PDCS_SOC_COUNTS", "1,2,3,4,32,1024,65536")
const SAMPLES = parse(Int, get(ENV, "PDCS_SOC_SAMPLES", "10"))
const MAX_ELEMENTS = parse(Int, get(ENV, "PDCS_SOC_MAX_ELEMENTS", "8000000"))
const SEED = parse(Int, get(ENV, "PDCS_SOC_SEED", "2026"))
const SIGMA = parse(Float64, get(ENV, "PDCS_SOC_SIGMA", "1.0"))
isfinite(SIGMA) && SIGMA > 0 || error("PDCS_SOC_SIGMA must be finite and positive")

"Closed-form Euclidean projection onto {(t,x): ||x|| <= t}."
function reference_soc_projection!(z, dimension)
    for first in 1:dimension:length(z)
        tail = @view z[first + 1:first + dimension - 1]
        tail_norm = sqrt(sum(abs2, tail))
        t = z[first]
        if tail_norm <= t
            continue
        elseif tail_norm <= -t
            @views z[first:first + dimension - 1] .= 0
        else
            alpha = (tail_norm + t) / (2tail_norm)
            z[first] = (tail_norm + t) / 2
            tail .*= alpha
        end
    end
    return z
end

function buffers(input, dimension, count)
    total = length(input)
    starts = Int64.(0:dimension:total - dimension)
    sizes = fill(Int64(dimension), count)
    types = fill(Int64(20), count) # ordinary SOC projection
    zeros_gpu = CUDA.zeros(Float64, total)
    return (
        x=CuArray(input), bl=zeros_gpu, bu=zeros_gpu,
        scaled=zeros_gpu, scaled2=zeros_gpu, scaled_x=zeros_gpu,
        temp=zeros_gpu, warm=CUDA.ones(Float64, count),
        starts=starts, starts_gpu=CuArray(starts),
        sizes=sizes, sizes_gpu=CuArray(sizes),
        types=types, types_gpu=CuArray(types),
    )
end

function project!(strategy, b, count)
    args = (b.x, b.bl, b.bu, b.scaled, b.scaled2, b.scaled_x, b.temp, b.warm)
    if strategy === :gridWise
        PDCS_GPU.gridWise_block_proj(args..., b.starts, b.sizes_gpu, b.sizes, Int64(count), b.types)
    elseif strategy === :blockWise
        PDCS_GPU.blockWise_block_proj(args..., b.starts_gpu, b.sizes_gpu, Int64(count), b.types_gpu)
    elseif strategy === :warpWise
        PDCS_GPU.warpWise_block_proj(args..., b.starts_gpu, b.sizes_gpu, Int64(count), b.types_gpu)
    elseif strategy === :threadWise
        PDCS_GPU.threadWise_block_proj(args..., b.starts_gpu, b.sizes_gpu, Int64(count), b.types_gpu)
    else
        error("unknown strategy: $strategy")
    end
end

function benchmark_case(input, dimension, count, strategy)
    reference = reference_soc_projection!(copy(input), dimension)
    b = buffers(input, dimension, count)

    copyto!(b.x, input)
    project!(strategy, b, count) # load/warm the kernel before measuring
    CUDA.synchronize()
    error_inf = maximum(abs, Array(b.x) .- reference)
    tolerance = 2e-12 * max(1.0, maximum(abs, reference))
    error_inf <= tolerance || return NaN, error_inf, "FAIL"

    times_ms = Vector{Float64}(undef, SAMPLES)
    for sample in eachindex(times_ms)
        copyto!(b.x, input)
        CUDA.synchronize()
        start = time_ns()
        project!(strategy, b, count)
        CUDA.synchronize()
        times_ms[sample] = (time_ns() - start) / 1e6
    end
    return median(times_ms), error_inf, "PASS"
end

rng = MersenneTwister(SEED)
gpu_name = CUDA.name(CUDA.device())
println("gpu,cone_count,cone_dimension,total_elements,strategy,median_ms,max_error,status")

for count in COUNTS, dimension in DIMS
    count >= 1 || error("cone counts must be positive; got count=$count")
    dimension >= 2 || error("SOC dimensions must be at least two; got dimension=$dimension")
    total = count * dimension
    total <= MAX_ELEMENTS || continue
    input = SIGMA .* randn(rng, Float64, total)
    types = fill(Int64(20), count)
    chosen = PDCS_GPU.select_projection_strategy(fill(Int64(dimension), count), types)

    # The few-cone implementation launches once per cone, so deliberately huge
    # counts are excluded from that comparison. The solver never selects it there.
    strategies = count < 3 ? (:gridWise,) :
                 count <= 32 ? (:gridWise, :blockWise, :warpWise, :threadWise) :
                               (:blockWise, :warpWise, :threadWise)
    for strategy in strategies
        elapsed_ms, error_inf, status = benchmark_case(input, dimension, count, strategy)
        label = strategy === chosen ? "$(strategy)*" : string(strategy)
        @printf("\"%s\",%d,%d,%d,%s,%.6f,%.3e,%s\n",
                gpu_name, count, dimension, total, label, elapsed_ms, error_inf, status)
        flush(stdout)
    end
end
