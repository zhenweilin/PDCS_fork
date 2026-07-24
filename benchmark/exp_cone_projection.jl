#!/usr/bin/env julia

# GPU projection benchmark for three-dimensional exponential cones. Unlike an
# SOC, an exponential cone always has dimension three, so this benchmark varies
# cone count, projection variant, and GPU hierarchy strategy.

using CUDA
using Dates
using PDCS: PDCS_GPU, PDCS_CPU
using Printf
using Random
using Statistics

function option(name, default=nothing)
    prefix = "--$(name)="
    for (index, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix)+1:end]
        value == "--$(name)" && index < length(ARGS) && return ARGS[index+1]
    end
    return default
end

parse_ints(value) = parse.(Int, filter(x -> !isempty(x), split(value, ',')))
parse_symbols(value) = Symbol.(filter(x -> !isempty(x), split(value, ',')))

const COUNTS = parse_ints(option("cone-counts", "3,10,100,1000,10000,100000,1000000"))
const TRIALS = parse(Int, option("trials", "10"))
const SEED = parse(Int, option("seed", "2026"))
const OUTPUT = abspath(option("output", "benchmark/results/exp_projection/raw.csv"))
const STRATEGIES = Tuple(parse_symbols(option("strategies", "gridWise,blockWise,warpWise,threadWise")))
const VARIANTS = Tuple(parse_symbols(option("variants", "primalDiagonal")))
const INPUT_DISTRIBUTION = option("input-distribution", "heterogeneous")
const SIGMA = parse(Float64, option("sigma", "1.0"))
const MAX_GRIDWISE_CONES = parse(Int, option("max-gridwise-cones", "10000"))
const REFERENCE_SAMPLES = parse(Int, option("reference-samples", "256"))

const ALL_STRATEGIES = (:gridWise, :blockWise, :warpWise, :threadWise)
const ALL_VARIANTS = (:primal, :dual, :primalDiagonal, :dualDiagonal)
const LEGACY_STRATEGIES = Dict(:few => :gridWise, :moderate => :blockWise,
                               :sufficient => :warpWise, :massive => :threadWise)
const VARIANT_CODE = Dict(:primal => Int64(26), :dual => Int64(28),
                          :primalDiagonal => Int64(27), :dualDiagonal => Int64(29))

normalize_strategy(strategy) = get(LEGACY_STRATEGIES, strategy, strategy)
const NORMALIZED_STRATEGIES = Tuple(normalize_strategy.(STRATEGIES))

all(in(ALL_STRATEGIES), NORMALIZED_STRATEGIES) || error("unknown strategy")
all(in(ALL_VARIANTS), VARIANTS) || error("unknown exponential projection variant")
INPUT_DISTRIBUTION in ("similar", "heterogeneous") ||
    error("--input-distribution must be similar or heterogeneous")
isfinite(SIGMA) && SIGMA > 0 || error("--sigma must be finite and positive")
all(>=(3), COUNTS) || error("every cone count must be at least three")
TRIALS >= 1 || error("trials must be positive")
REFERENCE_SAMPLES >= 1 || error("reference samples must be positive")
CUDA.functional() || error("CUDA is not functional")

csv(value) = '"' * replace(string(value), '"' => "\"\"") * '"'
safe(call, fallback="unknown") = try string(call()) catch; fallback end

const DEVICE = CUDA.device()
const META = (
    run_id=get(ENV, "PDCS_RUN_ID", Dates.format(now(UTC), dateformat"yyyymmddTHHMMSSZ")),
    timestamp=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    gpu=CUDA.name(DEVICE),
    uuid=safe(() -> CUDA.uuid(DEVICE)),
    capability=safe(() -> CUDA.capability(DEVICE)),
    driver=get(ENV, "PDCS_NVIDIA_DRIVER", safe(CUDA.driver_version)),
    runtime=safe(CUDA.runtime_version),
    toolkit=get(ENV, "PDCS_CUDA_TOOLKIT", "unknown"),
    julia=string(VERSION),
    commit=get(ENV, "PDCS_GIT_COMMIT", "unknown"),
)

function make_input(count, seed, distribution, sigma)
    rng = MersenneTwister(seed)
    if distribution == "similar"
        # Keep every cone close to the same root-finding problem. The small
        # perturbation avoids benchmarking repeated bit-identical data.
        base = (1.0, 0.5, -1.0)
        input = Vector{Float64}(undef, 3count)
        for cone in 1:count
            first = 3cone - 2
            input[first:first+2] .= base .+ 0.01 .* randn(rng, 3)
        end
        return input
    end

    # Every coordinate is an independent N(0, sigma^2) sample. Varying sigma
    # changes input heterogeneity without changing the diagonal-scale law.
    return sigma .* randn(rng, Float64, 3count)
end

function make_scale(count, seed)
    rng = MersenneTwister(seed + 91_337)
    return exp.(0.02 .* randn(rng, Float64, 3count))
end

function reference!(value, variant, scale)
    if variant === :primal
        PDCS_CPU.exponent_proj!(value)
    elseif variant === :dual
        PDCS_CPU.dualExponent_proj!(value)
    elseif variant === :primalDiagonal
        PDCS_CPU.exponent_proj_diagonal!(value, scale)
    else
        temp = similar(value)
        PDCS_CPU.dualExponent_proj_diagonal!(value, scale, temp)
    end
    return value
end

function buffers(input, scale, count, variant)
    total = 3count
    starts = Int64.(0:3:total-3)
    sizes = fill(Int64(3), count)
    types = fill(VARIANT_CODE[variant], count)
    zeros_gpu = CUDA.zeros(Float64, total)
    return (
        x=CuArray(input), original=CuArray(input),
        bl=zeros_gpu, bu=zeros_gpu,
        scale=CuArray(scale), scale2=CuArray(scale.^2),
        scaled_x=CUDA.zeros(Float64, total), temp=CUDA.zeros(Float64, total),
        warm=CUDA.zeros(Float64, count),
        starts=starts, starts_gpu=CuArray(starts),
        sizes=sizes, sizes_gpu=CuArray(sizes),
        types=types, types_gpu=CuArray(types),
    )
end

function project!(strategy, b)
    args = (b.x, b.bl, b.bu, b.scale, b.scale2, b.scaled_x, b.temp, b.warm)
    count = Int64(length(b.sizes))
    if strategy === :gridWise
        PDCS_GPU.gridWise_block_proj(args..., b.starts, b.sizes_gpu, b.sizes, count, b.types)
    elseif strategy === :blockWise
        PDCS_GPU.blockWise_block_proj(args..., b.starts_gpu, b.sizes_gpu, count, b.types_gpu)
    elseif strategy === :warpWise
        PDCS_GPU.warpWise_block_proj(args..., b.starts_gpu, b.sizes_gpu, count, b.types_gpu)
    else
        PDCS_GPU.threadWise_block_proj(args..., b.starts_gpu, b.sizes_gpu, count, b.types_gpu)
    end
end

function sampled_indices(count)
    sample_count = min(count, REFERENCE_SAMPLES)
    return unique(round.(Int, range(1, count; length=sample_count)))
end

function reference_samples(input, scale, count, variant)
    indices = sampled_indices(count)
    references = Vector{NTuple{3,Float64}}(undef, length(indices))
    for (position, cone) in pairs(indices)
        range3 = 3cone-2:3cone
        value = copy(@view input[range3])
        reference!(value, variant, @view scale[range3])
        references[position] = Tuple(value)
    end
    return indices, references
end

function sample_error(x, indices, references)
    error = 0.0
    for (position, cone) in pairs(indices)
        result = Array(@view x[3cone-2:3cone])
        error = max(error, maximum(abs.(result .- references[position])))
    end
    return error
end

header = "run_id,timestamp_utc,gpu_name,gpu_uuid,compute_capability,driver_version,cuda_toolkit,cuda_runtime,julia_version,git_commit,cone_count,cone_dimension,input_distribution,input_sigma,variant,strategy,trial,seed,runtime_ms,max_error,status,note"
mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    println(io, header)
    for count in COUNTS, variant in VARIANTS
        input = make_input(count, SEED + count + VARIANT_CODE[variant], INPUT_DISTRIBUTION, SIGMA)
        scale = make_scale(count, SEED + count)
        indices, references = reference_samples(input, scale, count, variant)
        for strategy in NORMALIZED_STRATEGIES
            if strategy === :gridWise && count > MAX_GRIDWISE_CONES
                fields = (META.run_id, META.timestamp, META.gpu, META.uuid, META.capability,
                          META.driver, META.toolkit, META.runtime, META.julia, META.commit,
                          count, 3, INPUT_DISTRIBUTION, SIGMA, variant, strategy, 0, SEED, "", "",
                          "SKIPPED", "cone count exceeds --max-gridwise-cones")
                println(io, join(csv.(fields), ',')); flush(io)
                continue
            end
            b = buffers(input, scale, count, variant)
            copyto!(b.x, b.original); project!(strategy, b); CUDA.synchronize()
            for trial in 1:TRIALS
                copyto!(b.x, b.original)
                fill!(b.warm, 0.0)
                CUDA.synchronize()
                started = time_ns()
                project!(strategy, b)
                CUDA.synchronize()
                runtime_ms = (time_ns() - started) / 1e6
                error = sample_error(b.x, indices, references)
                status = isfinite(error) && error <= 5e-8 ? "PASS" : "FAIL"
                fields = (META.run_id, META.timestamp, META.gpu, META.uuid, META.capability,
                          META.driver, META.toolkit, META.runtime, META.julia, META.commit,
                          count, 3, INPUT_DISTRIBUTION, SIGMA, variant, strategy, trial, SEED,
                          @sprintf("%.9g", runtime_ms), @sprintf("%.9g", error), status, "")
                println(io, join(csv.(fields), ',')); flush(io)
            end
            b = nothing
            CUDA.reclaim()
        end
    end
end

@info "Exponential-cone projection benchmark complete" output=OUTPUT gpu=META.gpu
