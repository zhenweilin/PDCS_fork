#!/usr/bin/env julia

# Controlled warp-divergence benchmark for primal diagonal exponential-cone
# projection (projection type 27).
# The mixed layouts contain exactly the same cones and differ only in ordering.

using CUDA
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
flag(name) = any(==("--$name"), ARGS)

const CASE = option("case", "similar")
const STRATEGY = Symbol(option("strategy", "threadWise"))
const COUNT = parse(Int, option("cone-count", "1048576"))
const TRIALS = parse(Int, option("trials", "20"))
const SEED = parse(Int, option("seed", "2026"))
const OUTPUT = option("output", "")
const PROFILE_ONE = flag("profile-one")
const DURATION = parse(Float64, option("duration", "0"))
const REFERENCE_SAMPLES = parse(Int, option("reference-samples", "256"))
const INPUT_SIGMA = parse(Float64, option("sigma", "1.0"))
const DIAGONAL_SIGMA = parse(Float64, option("diagonal-sigma", "1.0"))

const CASES = ("similar", "heterogeneous", "mixed_grouped", "mixed_random", "mixed_interleaved")
const STRATEGIES = (:threadWise, :warpWise, :blockWise)
CASE in CASES || error("unknown case: $CASE")
STRATEGY in STRATEGIES || error("strategy must be threadWise, warpWise, or blockWise")
COUNT >= 32 && COUNT % 32 == 0 || error("cone count must be a multiple of 32")
isfinite(INPUT_SIGMA) && INPUT_SIGMA > 0 || error("--sigma must be finite and positive")
isfinite(DIAGONAL_SIGMA) && DIAGONAL_SIGMA > 0 ||
    error("--diagonal-sigma must be finite and positive")
CUDA.functional() || error("CUDA is not functional")

function canonical_cone!(x, first, class, rng; heterogeneous=false)
    if class == 1                         # feasible interior
        r = 0.2 + 0.02randn(rng); s = 1.0
        x[first:first+2] .= (r, s, 1.4s*exp(r/s))
    elseif class == 2                     # feasible boundary
        x[first:first+2] .= (-1.0, 0.0, 1.0)
    elseif class == 3                     # root-search case
        x[first:first+2] .= (1.0, 0.5, -1.0)
    else                                  # negative-orthant early exit
        x[first:first+2] .= (-1.0, -1.0, -1.0)
    end
    if heterogeneous
        x[first] += clamp(1.5randn(rng), -3.0, 3.0)
        x[first+1] += clamp(randn(rng), -1.5, 1.5)
        x[first+2] += clamp(1.5randn(rng), -3.0, 3.0)
    end
end

function make_case(case_name, count, seed)
    rng = MersenneTwister(seed)
    canonical = Vector{Float64}(undef, 3count)
    canonical_scale = Vector{Float64}(undef, 3count)
    for cone in 1:count
        first = 3cone - 2
        if case_name == "similar"
            canonical_cone!(canonical, first, 3, rng)
            canonical[first:first+2] .+= 0.01 .* randn(rng, 3)
        elseif case_name == "heterogeneous"
            canonical[first:first+2] .= INPUT_SIGMA .* randn(rng, 3)
        else
            canonical_cone!(canonical, first, mod(cone-1, 4)+1, rng)
        end
        canonical_scale[first:first+2] .= clamp.(
            abs.(DIAGONAL_SIGMA .* randn(rng, 3)), 1e-3, 1e3
        )
    end
    if case_name == "mixed_random"
        order = randperm(MersenneTwister(seed + 77_777), count)
        randomized = similar(canonical)
        randomized_scale = similar(canonical_scale)
        for destination in 1:count
            source = order[destination]
            randomized[3destination-2:3destination] .= canonical[3source-2:3source]
            randomized_scale[3destination-2:3destination] .= canonical_scale[3source-2:3source]
        end
        return randomized, randomized_scale
    end
    case_name != "mixed_grouped" && return canonical, canonical_scale

    grouped = similar(canonical)
    grouped_scale = similar(canonical_scale)
    group_size = count ÷ 4
    for destination in 1:count
        class = cld(destination, group_size)
        offset = mod(destination-1, group_size)
        source = 4offset + class
        grouped[3destination-2:3destination] .= canonical[3source-2:3source]
        grouped_scale[3destination-2:3destination] .= canonical_scale[3source-2:3source]
    end
    return grouped, grouped_scale
end

function buffers(input, scale, count)
    total = 3count
    starts = Int64.(0:3:total-3)
    sizes = fill(Int64(3), count)
    types = fill(Int64(27), count)
    zeros_gpu = CUDA.zeros(Float64, total)
    return (
        x=CuArray(input), original=CuArray(input),
        zero=zeros_gpu, scale=CuArray(scale), scale2=CuArray(scale.^2),
        scaled_x=CUDA.zeros(Float64, total), temp=CUDA.zeros(Float64, total),
        warm=CUDA.zeros(Float64, count),
        starts_gpu=CuArray(starts), sizes_gpu=CuArray(sizes), types_gpu=CuArray(types),
    )
end

function project!(strategy, b, count)
    args = (b.x, b.zero, b.zero, b.scale, b.scale2, b.scaled_x, b.temp, b.warm,
            b.starts_gpu, b.sizes_gpu, Int64(count), b.types_gpu)
    if strategy === :threadWise
        PDCS_GPU.threadWise_block_proj(args...)
    elseif strategy === :warpWise
        PDCS_GPU.warpWise_block_proj(args...)
    else
        PDCS_GPU.blockWise_block_proj(args...)
    end
end

function prepare!(b)
    copyto!(b.x, b.original)
    fill!(b.scaled_x, 0.0)
    fill!(b.temp, 0.0)
    fill!(b.warm, 0.0)
    CUDA.synchronize()
end

function reference_error(b, input, scale, count)
    indices = unique(round.(Int, range(1, count; length=min(count, REFERENCE_SAMPLES))))
    error = 0.0
    for cone in indices
        range3 = 3cone-2:3cone
        reference = copy(@view input[range3])
        PDCS_CPU.exponent_proj_diagonal!(reference, @view scale[range3])
        result = Array(@view b.x[range3])
        error = max(error, maximum(abs.(result .- reference)))
    end
    return error
end

input, scale = make_case(CASE, COUNT, SEED)
b = buffers(input, scale, COUNT)
prepare!(b); project!(STRATEGY, b, COUNT); CUDA.synchronize()

if PROFILE_ONE
    prepare!(b); project!(STRATEGY, b, COUNT); CUDA.synchronize()
elseif DURATION > 0
    println("PROFILE_READY case=$CASE strategy=$STRATEGY"); flush(stdout)
    let launches = 0, started = time_ns()
        while (time_ns()-started)/1e9 < DURATION
            prepare!(b); project!(STRATEGY, b, COUNT)
            launches += 1
        end
        elapsed = (time_ns()-started)/1e9
        @printf("case=%s strategy=%s launches=%d elapsed_seconds=%.6f\n", CASE, STRATEGY, launches, elapsed)
    end
else
    times = Float64[]
    errors = Float64[]
    for _ in 1:TRIALS
        prepare!(b)
        started = time_ns(); project!(STRATEGY, b, COUNT); CUDA.synchronize()
        push!(times, (time_ns()-started)/1e6)
        push!(errors, reference_error(b, input, scale, COUNT))
    end
    header = "case,strategy,variant,projection_type,cone_count,cone_dimension,input_sigma,diagonal_distribution,diagonal_sigma,seed,trials,mean_ms,std_ms,median_ms,min_ms,max_ms,max_error,status"
    deviation = length(times) > 1 ? std(times) : 0.0
    max_error = maximum(errors)
    status = isfinite(max_error) && max_error <= 5e-8 ? "PASS" : "FAIL"
    line = @sprintf("%s,%s,primalDiagonal,27,%d,3,%.9g,halfNormal,%.9g,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%s",
                    CASE, STRATEGY, COUNT, INPUT_SIGMA, DIAGONAL_SIGMA,
                    SEED, TRIALS, mean(times), deviation,
                    median(times), minimum(times), maximum(times), max_error, status)
    if isempty(OUTPUT)
        println(header); println(line)
    else
        mkpath(dirname(abspath(OUTPUT)))
        open(OUTPUT, "w") do io
            println(io, header); println(io, line)
        end
    end
end
