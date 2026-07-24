#!/usr/bin/env julia

using CUDA
using PDCS: PDCS_GPU
using Random
using Statistics
using Printf
using Dates

function option(name, default=nothing)
    prefix = "--$(name)="
    for (i, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix)+1:end]
        value == "--$(name)" && i < length(ARGS) && return ARGS[i+1]
    end
    return default
end
flag(name) = any(==("--$name"), ARGS)

const CASES = split(option("case", "similar"), ',')
const LEGACY_STRATEGIES = Dict(:massive => :threadWise, :sufficient => :warpWise,
                               :moderate => :blockWise)
normalize_strategy(value) = get(LEGACY_STRATEGIES, Symbol(value), Symbol(value))
const SELECTED_STRATEGIES = normalize_strategy.(split(option("strategy", "threadWise"), ','))
const COUNT = parse(Int, option("cone-count", "1048576"))
const DIMENSION = parse(Int, option("cone-dimension", "10"))
const SEED = parse(Int, option("seed", "2026"))
const TRIALS = parse(Int, option("trials", "10"))
const DURATION = parse(Float64, option("duration", "0"))
const OUTPUT = option("output", "")
const WARM_START = flag("warm-start")
const PROFILE_ONE = flag("profile-one")
const CHECK = flag("check")
const HETERO_SIGMA = parse(Float64, option("hetero-sigma", "2.0"))
const STRATEGIES = (:threadWise, :warpWise, :blockWise)

all(in(("uniform", "similar", "heterogeneous", "mixed_grouped", "mixed_interleaved")), CASES) || error("unknown case in $(join(CASES, ','))")
all(in(STRATEGIES), SELECTED_STRATEGIES) || error("strategy must be threadWise, warpWise, or blockWise")
(PROFILE_ONE || DURATION > 0) && (length(CASES) != 1 || length(SELECTED_STRATEGIES) != 1) && error("profile/duration mode accepts one case and strategy")
COUNT >= 32 || error("cone count must be at least 32")
COUNT % 32 == 0 || error("cone count must be a multiple of 32")
DIMENSION >= 3 || error("SOC dimension must be at least 3")
CUDA.functional() || error("CUDA is not functional")

function make_case(case_name, count, dimension, seed)
    total = count * dimension
    x = Vector{Float64}(undef, total)
    scale = Vector{Float64}(undef, total)
    vlen = dimension - 1  # number of tail components per cone

    # The two mixed layouts contain exactly the same cone multiset. Only their
    # placement into hardware warps differs.
    canonical_class(i) = mod(i-1, 4) + 1
    function source_index(i)
        case_name != "mixed_grouped" && return i
        group_size = count ÷ 4
        cls = cld(i, group_size)
        offset = mod(i-1, group_size)
        return 4offset + cls
    end

    canonical_x = case_name in ("mixed_grouped", "mixed_interleaved") ? Vector{Float64}(undef, total) : x
    canonical_scale = case_name in ("mixed_grouped", "mixed_interleaved") ? Vector{Float64}(undef, total) : scale
    sigma = case_name == "heterogeneous" ? HETERO_SIGMA : case_name == "similar" ? 0.05 : 0.0

    # Use two independent RNGs: seed + dimension for scales, seed + dimension + 1 for x
    rng_d = MersenneTwister(seed + dimension)
    rng_x = MersenneTwister(seed + dimension + 1)

    # Pre-generate all random data at once: each cone gets unique, independent values
    n_tail = count * vlen
    noise_d = randn(rng_d, n_tail)   # all scale noises: N(0,1)
    noise_x = randn(rng_x, n_tail)   # all x-component noises: N(0,1)

    for cone in 1:count
        first = (cone-1)*dimension + 1
        canonical_scale[first] = 1.0

        # Index into pre-generated noise arrays (cone * vlen offset for each cone)
        start_idx = (cone-1)*vlen + 1
        idx_range = start_idx:start_idx+vlen-1

        tail_d = @view canonical_scale[first+1:first+vlen]
        tail_x = @view canonical_x[first+1:first+vlen]

        tail_d .= clamp.(abs.(sigma .* @view(noise_d[idx_range])), 1e-3, 1e3)
        tail_x .= sigma .* @view(noise_x[idx_range])
        d = tail_d
        tail = tail_x
        norm_scaled = sqrt(sum(abs2, d .* tail))
        norm_inverse = sqrt(sum(abs2, tail ./ d))
        cls = canonical_class(cone)
        if case_name in ("uniform", "similar", "heterogeneous")
            canonical_x[first] = 0.20 * norm_scaled       # positive-t root search
        elseif cls == 1
            canonical_x[first] = 1.25 * norm_scaled       # already feasible
        elseif cls == 2
            canonical_x[first] = -1.25 * norm_inverse     # polar/zero
        elseif cls == 3
            canonical_x[first] = 0.20 * norm_scaled       # positive-t root search
        else
            canonical_x[first] = -0.20 * norm_inverse     # negative-t root search
        end
    end

    if case_name == "mixed_grouped"
        for destination in 1:count
            source = source_index(destination)
            dst = (destination-1)*dimension+1:destination*dimension
            src = (source-1)*dimension+1:source*dimension
            x[dst] .= canonical_x[src]
            scale[dst] .= canonical_scale[src]
        end
    elseif case_name == "mixed_interleaved"
        x = canonical_x
        scale = canonical_scale
    end
    return x, scale
end

function buffers(input, scale, count, dimension)
    total = length(input)
    starts = Int64.(0:dimension:total-dimension)
    sizes = fill(Int64(dimension), count)
    types = fill(Int64(22), count) # diagonally rescaled SOC
    return (
        x=CuArray(input), original=CuArray(input),
        bl=CUDA.zeros(Float64,total), bu=CUDA.zeros(Float64,total),
        scale=CuArray(scale), scale2=CuArray(scale.^2),
        scale_x=CUDA.zeros(Float64,total), temp=CUDA.zeros(Float64,total),
        warm=CUDA.zeros(Float64,count),
        starts=starts, starts_gpu=CuArray(starts),
        sizes=sizes, sizes_gpu=CuArray(sizes),
        types=types, types_gpu=CuArray(types),
    )
end

function project!(strategy, b)
    args = (b.x,b.bl,b.bu,b.scale,b.scale2,b.scale_x,b.temp,b.warm)
    count = Int64(length(b.sizes))
    if strategy === :threadWise
        PDCS_GPU.threadWise_block_proj(args...,b.starts_gpu,b.sizes_gpu,count,b.types_gpu)
    elseif strategy === :warpWise
        PDCS_GPU.warpWise_block_proj(args...,b.starts_gpu,b.sizes_gpu,count,b.types_gpu)
    else
        PDCS_GPU.blockWise_block_proj(args...,b.starts_gpu,b.sizes_gpu,count,b.types_gpu)
    end
end

function prepare!(b; preserve_warm=false)
    copyto!(b.x,b.original)
    preserve_warm || fill!(b.warm,0.0)
    fill!(b.scale_x,0.0)
    fill!(b.temp,0.0)
    CUDA.synchronize()
end

function one_timing!(strategy,b; preserve_warm=false)
    prepare!(b; preserve_warm)
    start = time_ns()
    project!(strategy,b)
    CUDA.synchronize()
    return (time_ns()-start)/1e6
end

function run_case(case_name, strategy)
    input, scale = make_case(case_name,COUNT,DIMENSION,SEED)
    b = buffers(input,scale,COUNT,DIMENSION)
    prepare!(b)
    project!(strategy,b)
    CUDA.synchronize()

    if PROFILE_ONE
        prepare!(b; preserve_warm=WARM_START)
        project!(strategy,b)
        CUDA.synchronize()
        return nothing
    elseif DURATION > 0
      let launches = 0, elapsed = 0.0, start = time_ns()
        println("PROFILE_READY case=$case_name strategy=$strategy")
        flush(stdout)
        start = time_ns()
        while elapsed < DURATION
            prepare!(b; preserve_warm=WARM_START)
            project!(strategy,b)
            launches += 1
            elapsed = (time_ns()-start)/1e9
        end
        @printf("case=%s strategy=%s warm_start=%s launches=%d elapsed_seconds=%.6f\n",case_name,strategy,WARM_START,launches,elapsed)
      end
      return nothing
    else
        max_error = NaN
        if CHECK
            prepare!(b)
            project!(:warpWise,b)
            reference = copy(b.x)
            prepare!(b)
            project!(strategy,b)
            CUDA.synchronize()
            max_error = maximum(abs.(b.x .- reference))
        end
        times = [one_timing!(strategy,b; preserve_warm=WARM_START) for _ in 1:TRIALS]
    line = @sprintf("%s,%s,%s,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",
                    case_name,strategy,WARM_START,COUNT,DIMENSION,SEED,TRIALS,
                    mean(times),std(times),median(times),minimum(times),maximum(times),max_error)
        return line
    end
end

lines = String[]
for case_name in CASES, strategy in SELECTED_STRATEGIES
    result = run_case(case_name,strategy)
    result === nothing || push!(lines,result)
end
if !isempty(lines)
    header = "case,strategy,warm_start,cone_count,cone_dimension,seed,trials,mean_ms,std_ms,median_ms,min_ms,max_ms,max_error_vs_warpWise"
    if isempty(OUTPUT)
        println(header); foreach(println,lines)
    else
        mkpath(dirname(abspath(OUTPUT)))
        open(OUTPUT,"w") do io
            println(io,header); foreach(line -> println(io,line),lines)
        end
    end
end
