#!/usr/bin/env julia

# GPU-only, fixed-cone-count SOC projection experiment for reviewer R1-2.

using CUDA
using Dates
using PDCS: PDCS_GPU
using Printf
using Random
using Statistics

function argument(name, default=nothing)
    prefix = "--$(name)="
    for (i, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix) + 1:end]
        value == "--$(name)" && i < length(ARGS) && return ARGS[i + 1]
    end
    return default
end

parse_int_list(value) = parse.(Int, split(value, ','))
const CONE_COUNT = parse(Int, argument("cone-count", get(ENV, "PDCS_SOC_CONE_COUNT", "100")))
const DIMENSIONS = parse_int_list(argument("dimensions", get(ENV, "PDCS_SOC_DIMS", "4,16,64,256,1024,4096,16384,65536,262144,1048576,4194304")))
const TRIALS = parse(Int, argument("trials", get(ENV, "PDCS_SOC_TRIALS", "10")))
const BASE_SEED = parse(Int, argument("seed", get(ENV, "PDCS_SOC_SEED", "2026")))
const SIGMA = parse(Float64, argument("sigma", get(ENV, "PDCS_SOC_SIGMA", "1.0")))
const TIMEOUT_SECONDS = parse(Float64, argument("timeout", get(ENV, "PDCS_SOC_TIMEOUT", "15")))
const MEMORY_RESERVE = parse(Float64, argument("memory-reserve", get(ENV, "PDCS_SOC_MEMORY_RESERVE", "0.20")))
const OUTPUT = abspath(argument("output", get(ENV, "PDCS_SOC_RAW_CSV", "benchmark/results/soc_dimension_raw.csv")))
const ALL_STRATEGIES = (:gridWise, :blockWise, :warpWise, :threadWise)
const LEGACY_STRATEGIES = Dict(:few => :gridWise, :moderate => :blockWise,
                               :sufficient => :warpWise, :massive => :threadWise)
normalize_strategy(value) = get(LEGACY_STRATEGIES, Symbol(value), Symbol(value))
parse_strategies(value) = Tuple(normalize_strategy.(filter(x -> !isempty(x), split(value, ','))))
const STRATEGIES = parse_strategies(argument("strategies", get(ENV, "PDCS_SOC_STRATEGIES", "gridWise,blockWise,warpWise,threadWise")))
const SKIPPED_STRATEGIES = parse_strategies(argument("skip-strategies", get(ENV, "PDCS_SOC_SKIP_STRATEGIES", "")))
const LABELS = Dict(:gridWise => "Grid-wise", :blockWise => "Block-wise",
                    :warpWise => "Warp-wise", :threadWise => "Thread-wise")

CONE_COUNT >= 3 || error("cone count must be at least 3 for all three GPU strategies")
TRIALS >= 1 || error("trials must be positive")
isfinite(SIGMA) && SIGMA > 0 || error("--sigma must be finite and positive")
all(>=(2), DIMENSIONS) || error("every full SOC dimension must be at least 2")
all(in(ALL_STRATEGIES), STRATEGIES) || error("unknown strategy in --strategies")
all(in(ALL_STRATEGIES), SKIPPED_STRATEGIES) || error("unknown strategy in --skip-strategies")
isempty(intersect(STRATEGIES, SKIPPED_STRATEGIES)) || error("a strategy cannot be both run and skipped")
0 <= MEMORY_RESERVE < 1 || error("memory reserve must be in [0,1)")
CUDA.functional() || error("CUDA.jl cannot use a GPU; run the runner's preflight checks")

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

function reference_soc_projection!(z)
    t = z[1]
    tail = @view z[2:end]
    tail_norm = sqrt(sum(abs2, tail))
    if tail_norm <= t
        return z
    elseif tail_norm <= -t
        fill!(z, 0)
    else
        alpha = (tail_norm + t) / (2tail_norm)
        z[1] = (tail_norm + t) / 2
        tail .*= alpha
    end
    return z
end

function buffers(dimension, count)
    total = dimension * count
    starts = Int64.(0:dimension:total - dimension)
    sizes = fill(Int64(dimension), count)
    types = fill(Int64(20), count)
    zero_work = CUDA.zeros(Float64, total)
    return (
        x=CUDA.zeros(Float64, total), zero=zero_work,
        warm=CUDA.ones(Float64, count), starts=starts,
        starts_gpu=CuArray(starts), sizes=sizes, sizes_gpu=CuArray(sizes),
        types=types, types_gpu=CuArray(types),
    )
end

function project!(strategy, b)
    args = (b.x, b.zero, b.zero, b.zero, b.zero, b.zero, b.zero, b.warm)
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

function write_row(io; dimension, strategy, trial, seed, runtime_ms="", max_error="", status, note="")
    fields = (META.run_id, META.timestamp, META.gpu, META.uuid, META.capability,
              META.driver, META.toolkit, META.runtime, META.julia, META.commit,
              CONE_COUNT, dimension, CONE_COUNT * dimension, SIGMA, LABELS[strategy],
              trial, seed, runtime_ms, max_error, status, note)
    println(io, join(csv.(fields), ','))
    flush(io)
end

function sampled_cones(x, dimension)
    indices = unique((1, cld(CONE_COUNT, 2), CONE_COUNT))
    return [(index, Array(@view x[(index - 1) * dimension + 1:index * dimension])) for index in indices]
end

mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    println(io, "run_id,timestamp_utc,gpu_name,gpu_uuid,compute_capability,driver_version,cuda_toolkit,cuda_runtime,julia_version,git_commit,cone_count,cone_dimension,total_dimension,input_sigma,strategy,trial,seed,runtime_ms,max_error,status,note")
    for dimension in DIMENSIONS
            for strategy in SKIPPED_STRATEGIES
                write_row(io; dimension, strategy, trial=0, seed=BASE_SEED,
                          status="SKIPPED_TIMEOUT_RISK",
                          note="not launched because this strategy performs one sequential projection per cone")
            end
            total = CONE_COUNT * dimension
            # x and the shared scratch array are the dominant allocations. Use
            # a conservative third-array allowance plus 256 MiB for CUDA state.
            required = 3 * total * sizeof(Float64) + 256 * 1024^2
            available = Int(CUDA.free_memory())
            if required > floor(Int, available * (1 - MEMORY_RESERVE))
                for strategy in STRATEGIES
                    write_row(io; dimension, strategy, trial=0, seed=BASE_SEED,
                              status="SKIPPED_MEMORY",
                              note="estimated_bytes=$(required);available_bytes=$(available)")
                end
                continue
            end

            b = try
                buffers(dimension, CONE_COUNT)
            catch err
                if err isa CUDA.OutOfGPUMemoryError
                    for strategy in STRATEGIES
                        write_row(io; dimension, strategy, trial=0, seed=BASE_SEED,
                                  status="SKIPPED_MEMORY", note=sprint(showerror, err))
                    end
                    CUDA.reclaim()
                    continue
                end
                rethrow()
            end

            for strategy in STRATEGIES
                timed_out = false
                # Compile and warm the selected kernel without timing it.
                CUDA.seed!(BASE_SEED)
                randn!(b.x)
                b.x .*= SIGMA
                project!(strategy, b)
                CUDA.synchronize()

                for trial in 1:TRIALS
                    seed = BASE_SEED + 10_000 * findfirst(==(dimension), DIMENSIONS) + trial
                    CUDA.seed!(seed)
                    randn!(b.x)
                    b.x .*= SIGMA
                    samples = trial == 1 ? sampled_cones(b.x, dimension) : Tuple{Int,Vector{Float64}}[]
                    CUDA.synchronize()
                    start_ns = time_ns()
                    try
                        project!(strategy, b)
                        CUDA.synchronize()
                    catch err
                        write_row(io; dimension, strategy, trial, seed, status="ERROR",
                                  note=sprint(showerror, err))
                        break
                    end
                    elapsed_ms = (time_ns() - start_ns) / 1e6

                    max_error = 0.0
                    reference_scale = 1.0
                    if trial == 1
                        for (index, original) in samples
                            expected = reference_soc_projection!(original)
                            actual = Array(@view b.x[(index - 1) * dimension + 1:index * dimension])
                            max_error = max(max_error, maximum(abs, actual .- expected))
                            reference_scale = max(reference_scale, maximum(abs, expected))
                        end
                    end
                    tolerance = 2e-12 * reference_scale
                    status = max_error <= tolerance ? "PASS" : "FAIL"
                    elapsed_ms > 1000TIMEOUT_SECONDS && (status = "TIMEOUT"; timed_out = true)
                    error_text = trial == 1 ? @sprintf("%.6e", max_error) : ""
                    write_row(io; dimension, strategy, trial, seed,
                              runtime_ms=@sprintf("%.6f", elapsed_ms),
                              max_error=error_text, status)
                    (timed_out || status == "FAIL") && break
                end
            end
            b = nothing
            CUDA.reclaim()
        end
end

@info "SOC dimension benchmark complete" output=OUTPUT gpu=META.gpu
