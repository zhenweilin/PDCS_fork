#!/usr/bin/env julia

"""
Solve one CBF file with PDCS_GPU through JumpRW's public `solve_cbf` API.

Run from the JumpRW repository root:

    julia --startup-file=no --threads=auto \
        --project=external/PDCS_fork \
        external/PDCS_fork/test/solve_pdcs_gpu_cbf.jl \
        --input-file /path/to/model.cbf.gz
"""

using Dates

const GPU_SOLVER_NAME = "PDCS_GPU"
const GPU_JUMPRW_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PDCS_GPU_SOURCE = normpath(joinpath(
    @__DIR__,
    "..",
    "src",
    "pdcs_gpu",
    "PDCS_GPU.jl",
))

function gpu_usage(io::IO=stdout)
    println(io, "Usage:")
    println(
        io,
        "  julia --threads=auto --project=external/PDCS_fork " *
        "external/PDCS_fork/test/solve_pdcs_gpu_cbf.jl [options]",
    )
    println(io)
    println(io, "Options:")
    println(io, "  --input_file PATH    CBF, CBF.GZ, or CBF.BZ2 file (required)")
    println(io, "  --tolerance VALUE    Absolute/relative tolerance (default: 1e-6)")
    println(io, "  --time_limit SECONDS Solver time limit (default: 3600)")
    println(io, "  --verbose LEVEL      0, 1, or 2 (default: 2)")
    println(io, "  --workers COUNT      JumpRW reader workers")
    println(io, "  -h, --help           Show this help")
end

normalize_gpu_option(name) = replace(name, '-' => '_')

function parse_gpu_arguments(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        startswith(argument, "--") || error(
            "unexpected positional argument: $argument",
        )
        body = argument[3:end]
        if occursin('=', body)
            key, value = split(body, '='; limit=2)
        else
            index == length(arguments) && error("missing value for $argument")
            key = body
            value = arguments[index + 1]
            index += 1
        end
        key = normalize_gpu_option(key)
        haskey(values, key) && error("duplicate option --$key")
        values[key] = value
        index += 1
    end

    allowed = Set((
        "input_file",
        "tolerance",
        "time_limit",
        "verbose",
        "workers",
    ))
    unknown = sort!(collect(setdiff(Set(keys(values)), allowed)))
    isempty(unknown) || error(
        "unknown option(s): " * join("--" .* unknown, ", "),
    )
    haskey(values, "input_file") || error("missing required --input_file")

    input_file = abspath(values["input_file"])
    isfile(input_file) || error("input file does not exist: $input_file")
    lower = lowercase(input_file)
    any(extension -> endswith(lower, extension), (
        ".cbf",
        ".cbf.gz",
        ".cbf.bz2",
    )) || error("input must be a .cbf, .cbf.gz, or .cbf.bz2 file")

    tolerance = parse(Float64, get(values, "tolerance", "1e-6"))
    time_limit = parse(Float64, get(values, "time_limit", "3600"))
    verbose = parse(Int, get(values, "verbose", "2"))
    workers = parse(Int, get(values, "workers", string(Threads.nthreads())))
    isfinite(tolerance) && tolerance > 0 || error(
        "--tolerance must be positive",
    )
    isfinite(time_limit) && time_limit > 0 || error(
        "--time_limit must be positive",
    )
    verbose in 0:2 || error("--verbose must be 0, 1, or 2")
    workers > 0 || error("--workers must be positive")
    return (; input_file, tolerance, time_limit, verbose, workers)
end

function gpu_optimizer_factory(options)
    return () -> begin
        optimizer = PDCS_GPU.Optimizer()
        for (name, value) in (
            "abs_tol" => options.tolerance,
            "rel_tol" => options.tolerance,
            "verbose" => options.verbose,
            "time_limit_secs" => options.time_limit,
            "max_outer_iter" => 3_000_000_000,
            "max_inner_iter" => 3_000_000_000,
        )
            MOI.set(optimizer, MOI.RawOptimizerAttribute(name), value)
        end
        return optimizer
    end
end

function solve_gpu_model(
    options;
    solve_cbf=JumpRW.solve_cbf,
    make_optimizer=gpu_optimizer_factory,
    synchronize=CUDA.synchronize,
)
    synchronize()
    started = time_ns()
    model = solve_cbf(
        options.input_file,
        make_optimizer(options);
        fast_path=:auto,
        workers=options.workers,
        add_bridges=true,
    )
    synchronize()
    return model, Float64(time_ns() - started) / 1.0e9
end

function maybe_gpu_call(function_, default)
    try
        return function_()
    catch
        return default
    end
end

function gpu_jump_read_seconds(model)
    haskey(model.ext, :JumpRW_CBF_timings) || return NaN
    return Float64(model.ext[:JumpRW_CBF_timings].total_seconds)
end

function run_gpu_solve(options)
    println("STARTED_UTC=$(now(UTC))")
    println("SOLVER=$GPU_SOLVER_NAME")
    println("CUDA_DEVICE=$(CUDA.name(CUDA.device()))")
    println("INPUT=$(options.input_file)")
    println("TOLERANCE=$(options.tolerance)")
    println("TIME_LIMIT=$(options.time_limit)")
    println("JUMPRW_WORKERS=$(options.workers)")
    flush(stdout)

    model, wall_seconds = solve_gpu_model(options)
    termination = JuMP.termination_status(model)
    primal = JuMP.primal_status(model)
    result_count = JuMP.result_count(model)
    objective = primal in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) ?
        maybe_gpu_call(() -> JuMP.objective_value(model), NaN) : NaN
    iterations = maybe_gpu_call(
        () -> MOI.get(JuMP.backend(model), PDCS_GPU.PDHGIterations()),
        missing,
    )

    println("RUN_STATUS=COMPLETE")
    println("TERMINATION_STATUS=$termination")
    println("PRIMAL_STATUS=$primal")
    println("RAW_STATUS=$(repr(maybe_gpu_call(() -> JuMP.raw_status(model), "")))")
    println("RESULT_COUNT=$result_count")
    println("OBJECTIVE_VALUE=$objective")
    println("ITERATIONS=$iterations")
    println("JUMPRW_READ_SECONDS=$(gpu_jump_read_seconds(model))")
    println("SOLVE_CBF_WALL_SECONDS=$wall_seconds")
    println("SOLVER_SECONDS=$(maybe_gpu_call(() -> JuMP.solve_time(model), NaN))")
    println("FINISHED_UTC=$(now(UTC))")
    flush(stdout)
    return 0
end

function gpu_main(arguments=ARGS)
    return run_gpu_solve(parse_gpu_arguments(arguments))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if any(argument -> argument in ("-h", "--help"), ARGS)
        gpu_usage()
        exit(0)
    end
    pushfirst!(LOAD_PATH, GPU_JUMPRW_ROOT)
    @eval using JumpRW
    @eval using JuMP
    @eval using CUDA
    @eval import MathOptInterface as MOI
    CUDA.functional() || error("CUDA is not functional for PDCS_GPU")
    include(PDCS_GPU_SOURCE)
    exit(gpu_main())
end
