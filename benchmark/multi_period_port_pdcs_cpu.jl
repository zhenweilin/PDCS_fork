#!/usr/bin/env julia

"""
Solve CBF files with PDCS CPU through JumpRW's public `solve_cbf` API.

Run from the JumpRW repository root with the PDCS environment:

    julia --project=external/PDCS_fork \
        external/PDCS_fork/benchmark/multi_period_port_pdcs_cpu.jl
"""

using Dates

const SOLVER_NAME = "PDCS_CPU"
const SOLVER_ALIASES = Set(("pdcscpu",))
const JUMPRW_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PDCS_CPU_SOURCE = normpath(joinpath(
    @__DIR__,
    "..",
    "src",
    "pdcs_cpu",
    "PDCS_CPU.jl",
))
const DEFAULT_INPUT_FOLDER = joinpath(JUMPRW_ROOT, "test_data", "small_scale")
const DEFAULT_OUTPUT_FOLDER = joinpath(
    @__DIR__,
    "results",
    "small_scale",
    "pdcs_cpu",
)
const CBF_EXTENSIONS = (".cbf", ".cbf.gz", ".cbf.bz2")

function usage(io::IO=stdout)
    println(io, "Usage:")
    println(
        io,
        "  julia --project=external/PDCS_fork " *
        "external/PDCS_fork/benchmark/multi_period_port_pdcs_cpu.jl [options]",
    )
    println(io)
    println(io, "Options:")
    println(io, "  --input_folder PATH   CBF folder (default: test_data/small_scale)")
    println(io, "  --output_folder PATH  Raw-log folder")
    println(io, "  --solver PDCS_CPU     Optional solver-name validation")
    println(io, "  --tolerance VALUE     Absolute/relative tolerance (default: 1e-6)")
    println(io, "  --time_limit SECONDS  Optional per-instance time limit")
    println(io, "  --verbose LEVEL       0, 1, or 2 (default: 2)")
    println(io, "  --workers COUNT       JumpRW reader workers")
    println(io, "  -h, --help            Show this help")
end

normalize_option(name) = replace(name, '-' => '_')
normalize_solver(name) = lowercase(replace(name, r"[-_]" => ""))

function parse_arguments(arguments)
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
        key = normalize_option(key)
        haskey(values, key) && error("duplicate option --$key")
        values[key] = value
        index += 1
    end

    allowed = Set((
        "input_folder",
        "output_folder",
        "solver",
        "tolerance",
        "time_limit",
        "verbose",
        "workers",
    ))
    unknown = sort!(collect(setdiff(Set(keys(values)), allowed)))
    isempty(unknown) || error(
        "unknown option(s): " * join("--" .* unknown, ", "),
    )

    if haskey(values, "solver")
        normalize_solver(values["solver"]) in SOLVER_ALIASES || error(
            "this script only supports --solver PDCS_CPU",
        )
    end

    input_folder = abspath(get(values, "input_folder", DEFAULT_INPUT_FOLDER))
    isdir(input_folder) || error("input folder does not exist: $input_folder")
    output_folder = abspath(get(values, "output_folder", DEFAULT_OUTPUT_FOLDER))
    tolerance = parse(Float64, get(values, "tolerance", "1e-6"))
    time_limit = haskey(values, "time_limit") ?
        parse(Float64, values["time_limit"]) : nothing
    verbose = parse(Int, get(values, "verbose", "2"))
    workers = parse(Int, get(values, "workers", string(Threads.nthreads())))

    isfinite(tolerance) && tolerance > 0 || error(
        "--tolerance must be positive",
    )
    isnothing(time_limit) ||
        (isfinite(time_limit) && time_limit > 0) ||
        error("--time_limit must be positive")
    verbose in 0:2 || error("--verbose must be 0, 1, or 2")
    workers > 0 || error("--workers must be positive")

    return (
        ;
        input_folder,
        output_folder,
        tolerance,
        time_limit,
        verbose,
        workers,
    )
end

function cbf_files(folder)
    files = filter(readdir(folder; join=true)) do path
        lower = lowercase(basename(path))
        return isfile(path) && any(
            extension -> endswith(lower, extension),
            CBF_EXTENSIONS,
        )
    end
    sort!(files; by=path -> lowercase(basename(path)))
    isempty(files) && error("no supported CBF files found in $folder")
    return files
end

function log_path(output_folder, input_path)
    stem = replace(
        basename(input_path),
        r"(?i)\.cbf(?:\.gz|\.bz2)?$" => "",
    )
    return joinpath(output_folder, stem * ".raw.log")
end

function optimizer_factory(options)
    return () -> begin
        optimizer = PDCS_CPU.Optimizer()
        attributes = Pair{String,Any}[
            "abs_tol" => options.tolerance,
            "rel_tol" => options.tolerance,
            "verbose" => options.verbose,
            "max_outer_iter" => 3_000_000_000,
            "max_inner_iter" => 3_000_000_000,
        ]
        isnothing(options.time_limit) || push!(
            attributes,
            "time_limit_secs" => options.time_limit,
        )
        for (name, value) in attributes
            MOI.set(optimizer, MOI.RawOptimizerAttribute(name), value)
        end
        return optimizer
    end
end

function solve_model(
    path,
    options;
    solve_cbf=JumpRW.solve_cbf,
    make_optimizer=optimizer_factory,
)
    started = time_ns()
    model = solve_cbf(
        path,
        make_optimizer(options);
        fast_path=:auto,
        workers=options.workers,
        add_bridges=true,
    )
    return model, (time_ns() - started) / 1e9
end

function maybe_call(function_, default)
    try
        return function_()
    catch
        return default
    end
end

function jump_read_seconds(model)
    haskey(model.ext, :JumpRW_CBF_timings) || return NaN
    timings = model.ext[:JumpRW_CBF_timings]
    return Float64(getproperty(timings, :total_seconds))
end

function solve_one(path, options)
    println("STARTED_UTC=$(now(UTC))")
    println("SOLVER=$SOLVER_NAME")
    println("INPUT=$(abspath(path))")
    println("TOLERANCE=$(options.tolerance)")
    println("TIME_LIMIT=$(something(options.time_limit, "solver_default"))")
    println("JUMPRW_WORKERS=$(options.workers)")
    flush(stdout)

    model, wall_seconds = solve_model(path, options)
    termination = JuMP.termination_status(model)
    primal = JuMP.primal_status(model)
    result_count = JuMP.result_count(model)
    objective = primal in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) ?
        maybe_call(() -> JuMP.objective_value(model), NaN) : NaN
    raw_status = maybe_call(() -> JuMP.raw_status(model), "")
    solver_seconds = maybe_call(() -> JuMP.solve_time(model), NaN)
    iterations = maybe_call(
        () -> MOI.get(JuMP.backend(model), PDCS_CPU.PDHGIterations()),
        missing,
    )

    println("RUN_STATUS=COMPLETE")
    println("TERMINATION_STATUS=$termination")
    println("PRIMAL_STATUS=$primal")
    println("RAW_STATUS=$(repr(raw_status))")
    println("RESULT_COUNT=$result_count")
    println("OBJECTIVE_VALUE=$objective")
    println("ITERATIONS=$iterations")
    println("JUMPRW_READ_SECONDS=$(jump_read_seconds(model))")
    println("SOLVE_CBF_WALL_SECONDS=$wall_seconds")
    println("SOLVER_SECONDS=$solver_seconds")
    println("FINISHED_UTC=$(now(UTC))")
    flush(stdout)
    return (
        ;
        ok=true,
        termination=string(termination),
        objective,
        solve_wall_seconds=wall_seconds,
        error="",
    )
end

function solve_with_log(path, options)
    raw_log = log_path(options.output_folder, path)
    result = open(raw_log, "w") do stream
        redirect_stdout(stream) do
            redirect_stderr(stream) do
                try
                    solve_one(path, options)
                catch error_value
                    error_value isa InterruptException && rethrow()
                    backtrace = catch_backtrace()
                    message = sprint(showerror, error_value)
                    println("RUN_STATUS=EXCEPTION")
                    println("ERROR=$(repr(message))")
                    println("FINISHED_UTC=$(now(UTC))")
                    showerror(stderr, error_value, backtrace)
                    println(stderr)
                    flush(stream)
                    (
                        ;
                        ok=false,
                        termination="EXCEPTION",
                        objective=NaN,
                        solve_wall_seconds=NaN,
                        error=message,
                    )
                end
            end
        end
    end
    return result, raw_log
end

function main(arguments=ARGS)
    options = parse_arguments(arguments)
    files = cbf_files(options.input_folder)
    mkpath(options.output_folder)
    println(
        "BATCH_START solver=$SOLVER_NAME instances=$(length(files)) " *
        "input=$(options.input_folder) logs=$(options.output_folder)",
    )

    completed = 0
    failed = 0
    for (index, path) in enumerate(files)
        println("INSTANCE_START index=$index file=$(repr(basename(path)))")
        result, raw_log = solve_with_log(path, options)
        if result.ok
            completed += 1
            println(
                "INSTANCE_COMPLETE index=$index " *
                "file=$(repr(basename(path))) " *
                "status=$(result.termination) " *
                "objective=$(result.objective) " *
                "solve_wall_seconds=$(result.solve_wall_seconds) " *
                "log=$raw_log",
            )
        else
            failed += 1
            message = replace(result.error, '\n' => ' ')
            println(
                stderr,
                "INSTANCE_FAILED index=$index " *
                "file=$(repr(basename(path))) " *
                "error=$(repr(message)) log=$raw_log",
            )
        end
        GC.gc(true)
    end
    println(
        "BATCH_COMPLETE solver=$SOLVER_NAME " *
        "completed=$completed failed=$failed",
    )
    return failed == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    if any(argument -> argument in ("-h", "--help"), ARGS)
        usage()
        exit(0)
    end
    pushfirst!(LOAD_PATH, JUMPRW_ROOT)
    @eval using JumpRW
    @eval using JuMP
    @eval import MathOptInterface as MOI
    # The PDCS package entry point eagerly loads PDCS_GPU, which requires a
    # CUDA driver even for CPU-only runs. Load the CPU submodule directly.
    include(PDCS_CPU_SOURCE)
    exit(main())
end
