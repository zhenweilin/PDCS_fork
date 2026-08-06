#!/usr/bin/env julia

"""
Solve every `.cbf.gz` file in a folder with cuPDCS.

Run from the repository root with the main PDCS environment:

    julia --project=. benchmark/multi_period_port_cupdcs.jl \
        --input_folder /path/to/cbf_files

Each instance gets an isolated raw log. Use `--output_folder` to select the
log directory. `--solver cuPDCS` is accepted for compatibility with the
multi-solver command line interface.
"""

using Dates

const SOLVER_NAME = "cuPDCS"
const SOLVER_ALIASES = Set(("cupdcs",))

function usage(io::IO = stdout)
    println(io, "Usage:")
    println(
        io,
        "  julia --project=. benchmark/multi_period_port_cupdcs.jl " *
        "--input_folder PATH [options]",
    )
    println(io)
    println(io, "Options:")
    println(io, "  --input_folder PATH   Folder containing .cbf.gz files (required)")
    println(io, "  --output_folder PATH  Raw-log folder")
    println(io, "  --solver cuPDCS       Optional solver-name validation")
    println(io, "  --tolerance VALUE     Absolute and relative tolerance (default: 1e-6)")
    println(io, "  --time_limit SECONDS  Optional per-instance solver time limit")
    println(io, "  --verbose LEVEL       Solver verbosity: 0, 1, or 2 (default: 2)")
    println(io, "  -h, --help            Show this help")
end

normalize_option(name) = replace(name, '-' => '_')
normalize_solver(name) = lowercase(replace(name, r"[-_]" => ""))

function parse_arguments(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("-h", "--help")
            usage()
            exit(0)
        end
        startswith(argument, "--") ||
            error("unexpected positional argument: $argument")
        body = argument[3:end]
        if occursin('=', body)
            key, value = split(body, '='; limit = 2)
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
    ))
    unknown = sort!(collect(setdiff(Set(keys(values)), allowed)))
    isempty(unknown) || error("unknown option(s): " * join("--" .* unknown, ", "))
    haskey(values, "input_folder") || error("missing required option --input_folder")

    if haskey(values, "solver")
        normalize_solver(values["solver"]) in SOLVER_ALIASES ||
            error("this script only supports --solver cuPDCS")
    end

    input_folder = abspath(values["input_folder"])
    isdir(input_folder) || error("input folder does not exist: $input_folder")
    output_folder = abspath(get(
        values,
        "output_folder",
        joinpath(@__DIR__, "results", "multi_period_port", "cupdcs"),
    ))
    tolerance = parse(Float64, get(values, "tolerance", "1e-6"))
    time_limit = haskey(values, "time_limit") ?
        parse(Float64, values["time_limit"]) : nothing
    verbose = parse(Int, get(values, "verbose", "2"))

    isfinite(tolerance) && tolerance > 0 || error("--tolerance must be positive")
    isnothing(time_limit) ||
        (isfinite(time_limit) && time_limit > 0) ||
        error("--time_limit must be positive")
    verbose in 0:2 || error("--verbose must be 0, 1, or 2")
    return (
        input_folder = input_folder,
        output_folder = output_folder,
        tolerance = tolerance,
        time_limit = time_limit,
        verbose = verbose,
    )
end

function cbf_files(folder)
    files = filter(
        path -> isfile(path) && endswith(lowercase(basename(path)), ".cbf.gz"),
        readdir(folder; join = true),
    )
    sort!(files; by = path -> lowercase(basename(path)))
    isempty(files) && error("no .cbf.gz files found in $folder")
    return files
end

function load_cbf(path)
    Sys.which("gzip") === nothing && error("gzip executable was not found")
    # model = MOI.FileFormats.CBF.Model()
    # open(`gzip -dc -- $path`) do stream
    #     read!(stream, model)
    # end
    model = MOI.Utilities.Model{Float64}()
    println("begin to load model")
    flush(stdout)
    MOI.read_from_file(model, path)
    println("has loaded model")
    flush(stdout)
    return model
end

function build_optimizer(options)
    optimizer = MOI.instantiate(PDCS_GPU.Optimizer; with_bridge_type = Float64)
    attributes = Pair{String,Any}[
        "abs_tol" => options.tolerance,
        "rel_tol" => options.tolerance,
        "verbose" => options.verbose,
        # Effectively unlimited: the time limit governs termination.
        # Product must fit in Int64 to avoid overflow in termination check.
        "max_outer_iter" => 3_000_000_000,
        "max_inner_iter" => 3_000_000_000,
    ]
    if !isnothing(options.time_limit)
        push!(attributes, "time_limit_secs" => options.time_limit)
    end
    for (name, value) in attributes
        MOI.set(optimizer, MOI.RawOptimizerAttribute(name), value)
    end
    return optimizer
end

function maybe_get(optimizer, attribute, default)
    try
        return MOI.get(optimizer, attribute)
    catch
        return default
    end
end

function solve_one(path, options)
    println("STARTED_UTC=$(now(UTC))")
    println("SOLVER=$SOLVER_NAME")
    println("INPUT=$(abspath(path))")
    println("TOLERANCE=$(options.tolerance)")
    println("TIME_LIMIT=$(something(options.time_limit, "solver_default"))")
    println("CUDA_DEVICE=$(CUDA.name(CUDA.device()))")
    flush(stdout)

    load_started = time()
    cbf_model = load_cbf(path)
    load_seconds = time() - load_started

    setup_started = time()
    optimizer = build_optimizer(options)
    MOI.copy_to(optimizer, cbf_model)
    setup_seconds = time() - setup_started

    CUDA.synchronize()
    solve_started = time()
    MOI.optimize!(optimizer)
    CUDA.synchronize()
    solve_wall_seconds = time() - solve_started

    termination = maybe_get(optimizer, MOI.TerminationStatus(), "UNKNOWN")
    raw_status = maybe_get(optimizer, MOI.RawStatusString(), "")
    result_count = maybe_get(optimizer, MOI.ResultCount(), 0)
    objective = result_count > 0 ?
        maybe_get(optimizer, MOI.ObjectiveValue(), NaN) : NaN
    solver_seconds = maybe_get(optimizer, MOI.SolveTimeSec(), NaN)
    iterations = maybe_get(optimizer, PDCS_GPU.PDHGIterations(), missing)

    println("TERMINATION_STATUS=$termination")
    println("RAW_STATUS=$(repr(raw_status))")
    println("RESULT_COUNT=$result_count")
    println("OBJECTIVE_VALUE=$objective")
    println("ITERATIONS=$iterations")
    println("LOAD_SECONDS=$load_seconds")
    println("SETUP_SECONDS=$setup_seconds")
    println("SOLVE_WALL_SECONDS=$solve_wall_seconds")
    println("SOLVER_SECONDS=$solver_seconds")
    println("FINISHED_UTC=$(now(UTC))")
    flush(stdout)
    return (
        ok = true,
        termination = string(termination),
        objective = objective,
        solve_wall_seconds = solve_wall_seconds,
        error = "",
    )
end

function log_path(output_folder, input_path)
    stem = replace(basename(input_path), r"(?i)\.cbf\.gz$" => "")
    return joinpath(output_folder, stem * ".raw.log")
end

function solve_with_log(path, options)
    raw_log = log_path(options.output_folder, path)
    result = open(raw_log, "w") do stream
        redirect_stdout(stream) do
            redirect_stderr(stream) do
                try
                    solve_one(path, options)
                catch error_value
                    backtrace = catch_backtrace()
                    message = sprint(showerror, error_value)
                    println("RUN_STATUS=EXCEPTION")
                    println("ERROR=$(repr(message))")
                    println("FINISHED_UTC=$(now(UTC))")
                    showerror(stderr, error_value, backtrace)
                    println(stderr)
                    flush(stream)
                    (
                        ok = false,
                        termination = "EXCEPTION",
                        objective = NaN,
                        solve_wall_seconds = NaN,
                        error = message,
                    )
                end
            end
        end
    end
    return result, raw_log
end

function main()
    options = parse_arguments(ARGS)
    files = cbf_files(options.input_folder)
    CUDA.functional() || error("CUDA is not functional for cuPDCS")
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
                "INSTANCE_COMPLETE index=$index file=$(repr(basename(path))) " *
                "status=$(result.termination) objective=$(result.objective) " *
                "solve_wall_seconds=$(result.solve_wall_seconds) log=$raw_log",
            )
        else
            failed += 1
            message = replace(result.error, '\n' => ' ')
            println(stderr,
                "INSTANCE_FAILED index=$index file=$(repr(basename(path))) " *
                "error=$(repr(message)) log=$raw_log",
            )
        end
        GC.gc(true)
        CUDA.reclaim()
    end
    println("BATCH_COMPLETE solver=$SOLVER_NAME completed=$completed failed=$failed")
    failed == 0 || exit(1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if any(argument -> argument in ("-h", "--help"), ARGS)
        usage()
        exit(0)
    end
    @eval using CUDA
    @eval using PDCS: PDCS_GPU
    @eval import MathOptInterface as MOI
    main()
end
