#!/usr/bin/env julia

"""
Read every `.cbf.gz` file in a folder with JumpRW and solve it with cuPDCS.

Run from the repository root with the main PDCS environment:

    julia --threads=auto --project=. \
        benchmark/multi_period_port_cupdcs_parallel.jl \
        --input_folder /path/to/cbf_files

Each instance gets an isolated raw log. Use `--output_folder` to select the
log directory. `--solver cuPDCS` is accepted for compatibility with the
multi-solver command line interface.
"""

using Dates

const SOLVER_NAME = "cuPDCS"
const SOLVER_ALIASES = Set(("cupdcs",))
const JUMPRW_ROOT = normpath(joinpath(@__DIR__, "..", "external", "Jump_RW"))

function usage(io::IO = stdout)
    println(io, "Usage:")
    println(
        io,
        "  julia --threads=auto --project=. " *
        "benchmark/multi_period_port_cupdcs_parallel.jl " *
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
    println(io, "  --workers COUNT       JumpRW parser workers (default: Julia threads)")
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
        "workers",
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
    workers = parse(Int, get(values, "workers", string(Threads.nthreads())))

    isfinite(tolerance) && tolerance > 0 || error("--tolerance must be positive")
    isnothing(time_limit) ||
        (isfinite(time_limit) && time_limit > 0) ||
        error("--time_limit must be positive")
    verbose in 0:2 || error("--verbose must be 0, 1, or 2")
    workers > 0 || error("--workers must be positive")
    return (
        input_folder = input_folder,
        output_folder = output_folder,
        tolerance = tolerance,
        time_limit = time_limit,
        verbose = verbose,
        workers = workers,
    )
end

function cbf_files(folder)
    files = filter(
        path -> isfile(path) && endswith(lowercase(basename(path)), ".cbf.gz"),
        readdir(folder; join = true),
    )
    # Sort by problem size (ascending): extract the numeric field before "_seed" in
    # filenames like "5_3_..._planning_63_seed2026.cbf.gz".
    function extract_size(path)
        stem = basename(path)
        m = match(r"_(\d+)_seed\d+\.cbf\.gz$", stem)
        return m === nothing ? 0 : parse(Int, m.captures[1])
    end
    sort!(files; by = extract_size)
    isempty(files) && error("no .cbf.gz files found in $folder")
    return files
end

function build_model_conic(path, options)
    # Fast path: read raw conic data in parallel (no JuMP incremental build),
    # then construct the model with a pre-populated optimizer cache.
    println("begin JumpRW.read_cbf_conic_data (parallel parse, no JuMP build)...")
    flush(stdout)
    data = JumpRW.read_cbf_conic_data(path; workers=options.workers, coalesce=true)
    println("conic data loaded: n=$(data.num_variables) m=$(data.num_rows)")
    flush(stdout)

    optimizer = PDCS_GPU.Optimizer()
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

    model = PDCS_GPU.model_from_conic_data(data; optimizer=optimizer)
    println("model_from_conic_data done (cache pre-built, copy_to bypassed)")
    flush(stdout)
    return model
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
    println("JUMPRW_WORKERS=$(options.workers)")
    println("CUDA_DEVICE=$(CUDA.name(CUDA.device()))")
    flush(stdout)

    load_started = time()
    model = build_model_conic(path, options)
    load_seconds = time() - load_started
    println("JUMPRW_LOAD_AND_BUILD_SECONDS=$load_seconds")
    flush(stdout)

    CUDA.synchronize()
    solve_started = time()
    println("begin JuMP.optimize! (direct dispatch, no copy_to for conic data)")
    flush(stdout)
    JuMP.optimize!(model)
    CUDA.synchronize()
    solve_wall_seconds = time() - solve_started
    println("SOLVE_WALL_SECONDS=$solve_wall_seconds")
    flush(stdout)

    backend = JuMP.backend(model)
    termination = maybe_get(backend, MOI.TerminationStatus(), "UNKNOWN")
    raw_status = maybe_get(backend, MOI.RawStatusString(), "")
    result_count = maybe_get(backend, MOI.ResultCount(), 0)
    objective = result_count > 0 ?
        maybe_get(backend, MOI.ObjectiveValue(), NaN) : NaN
    solver_seconds = maybe_get(backend, MOI.SolveTimeSec(), NaN)
    iterations = maybe_get(backend, PDCS_GPU.PDHGIterations(), missing)
    copy_seconds = solve_wall_seconds - solver_seconds  # wall − solver time = copy overhead

    println("TERMINATION_STATUS=$termination")
    println("RAW_STATUS=$(repr(raw_status))")
    println("RESULT_COUNT=$result_count")
    println("OBJECTIVE_VALUE=$objective")
    println("ITERATIONS=$iterations")
    println("JUMPRW_LOAD_SETUP_SECONDS=$load_seconds")
    println("SOLVE_WALL_SECONDS=$solve_wall_seconds")
    println("SOLVER_SECONDS=$solver_seconds")
    println("COPY_SECONDS=$copy_seconds")
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
    @eval using PDCS
    # Load GPU module directly (extension auto-load may fail with CUDA version drift).
    include(joinpath(pkgdir(PDCS), "src", "pdcs_gpu", "PDCS_GPU.jl"))
    PDCS._register_gpu!(PDCS_GPU)
    @eval using JuMP
    @eval import MathOptInterface as MOI
    JUMPRW_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, JUMPRW_ROOT)
    @eval using JumpRW
    main()
end
