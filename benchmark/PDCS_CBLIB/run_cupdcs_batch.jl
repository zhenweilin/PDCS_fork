#!/usr/bin/env julia

using CUDA
using Dates
using JuMP
using Logging
using SHA
using TOML

import MathOptInterface as MOI

const BENCHMARK_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_DIR, "..", ".."))

function parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        key = arguments[index]
        startswith(key, "--") || error("unexpected positional argument: $key")
        index == length(arguments) && error("missing value for $key")
        haskey(values, key) && error("duplicate option: $key")
        values[key] = arguments[index + 1]
        index += 2
    end
    for key in ("--input-root", "--case-list", "--result-root", "--expected-cases")
        haskey(values, key) || error("missing required option $key")
    end
    tolerance = parse(Float64, get(values, "--tolerance", "1e-6"))
    tolerance == 1.0e-6 || error("cuPDCS CBF tolerance must be exactly 1e-6")
    time_limit = parse(Float64, get(values, "--time-limit", "3600"))
    time_limit == 3600.0 || error("each CBF time limit must be exactly 3600 seconds")
    print_frequency = parse(Int, get(values, "--print-frequency", "10000"))
    verbose = parse(Int, get(values, "--verbose", "2"))
    print_frequency > 0 || error("--print-frequency must be positive")
    verbose in 0:2 || error("--verbose must be 0, 1, or 2")
    return (
        input_root = abspath(values["--input-root"]),
        case_list = abspath(values["--case-list"]),
        result_root = abspath(values["--result-root"]),
        expected_cases = parse(Int, values["--expected-cases"]),
        tolerance,
        time_limit,
        print_frequency,
        verbose,
        required_julia = get(values, "--required-julia", "1.10.4"),
        rerun = lowercase(get(values, "--rerun", "false")) in ("1", "true", "yes"),
    )
end

function atomic_toml(path::AbstractString, value::AbstractDict)
    mkpath(dirname(path))
    temporary, stream = mktemp(dirname(path))
    try
        TOML.print(stream, value; sorted = true)
        close(stream)
        mv(temporary, path; force = true)
    catch
        isopen(stream) && close(stream)
        isfile(temporary) && rm(temporary)
        rethrow()
    end
    return path
end

function git_commit()
    try
        return strip(read(`git -C $REPO_ROOT rev-parse HEAD`, String))
    catch
        return "unavailable"
    end
end

function input_sha256(path)
    return bytes2hex(open(sha256, path))
end

function case_stem(relative_path)
    lower = lowercase(relative_path)
    for suffix in (".cbf.gz", ".cbf.bz2", ".cbf")
        if endswith(lower, suffix)
            return relative_path[1:(end - length(suffix))]
        end
    end
    error("unsupported CBF extension: $relative_path")
end

function result_paths(options, relative_path)
    stem = case_stem(relative_path)
    case_dir = joinpath(options.result_root, "cupdcs", "cases", stem)
    return (
        result = joinpath(case_dir, "result.toml"),
        log = joinpath(options.result_root, "cupdcs", "logs", stem * ".log"),
    )
end

function read_cases(options)
    isdir(options.input_root) || error("input root does not exist: $(options.input_root)")
    isfile(options.case_list) || error("case list does not exist: $(options.case_list)")
    cases = String[]
    for raw in readlines(options.case_list)
        relative_path = strip(raw)
        (isempty(relative_path) || startswith(relative_path, '#')) && continue
        isabspath(relative_path) && error("case path must be relative: $relative_path")
        normpath(relative_path) == relative_path || error("non-normal case path: $relative_path")
        full_path = joinpath(options.input_root, relative_path)
        isfile(full_path) || error("listed CBF does not exist: $full_path")
        push!(cases, relative_path)
    end
    length(unique(cases)) == length(cases) || error("case list contains duplicates")
    length(cases) == options.expected_cases || error(
        "expected $(options.expected_cases) cases, found $(length(cases))",
    )
    return cases
end

function set_cupdcs_attributes!(model, options)
    for (name, value) in (
        "abs_tol" => options.tolerance,
        "rel_tol" => options.tolerance,
        "time_limit_secs" => options.time_limit,
        "max_outer_iter" => 3_000_000_000,
        "max_inner_iter" => 3_000_000_000,
        "check_terminate_freq" => 1_000,
        "print_freq" => options.print_frequency,
        "verbose" => options.verbose,
        "sparse_index_type" => :auto,
        "use_scaling" => true,
        "rescaling_method" => :ruiz_pock_chambolle,
        "use_adaptive_restart" => true,
        "use_restart" => true,
        "use_adaptive_step" => true,
        "use_adaptive_step_size_weight" => true,
        "use_aggressive" => true,
        "use_reflection" => true,
        "use_resolving" => true,
        "use_accelerated" => false,
        "use_halpern" => false,
        "use_kkt_restart" => false,
        "use_duality_gap_restart" => true,
        "logfile" => nothing,
    )
        JuMP.set_optimizer_attribute(model, name, value)
    end
    return model
end

function safe_float(value)
    try
        return Float64(value)
    catch
        return NaN
    end
end

function solve_one!(options, relative_path, gpu_name, commit, runtime_status)
    paths = result_paths(options, relative_path)
    if isfile(paths.result) && !options.rerun
        prior_status = try
            String(get(TOML.parsefile(paths.result), "run_status", ""))
        catch
            ""
        end
        if prior_status in ("passed", "failed")
            return prior_status, true
        end
    end

    input_path = joinpath(options.input_root, relative_path)
    mkpath(dirname(paths.log))
    started = now(UTC)
    result = Dict{String,Any}(
        "schema_version" => 1,
        "solver" => "cupdcs",
        "modeling_interface" => "JuMP.read_from_file",
        "case" => relative_path,
        "input" => input_path,
        "input_sha256" => input_sha256(input_path),
        "git_commit" => commit,
        "gpu_name" => gpu_name,
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "tolerance" => options.tolerance,
        "time_limit_seconds" => options.time_limit,
        "started_utc" => string(started),
        "run_status" => "runtime_error",
        "termination_status" => "EXCEPTION",
        "status_accepted" => false,
        "solver_tolerance_accepted" => false,
        "gridwise_runtime" => repr(runtime_status),
        "solver_log" => paths.log,
    )
    wall_started = time()
    model = nothing
    open(paths.log, "w") do stream
        logger = ConsoleLogger(stream, Logging.Info)
        redirect_stdout(stream) do
            redirect_stderr(stream) do
                with_logger(logger) do
                    println("CUPDCS_CBF_LOG_VERSION=1")
                    println("MODELING_INTERFACE=JuMP.read_from_file")
                    println("CASE=$relative_path")
                    println("INPUT=$input_path")
                    println("TOLERANCE=$(options.tolerance)")
                    println("TIME_LIMIT_SECONDS=$(options.time_limit)")
                    println("GPU_NAME=$gpu_name")
                    println("JULIA_VERSION=$(VERSION)")
                    println("GRIDWISE_RUNTIME_BEFORE=$(repr(runtime_status))")
                    println("CUDA_MEMORY_STATUS_BEFORE")
                    CUDA.memory_status()
                    flush(stream)
                    try
                        read_started = time()
                        model = JuMP.read_from_file(input_path)
                        result["read_seconds"] = time() - read_started
                        result["num_variables"] = JuMP.num_variables(model)
                        result["num_constraints"] = JuMP.num_constraints(
                            model;
                            count_variable_in_set_constraints = true,
                        )
                        JuMP.set_optimizer(
                            model,
                            PDCS_GPU.Optimizer;
                            add_bridges = true,
                        )
                        set_cupdcs_attributes!(model, options)
                        solve_started = time()
                        JuMP.optimize!(model)
                        CUDA.synchronize()
                        result["solve_wall_seconds"] = time() - solve_started
                        termination = JuMP.termination_status(model)
                        result["termination_status"] = string(termination)
                        result["raw_status"] = JuMP.raw_status(model)
                        result["primal_status"] = string(JuMP.primal_status(model))
                        result["dual_status"] = string(JuMP.dual_status(model))
                        result["result_count"] = JuMP.result_count(model)
                        result["solver_seconds"] = safe_float(JuMP.solve_time(model))
                        metrics = MOI.get(
                            JuMP.unsafe_backend(model),
                            MOI.RawOptimizerAttribute("result_metrics"),
                        )
                        for name in propertynames(metrics)
                            value = getproperty(metrics, name)
                            if value isa Real || value isa AbstractString || value isa Bool
                                result[string(name)] = value
                            end
                        end
                        metric_values = abs.(Float64[
                            metrics.l_inf_rel_primal_res,
                            metrics.l_inf_rel_dual_res,
                            metrics.relative_gap,
                        ])
                        result["solver_relative_kkt_max"] = maximum(metric_values)
                        result["status_accepted"] = termination == MOI.OPTIMAL
                        result["solver_tolerance_accepted"] =
                            all(isfinite, metric_values) &&
                            maximum(metric_values) <= options.tolerance
                        result["run_status"] =
                            result["status_accepted"] &&
                            result["solver_tolerance_accepted"] ? "passed" : "failed"
                        if JuMP.has_values(model)
                            result["objective_value"] = safe_float(JuMP.objective_value(model))
                        end
                        println(
                            "CUPDCS_CBF_RESULT case=$relative_path " *
                            "status=$(result["termination_status"]) " *
                            "run_status=$(result["run_status"]) " *
                            "kkt=$(result["solver_relative_kkt_max"]) " *
                            "solve_seconds=$(result["solver_seconds"])",
                        )
                    catch error_value
                        result["error"] = sprint(
                            showerror,
                            error_value,
                            catch_backtrace(),
                        )
                        println("CUPDCS_CBF_EXCEPTION case=$relative_path")
                        showerror(stream, error_value, catch_backtrace())
                        println(stream)
                    finally
                        try
                            CUDA.synchronize()
                        catch
                        end
                        println("CUDA_MEMORY_STATUS_AFTER")
                        try
                            CUDA.memory_status()
                        catch error_value
                            println("CUDA_MEMORY_STATUS_ERROR=$(sprint(showerror, error_value))")
                        end
                        println("GRIDWISE_RUNTIME_AFTER=$(repr(PDCS_GPU.gridWise_runtime_status()))")
                        flush(stream)
                    end
                end
            end
        end
    end
    result["elapsed_wall_seconds"] = time() - wall_started
    result["finished_utc"] = string(now(UTC))
    atomic_toml(paths.result, result)
    model = nothing
    GC.gc(true)
    try
        CUDA.reclaim()
    catch
    end
    return result["run_status"], false
end

function main()
    options = parse_cli(ARGS)
    string(VERSION) == options.required_julia || error(
        "required Julia $(options.required_julia), got $VERSION",
    )
    CUDA.functional() || error("CUDA is not functional")
    gpu_name = CUDA.name(CUDA.device())
    occursin("H100", gpu_name) || error("expected H100, found $gpu_name")
    if PDCS_GPU.few_block_proj_ptr[] == C_NULL
        PDCS_GPU.__init__()
    end
    runtime_status = PDCS_GPU.check_gridWise_runtime!()
    runtime_status.state == :passed || error(
        "strict grid-wise runtime self-test did not pass: $runtime_status",
    )
    cases = read_cases(options)
    mkpath(options.result_root)
    commit = git_commit()
    counts = Dict("passed" => 0, "failed" => 0, "runtime_error" => 0, "existing" => 0)
    batch_started = now(UTC)
    println(
        "CUPDCS_CBF_BATCH_START cases=$(length(cases)) gpu=$gpu_name " *
        "tolerance=$(options.tolerance) per_case_time_limit=$(options.time_limit) " *
        "julia=$(VERSION) utc=$batch_started",
    )
    for (index, relative_path) in enumerate(cases)
        println("CUPDCS_CBF_CASE_START index=$index total=$(length(cases)) case=$relative_path")
        flush(stdout)
        status, skipped = solve_one!(
            options,
            relative_path,
            gpu_name,
            commit,
            runtime_status,
        )
        key = skipped ? "existing" :
            (haskey(counts, status) ? status : "runtime_error")
        counts[key] = get(counts, key, 0) + 1
        println(
            "CUPDCS_CBF_CASE_FINISH index=$index total=$(length(cases)) " *
            "case=$relative_path status=$status skipped=$skipped",
        )
        flush(stdout)
    end
    summary = Dict{String,Any}(
        "schema_version" => 1,
        "solver" => "cupdcs",
        "modeling_interface" => "JuMP.read_from_file",
        "expected_cases" => options.expected_cases,
        "listed_cases" => length(cases),
        "passed" => counts["passed"],
        "failed" => counts["failed"],
        "runtime_error" => counts["runtime_error"],
        "existing" => counts["existing"],
        "tolerance" => options.tolerance,
        "per_case_time_limit_seconds" => options.time_limit,
        "gpu_name" => gpu_name,
        "julia_version" => string(VERSION),
        "git_commit" => commit,
        "started_utc" => string(batch_started),
        "finished_utc" => string(now(UTC)),
        "complete" => true,
    )
    atomic_toml(joinpath(options.result_root, "campaign_summary.toml"), summary)
    println("CUPDCS_CBF_BATCH_FINISH summary=$(joinpath(options.result_root, "campaign_summary.toml"))")
    return 0
end

Base.include(Main, joinpath(REPO_ROOT, "src", "pdcs_gpu", "PDCS_GPU.jl"))
exit(main())
