#!/usr/bin/env julia

using CUDA
using JSON
using Logging
using PDCS
using Dates

import MathOptInterface as MOI

const GPU = PDCS.PDCS_GPU
const PROGRESSIVE_CONFIGURATIONS = (
    "pdhg",
    "pdhg_restart",
    "pdhg_restart_scaling",
    "pdhg_restart_scaling_reflection",
    "pdhg_restart_scaling_reflection_adaptive_primal_weight",
    "pdhg_restart_scaling_reflection_adaptive",
)
const CONFIGURATIONS = Set([
    "full",
    "no_scaling",
    "no_adaptive_step",
    "no_adaptive_primal_weight",
    "no_restart",
    "no_reflection",
    "no_halpern",
    "with_halpern_candidate",
    "without_halpern_candidate",
    PROGRESSIVE_CONFIGURATIONS...,
])


function parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        key = arguments[index]
        startswith(key, "--") || error("unexpected positional argument: $key")
        index == length(arguments) && error("missing value for $key")
        values[key[3:end]] = arguments[index + 1]
        index += 2
    end
    required = ("input-dir", "run-order", "output-dir", "instance-id")
    for name in required
        haskey(values, name) || error("missing required option --$name")
    end
    return (
        input_dir = abspath(values["input-dir"]),
        run_order = abspath(values["run-order"]),
        output_dir = abspath(values["output-dir"]),
        instance_id = values["instance-id"],
        tolerance = parse(Float64, get(values, "tolerance", "1e-6")),
        time_limit = parse(Float64, get(values, "time-limit", "600")),
        check_frequency = parse(Int, get(values, "check-frequency", "2000")),
        print_frequency = parse(Int, get(values, "print-frequency", "20000")),
        warmup = get(values, "warmup", "true") == "true",
        expected_config_count = parse(
            Int,
            get(values, "expected-config-count", "6"),
        ),
    )
end


function read_instance_tasks(run_order_path, instance_id, expected_config_count)
    lines = readlines(run_order_path)
    isempty(lines) && error("empty run-order file")
    header = split(lines[1], ',')
    positions = Dict(name => index for (index, name) in enumerate(header))
    tasks = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ',')
        fields[positions["instance_id"]] == instance_id || continue
        configuration = fields[positions["configuration"]]
        configuration in CONFIGURATIONS ||
            error("unknown configuration in run order: $configuration")
        push!(
            tasks,
            (
                task_index = parse(Int, fields[positions["task_index"]]),
                instance_id = fields[positions["instance_id"]],
                filename = fields[positions["filename"]],
                configuration = configuration,
                within_instance_order = parse(
                    Int,
                    fields[positions["within_instance_order"]],
                ),
            ),
        )
    end
    length(tasks) == expected_config_count ||
        error(
            "expected $expected_config_count tasks for $instance_id, " *
            "found $(length(tasks))",
        )
    sort!(tasks; by = task -> task.within_instance_order)
    return tasks
end


function load_cbf(path)
    model = MOI.FileFormats.CBF.Model()
    open(`gzip -dc $path`) do stream
        read!(stream, model)
    end
    return model
end


function flags(configuration)
    configuration in CONFIGURATIONS ||
        error("unknown ablation configuration: $configuration")
    if configuration in PROGRESSIVE_CONFIGURATIONS
        stage = findfirst(==(configuration), PROGRESSIVE_CONFIGURATIONS)
        return (
            use_scaling = stage >= 3,
            use_adaptive_step = stage >= 6,
            use_adaptive_step_size_weight = stage >= 5,
            use_restart = stage >= 2,
            use_reflection = stage >= 4,
            use_halpern = false,
        )
    end
    if configuration in (
        "with_halpern_candidate",
        "without_halpern_candidate",
    )
        return (
            use_scaling = true,
            use_adaptive_step = true,
            use_adaptive_step_size_weight = true,
            use_restart = true,
            use_reflection = true,
            use_halpern = configuration == "with_halpern_candidate",
        )
    end
    return (
        use_scaling = configuration != "no_scaling",
        use_adaptive_step = configuration != "no_adaptive_step",
        use_adaptive_step_size_weight =
            configuration != "no_adaptive_primal_weight",
        use_restart = configuration != "no_restart",
        use_reflection = configuration != "no_reflection",
        use_halpern = false,
    )
end


function configure!(
    optimizer,
    configuration;
    tolerance,
    time_limit,
    check_frequency,
    print_frequency,
    verbose,
)
    resolved = flags(configuration)
    attributes = (
        "verbose" => verbose,
        "time_limit_secs" => time_limit,
        "abs_tol" => tolerance,
        "rel_tol" => tolerance,
        "check_terminate_freq" => check_frequency,
        "print_freq" => print_frequency,
        "use_scaling" => resolved.use_scaling,
        "use_adaptive_restart" => resolved.use_restart,
        "use_restart" => resolved.use_restart,
        "use_adaptive_step" => resolved.use_adaptive_step,
        "use_adaptive_step_size_weight" =>
            resolved.use_adaptive_step_size_weight,
        "use_aggressive" => resolved.use_reflection,
        "use_reflection" => resolved.use_reflection,
        "use_halpern" => resolved.use_halpern,
        "use_resolving" => true,
        "use_kkt_restart" => false,
        "use_duality_gap_restart" => true,
        "logfile" => nothing,
    )
    for (name, value) in attributes
        MOI.set(optimizer, MOI.RawOptimizerAttribute(name), value)
    end
    return resolved
end


function instantiate_model(cbf_model)
    optimizer = MOI.instantiate(GPU.Optimizer; with_bridge_type = Float64)
    MOI.copy_to(optimizer, cbf_model)
    return optimizer
end


"""
Return the cuPDCS optimizer below MOI's bridge and caching layers.

`RawOptimizerAttribute("result_metrics")` contains a `NamedTuple`, not an
index-bearing MOI function. Asking `LazyBridgeOptimizer` for this attribute can
nevertheless try to reverse-map the value when a variable bridge such as
`ZerosBridge` is present. Reading it from the actual solver backend avoids that
invalid unbridging step.
"""
function pdcs_backend(optimizer)
    bridge_model = getfield(optimizer, :model)
    backend = getfield(bridge_model, :optimizer)
    backend isa GPU.Optimizer ||
        error("unexpected optimizer stack: backend has type $(typeof(backend))")
    return backend
end


function json_value(value)
    if value isa AbstractFloat && !isfinite(value)
        # JSON has no representation for NaN or +/-Inf. Preserve the record
        # itself and encode an unavailable numerical metric as `null`; the
        # analyzer already treats null as a missing/non-finite value.
        return nothing
    elseif value isa Symbol || value isa MOI.TerminationStatusCode
        return string(value)
    elseif value isa NamedTuple
        return Dict(string(key) => json_value(getfield(value, key)) for key in keys(value))
    elseif value isa AbstractDict
        return Dict(string(key) => json_value(item) for (key, item) in value)
    elseif value isa Tuple || value isa AbstractVector
        return [json_value(item) for item in value]
    end
    return value
end


function atomic_json(path, data)
    temporary = path * ".tmp"
    open(temporary, "w") do stream
        JSON.print(stream, json_value(data), 2)
        write(stream, '\n')
        flush(stream)
    end
    mv(temporary, path; force = true)
end


function next_attempt_directory(case_directory)
    mkpath(case_directory)
    numbers = Int[]
    for name in readdir(case_directory)
        match_result = match(r"^attempt_(\d+)$", name)
        match_result === nothing && continue
        push!(numbers, parse(Int, match_result.captures[1]))
    end
    number = isempty(numbers) ? 1 : maximum(numbers) + 1
    directory = joinpath(case_directory, "attempt_" * lpad(string(number), 3, '0'))
    mkpath(directory)
    return directory
end


function solve_task(cbf_model, task, options, attempt_directory)
    raw_log = joinpath(attempt_directory, "solver.raw.log")
    result_path = joinpath(attempt_directory, "result.json")
    resolved = flags(task.configuration)
    result = Dict{String,Any}(
        "task_index" => task.task_index,
        "instance_id" => task.instance_id,
        "filename" => task.filename,
        "configuration" => task.configuration,
        "within_instance_order" => task.within_instance_order,
        "tolerance" => options.tolerance,
        "time_limit_seconds" => options.time_limit,
        "resolved_flags" => Dict(
            "diagonal_rescaling" => resolved.use_scaling,
            "adaptive_step" => resolved.use_adaptive_step,
            "adaptive_step_size_weight" =>
                resolved.use_adaptive_step_size_weight,
            "adaptive_primal_weight" =>
                resolved.use_adaptive_step_size_weight,
            "restart" => resolved.use_restart,
            "reflection" => resolved.use_reflection,
            "halpern" => resolved.use_halpern,
        ),
        "started_utc" => string(now(UTC)),
        "raw_log" => raw_log,
        "julia_version" => string(VERSION),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
    )

    optimize_completed = false
    open(raw_log, "w") do stream
        logger = SimpleLogger(stream, Logging.Info)
        redirect_stdout(stream) do
            redirect_stderr(stream) do
                with_logger(logger) do
                    println("PDCS_ABLATION_RAW_LOG_VERSION=1")
                    println("INSTANCE=$(task.instance_id)")
                    println("CONFIGURATION=$(task.configuration)")
                    println("TIME_LIMIT_SECONDS=$(options.time_limit)")
                    println("TOLERANCE=$(options.tolerance)")
                    println("RESOLVED_FLAGS=$(resolved)")
                    println("STARTED_UTC=" * string(result["started_utc"]))
                    flush(stream)
                    wall_start = time()
                    try
                        optimizer = instantiate_model(cbf_model)
                        configure!(
                            optimizer,
                            task.configuration;
                            tolerance = options.tolerance,
                            time_limit = options.time_limit,
                            check_frequency = options.check_frequency,
                            print_frequency = options.print_frequency,
                            verbose = 1,
                        )
                        MOI.optimize!(optimizer)
                        CUDA.synchronize()
                        metrics = MOI.get(
                            pdcs_backend(optimizer),
                            MOI.RawOptimizerAttribute("result_metrics"),
                        )
                        termination_status = MOI.get(optimizer, MOI.TerminationStatus())
                        metric_max = maximum(
                            (
                                metrics.l_inf_rel_primal_res,
                                metrics.l_inf_rel_dual_res,
                                metrics.relative_gap,
                            ),
                        )
                        verified = isfinite(metric_max) && metric_max <= options.tolerance
                        result["termination_status"] = string(termination_status)
                        result["metrics"] = json_value(metrics)
                        result["verification_metric_max"] = metric_max
                        result["verified_solved"] = verified
                        result["wall_seconds"] = time() - wall_start
                        result["run_status"] = "COMPLETED"
                        optimize_completed = true
                        println("PDCS_ABLATION_RESULT=$(JSON.json(json_value(result)))")
                    catch error_value
                        result["wall_seconds"] = time() - wall_start
                        result["run_status"] = "RUNTIME_ERROR"
                        result["error"] = sprint(
                            showerror,
                            error_value,
                            catch_backtrace(),
                        )
                        println("PDCS_ABLATION_EXCEPTION")
                        showerror(stream, error_value, catch_backtrace())
                        println(stream)
                    end
                    result["finished_utc"] = string(now(UTC))
                    println("FINISHED_UTC=" * string(result["finished_utc"]))
                    flush(stream)
                end
            end
        end
    end
    atomic_json(result_path, result)
    open(joinpath(attempt_directory, "DONE"), "w") do stream
        println(stream, result["run_status"])
    end
    return optimize_completed, result
end


function warmup!(input_dir, output_dir)
    warmup_path = joinpath(input_dir, "bss1.cbf.gz")
    isfile(warmup_path) || (warmup_path = first(sort(readdir(input_dir; join = true))))
    log_path = joinpath(output_dir, "warmup.raw.log")
    cbf_model = load_cbf(warmup_path)
    open(log_path, "w") do stream
        logger = SimpleLogger(stream, Logging.Info)
        redirect_stdout(stream) do
            redirect_stderr(stream) do
                with_logger(logger) do
                    println("WARMUP_START $(now(UTC))")
                    optimizer = instantiate_model(cbf_model)
                    configure!(
                        optimizer,
                        "full";
                        tolerance = 1e-3,
                        time_limit = 0.1,
                        check_frequency = 100,
                        print_frequency = 100_000,
                        verbose = 0,
                    )
                    try
                        MOI.optimize!(optimizer)
                        CUDA.synchronize()
                        println("WARMUP_COMPLETE $(now(UTC))")
                    catch error_value
                        println("WARMUP_ERROR")
                        showerror(stream, error_value, catch_backtrace())
                        println(stream)
                        rethrow()
                    end
                end
            end
        end
    end
    GC.gc(true)
    CUDA.reclaim()
end


function main()
    options = parse_cli(ARGS)
    tasks = read_instance_tasks(
        options.run_order,
        options.instance_id,
        options.expected_config_count,
    )
    mkpath(options.output_dir)
    options.warmup && warmup!(options.input_dir, options.output_dir)
    cbf_path = joinpath(options.input_dir, tasks[1].filename)
    isfile(cbf_path) || error("missing CBF input: $cbf_path")
    cbf_model = load_cbf(cbf_path)

    completed = 0
    failed = 0
    for task in tasks
        case_directory = joinpath(
            options.output_dir,
            "cases",
            task.instance_id,
            task.configuration,
        )
        case_done = joinpath(case_directory, "DONE")
        if isfile(case_done)
            println(
                "SKIP_DONE instance=$(task.instance_id) config=$(task.configuration)",
            )
            continue
        end
        attempt_directory = next_attempt_directory(case_directory)
        println(
            "RUN_TASK index=$(task.task_index) instance=$(task.instance_id) " *
            "config=$(task.configuration) attempt=$(basename(attempt_directory))",
        )
        success, result = solve_task(cbf_model, task, options, attempt_directory)
        if success
            completed += 1
            open(case_done, "w") do stream
                println(stream, relpath(attempt_directory, case_directory))
            end
        else
            failed += 1
        end
        println(
            "TASK_RESULT instance=$(task.instance_id) config=$(task.configuration) " *
            "status=" * string(result["run_status"]),
        )
        GC.gc(true)
        CUDA.reclaim()
    end
    println(
        "INSTANCE_BATCH_COMPLETE instance=$(options.instance_id) " *
        "completed=$completed failed=$failed",
    )
    failed == 0 || exit(2)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
