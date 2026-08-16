#!/usr/bin/env julia

"""
Run the R3.5 diagonal/scalar-cone rescaling ablation on a shard of the fixed
CBF corpus in `../PDCS_fork/benchmark/represent_data`.

All 62 files are formal instances. Four representative files are also used for
an untimed warm-up before the formal runs. Every instance is solved in both
modes, with a stable counterbalanced mode order derived from the filename and
`--order-seed`.

The per-solve limit defaults to 3,600 seconds and remains user-selectable with
`--time-limit SECONDS`.
"""

using CUDA
using Dates
using JSON
using Logging
using PDCS
using SHA

import MathOptInterface as MOI

const GPU = PDCS.PDCS_GPU
const CONFIGURATIONS = (:diagonal, :scalar_cone)
const CONFIGURATION_ORDERS = (
    (:diagonal, :scalar_cone),
    (:scalar_cone, :diagonal),
)


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
    default_input = normpath(joinpath(
        @__DIR__, "..", "..", "PDCS_fork", "benchmark", "represent_data",
    ))
    default_output = joinpath(
        @__DIR__, "results", "R3.5_scalar_cone_ablation",
    )
    options = (
        input_dir = abspath(get(values, "input-dir", default_input)),
        output_dir = abspath(get(values, "output-dir", default_output)),
        shard_index = parse(Int, get(values, "shard-index", "1")),
        shard_count = parse(Int, get(values, "shard-count", "1")),
        tolerance = parse(Float64, get(values, "tolerance", "1e-6")),
        time_limit = parse(Float64, get(values, "time-limit", "3600")),
        check_frequency = parse(Int, get(values, "check-frequency", "2000")),
        print_frequency = parse(Int, get(values, "print-frequency", "20000")),
        order_seed = parse(Int, get(values, "order-seed", "20260815")),
        warmup = lowercase(get(values, "warmup", "true")) == "true",
        repetitions = parse(Int, get(values, "repetitions", "2")),
        repetition_indices = [
            parse(Int, value)
            for value in filter(
                !isempty,
                strip.(split(get(values, "repetition-indices", ""), ',')),
            )
        ],
        instances = filter(
            !isempty,
            strip.(split(get(values, "instances", ""), ',')),
        ),
    )
    isdir(options.input_dir) || error("input directory not found: $(options.input_dir)")
    1 <= options.shard_index <= options.shard_count ||
        error("shard index must be in 1:shard_count")
    options.tolerance > 0 || error("tolerance must be positive")
    options.time_limit > 0 || error("time limit must be positive")
    options.check_frequency > 0 || error("check frequency must be positive")
    options.print_frequency > 0 || error("print frequency must be positive")
    options.repetitions > 0 || error("repetitions must be positive")
    all(index -> 1 <= index <= options.repetitions, options.repetition_indices) ||
        error("--repetition-indices must lie in 1:repetitions")
    length(unique(options.repetition_indices)) == length(options.repetition_indices) ||
        error("--repetition-indices contains duplicates")
    isempty(options.instances) || options.shard_count == 1 ||
        error("--instances requires --shard-count 1")
    return options
end


function load_cbf(path)
    model = MOI.FileFormats.CBF.Model()
    open(`gzip -dc -- $path`) do stream
        read!(stream, model)
    end
    return model
end


function pdcs_backend(optimizer)
    bridge_model = getfield(optimizer, :model)
    backend = getfield(bridge_model, :optimizer)
    backend isa GPU.Optimizer ||
        error("unexpected optimizer stack: backend has type $(typeof(backend))")
    return backend
end


function configure!(optimizer, configuration, options; warmup = false)
    rescaling_method, scalar_cone_rescaling = if configuration == :diagonal
        (:ruiz_pock_chambolle, false)
    elseif configuration == :scalar_cone
        (:ruiz_pock_chambolle, true)
    else
        error("unknown configuration: $configuration")
    end
    attributes = (
        "verbose" => 0,
        "time_limit_secs" => warmup ? min(options.time_limit, 0.2) : options.time_limit,
        "abs_tol" => warmup ? 1e-3 : options.tolerance,
        "rel_tol" => warmup ? 1e-3 : options.tolerance,
        "check_terminate_freq" => warmup ? 100 : options.check_frequency,
        "print_freq" => options.print_frequency,
        # Keep the same preconditioned PDHG loop in both modes, isolating the
        # choice between coordinate-wise and scalar-per-cone rescaling.
        "use_scaling" => true,
        "rescaling_method" => rescaling_method,
        "scalar_cone_rescaling" => scalar_cone_rescaling,
        "use_adaptive_restart" => true,
        "use_restart" => true,
        "use_adaptive_step" => true,
        "use_adaptive_step_size_weight" => true,
        "use_aggressive" => true,
        "use_reflection" => true,
        "use_resolving" => true,
        "use_halpern" => false,
        "use_kkt_restart" => false,
        "use_duality_gap_restart" => true,
        "max_outer_iter" => 3_000_000_000,
        "max_inner_iter" => 3_000_000_000,
        "logfile" => nothing,
    )
    for (name, value) in attributes
        MOI.set(optimizer, MOI.RawOptimizerAttribute(name), value)
    end
    return (; rescaling_method, scalar_cone_rescaling)
end


function instantiate_model(cbf_model)
    optimizer = MOI.instantiate(GPU.Optimizer; with_bridge_type = Float64)
    MOI.copy_to(optimizer, cbf_model)
    return optimizer
end


function safe_json_value(value)
    if value isa AbstractFloat && !isfinite(value)
        return nothing
    elseif value isa Symbol || value isa MOI.TerminationStatusCode
        return string(value)
    elseif value isa NamedTuple
        return Dict(string(key) => safe_json_value(getfield(value, key)) for key in keys(value))
    elseif value isa AbstractDict
        return Dict(string(key) => safe_json_value(item) for (key, item) in value)
    elseif value isa Tuple || value isa AbstractVector
        return [safe_json_value(item) for item in value]
    end
    return value
end


function atomic_json(path, value)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do stream
        JSON.print(stream, safe_json_value(value), 2)
        write(stream, '\n')
        flush(stream)
    end
    mv(temporary, path; force = true)
    return path
end


function configuration_order(filename, seed)
    digest = SHA.sha256(codeunits("$seed:$filename"))
    return CONFIGURATION_ORDERS[Int(digest[1]) % length(CONFIGURATION_ORDERS) + 1]
end


function result_paths(options, instance_id, configuration, repetition)
    directory = joinpath(options.output_dir, "cases", instance_id)
    stem = string(configuration) * "_rep" * lpad(string(repetition), 2, '0')
    return (
        json = joinpath(directory, stem * ".json"),
        done = joinpath(directory, stem * ".done"),
        log = joinpath(directory, stem * ".raw.log"),
    )
end


function selected_metrics(metrics)
    return (
        exit_code = metrics.exit_code,
        exit_status = metrics.exit_status,
        solve_time_sec = metrics.solve_time_sec,
        projection_time_sec = metrics.projection_time_sec,
        primal_projection_time_sec = metrics.primal_projection_time_sec,
        dual_slack_projection_time_sec = metrics.dual_slack_projection_time_sec,
        preprocessing_time_sec = metrics.preprocessing_time_sec,
        iterations = metrics.iterations,
        l_inf_rel_primal_res = metrics.l_inf_rel_primal_res,
        l_inf_rel_dual_res = metrics.l_inf_rel_dual_res,
        l_2_rel_primal_res = metrics.l_2_rel_primal_res,
        l_2_rel_dual_res = metrics.l_2_rel_dual_res,
        relative_gap = metrics.relative_gap,
        objective_value = metrics.objective_value,
        dual_objective_value = metrics.dual_objective_value,
        restart_count = metrics.restart_count,
    )
end


function run_configuration(
    cbf_model,
    filename,
    configuration,
    order_position,
    repetition,
    options,
)
    instance_id = replace(filename, r"\.cbf\.gz$" => "")
    paths = result_paths(options, instance_id, configuration, repetition)
    isfile(paths.done) && return :skipped
    mkpath(dirname(paths.log))
    result = Dict{String,Any}(
        "schema_version" => 1,
        "instance_id" => instance_id,
        "filename" => filename,
        "configuration" => string(configuration),
        "repetition" => repetition,
        "order_position" => order_position,
        "order_seed" => options.order_seed,
        "tolerance" => options.tolerance,
        "time_limit_sec" => options.time_limit,
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "cuda_device" => CUDA.name(CUDA.device()),
        "julia_version" => string(VERSION),
        "started_utc" => string(now(UTC)),
    )
    success = false
    open(paths.log, "w") do stream
        logger = SimpleLogger(stream, Logging.Info)
        redirect_stdout(stream) do
            redirect_stderr(stream) do
                with_logger(logger) do
                    try
                        optimizer = instantiate_model(cbf_model)
                        resolved = configure!(optimizer, configuration, options)
                        result["rescaling_method"] = string(resolved.rescaling_method)
                        result["scalar_cone_rescaling"] = resolved.scalar_cone_rescaling
                        CUDA.synchronize()
                        wall_start = time()
                        MOI.optimize!(optimizer)
                        CUDA.synchronize()
                        result["optimize_wall_sec"] = time() - wall_start
                        backend = pdcs_backend(optimizer)
                        raw_metrics = MOI.get(
                            backend,
                            MOI.RawOptimizerAttribute("result_metrics"),
                        )
                        metrics = selected_metrics(raw_metrics)
                        metric_max = maximum((
                            metrics.l_inf_rel_primal_res,
                            metrics.l_inf_rel_dual_res,
                            metrics.relative_gap,
                        ))
                        termination = MOI.get(optimizer, MOI.TerminationStatus())
                        verified = isfinite(metric_max) && metric_max <= options.tolerance
                        result["termination_status"] = string(termination)
                        result["metrics"] = safe_json_value(metrics)
                        result["verification_metric_max"] = safe_json_value(metric_max)
                        result["verified_solved"] = verified
                        result["solver_end_to_end_sec"] =
                            metrics.preprocessing_time_sec + metrics.solve_time_sec
                        result["projection_fraction"] =
                            metrics.solve_time_sec > 0 ?
                            metrics.projection_time_sec / metrics.solve_time_sec : nothing
                        result["projection_time_per_iteration_sec"] =
                            metrics.iterations > 0 ?
                            metrics.projection_time_sec / metrics.iterations : nothing
                        result["run_status"] = "COMPLETED"
                        success = true
                    catch error_value
                        result["run_status"] = "RUNTIME_ERROR"
                        result["error"] = sprint(
                            showerror,
                            error_value,
                            catch_backtrace(),
                        )
                        showerror(stream, error_value, catch_backtrace())
                        println(stream)
                    end
                    result["finished_utc"] = string(now(UTC))
                    flush(stream)
                end
            end
        end
    end
    atomic_json(paths.json, result)
    if success
        open(paths.done, "w") do stream
            println(stream, paths.json)
        end
    end
    GC.gc(true)
    CUDA.reclaim()
    return success ? :completed : :failed
end


function warmup!(options)
    # Exercise mixed-cone block-wise, one-cone thread-wise, multi-SOC
    # warp-wise, and pure-SOC block-wise paths before collecting formal times.
    warmup_filenames = (
        "bss1.cbf.gz",
        "gbd.cbf.gz",
        "fo7.cbf.gz",
        "classical_200_0.cbf.gz",
    )
    log_path = joinpath(
        options.output_dir,
        "warmup_shard_$(lpad(string(options.shard_index), 2, '0')).raw.log",
    )
    mkpath(dirname(log_path))
    open(log_path, "w") do stream
        redirect_stdout(stream) do
            redirect_stderr(stream) do
                for filename in warmup_filenames
                    path = joinpath(options.input_dir, filename)
                    isfile(path) || error("warm-up instance not found: $path")
                    model = load_cbf(path)
                    for configuration in CONFIGURATIONS
                        optimizer = instantiate_model(model)
                        configure!(optimizer, configuration, options; warmup = true)
                        try
                            MOI.optimize!(optimizer)
                            CUDA.synchronize()
                        catch error_value
                            showerror(stream, error_value, catch_backtrace())
                            println(stream)
                            rethrow()
                        end
                    end
                end
            end
        end
    end
    GC.gc(true)
    CUDA.reclaim()
end


function formal_files(options)
    files = sort(filter(
        name -> endswith(lowercase(name), ".cbf.gz"),
        readdir(options.input_dir),
    ))
    length(files) == 62 ||
        error("expected 62 formal instances, found $(length(files))")
    if !isempty(options.instances)
        requested = [
            endswith(lowercase(name), ".cbf.gz") ? name : name * ".cbf.gz"
            for name in options.instances
        ]
        length(unique(requested)) == length(requested) ||
            error("--instances contains duplicates")
        missing = setdiff(requested, files)
        isempty(missing) || error("unknown --instances entries: $missing")
        return sort(requested)
    end
    return [
        name for (index, name) in enumerate(files)
        if mod(index - 1, options.shard_count) + 1 == options.shard_index
    ]
end


function main()
    options = parse_cli(ARGS)
    CUDA.functional() || error("CUDA is not functional")
    mkpath(options.output_dir)
    options.warmup && warmup!(options)
    files = formal_files(options)
    println(
        "R3.5_SHARD_START shard=$(options.shard_index)/$(options.shard_count) " *
        "instances=$(length(files)) device=$(CUDA.name(CUDA.device()))",
    )
    completed = 0
    skipped = 0
    failed = 0
    for (instance_index, filename) in enumerate(files)
        path = joinpath(options.input_dir, filename)
        cbf_model = load_cbf(path)
        base_order = configuration_order(filename, options.order_seed)
        println("INSTANCE_START index=$instance_index filename=$filename base_order=$base_order")
        repetition_indices = isempty(options.repetition_indices) ?
            (1:options.repetitions) : options.repetition_indices
        for repetition in repetition_indices
            order = isodd(repetition) ? base_order : reverse(base_order)
            for (order_position, configuration) in enumerate(order)
                status = run_configuration(
                    cbf_model,
                    filename,
                    configuration,
                    order_position,
                    repetition,
                    options,
                )
                completed += status == :completed
                skipped += status == :skipped
                failed += status == :failed
                println(
                    "CONFIG_RESULT filename=$filename configuration=$configuration " *
                    "repetition=$repetition status=$status",
                )
                flush(stdout)
            end
        end
    end
    println(
        "R3.5_SHARD_COMPLETE shard=$(options.shard_index)/$(options.shard_count) " *
        "completed=$completed skipped=$skipped failed=$failed",
    )
    failed == 0 || exit(2)
end


main()
