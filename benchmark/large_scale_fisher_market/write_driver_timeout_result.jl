#!/usr/bin/env julia

using Dates
using TOML

function parse_cli(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        index == length(arguments) &&
            error("missing value for $(arguments[index])")
        options[arguments[index]] = arguments[index + 1]
        index += 2
    end
    for option in (
        "--solver",
        "--manifest",
        "--instance-id",
        "--raw-log",
        "--result",
        "--time-limit",
        "--setup-grace",
        "--tolerance",
        "--gpu",
        "--verbose-level",
    )
        haskey(options, option) || error("missing $option")
    end
    return options
end

function capture_value(pattern, text, default)
    matched = match(pattern, text)
    matched === nothing && return default
    return matched.captures[1]
end

function main()
    options = parse_cli(ARGS)
    manifest = TOML.parsefile(options["--manifest"])
    instance_id = options["--instance-id"]
    entries = [
        entry for entry in manifest["instances"]
        if entry["id"] == instance_id
    ]
    length(entries) == 1 ||
        error("expected one manifest entry for $instance_id")
    entry = only(entries)
    raw_log = isfile(options["--raw-log"]) ?
              read(options["--raw-log"], String) : ""
    solver = options["--solver"]
    time_limit = parse(Float64, options["--time-limit"])
    setup_grace = parse(Float64, options["--setup-grace"])
    status, setup_seconds, solve_seconds, explanation =
        if solver == "scs_gpu"
            (
                "LINEAR_SYSTEM_TIMEOUT",
                setup_grace,
                time_limit,
                "The external timeout expired while SCS GPU indirect was " *
                "inside a linear-system operation; no candidate solution " *
                "was returned to Julia.",
            )
        elseif solver == "cuclarabel"
            (
                "SETUP_TIMEOUT",
                time_limit + setup_grace,
                0.0,
                "The external timeout expired during cuClarabel/cuDSS " *
                "setup; no solver iteration or candidate solution was " *
                "returned to Julia.",
            )
        else
            (
                "EXTERNAL_TIMEOUT",
                setup_grace,
                time_limit,
                "The external timeout expired before the solver returned " *
                "a candidate solution to Julia.",
            )
        end

    digest = capture_value(
        r"FISHER_INSTANCE_READY[^\n]*digest=([0-9a-f]+)",
        raw_log,
        "MISSING",
    )
    utility_nnz = parse(
        Int,
        capture_value(
            r"FISHER_INSTANCE_READY[^\n]*utility_nnz=([0-9]+)",
            raw_log,
            "0",
        ),
    )
    generation_seconds = parse(
        Float64,
        capture_value(
            r"FISHER_INSTANCE_READY[^\n]*generation_seconds=([0-9.eE+-]+)",
            raw_log,
            "0",
        ),
    )
    result = Dict{String,Any}(
        "allocation_count" => entry["m"] * entry["n"],
        "cuda_visible_devices" => options["--gpu"],
        "gpu_name" => get(ENV, "FISHER_GPU_NAME", ""),
        "density" => entry["density"],
        "driver_recorded" => true,
        "elapsed_wall_seconds" => time_limit + setup_grace,
        "error" => explanation,
        "finished_utc" => string(Dates.now(Dates.UTC)),
        "generation_seconds" => generation_seconds,
        "gpu_backend" =>
            solver == "scs_gpu" ? "SCS.GpuIndirectSolver" :
            solver == "cuclarabel" ?
            "Clarabel direct_solve_method=:cudss" :
            "PDCS.PDCS_GPU.rpdhg_gpu_solve",
        "instance_id" => instance_id,
        "iterations" => 0,
        "julia_version" => string(VERSION),
        "m" => entry["m"],
        "n" => entry["n"],
        "numerical_digest" => digest,
        "raw_status" =>
            "external timeout after $(time_limit + setup_grace) seconds",
        "replicate" => entry["replicate"],
        "run_status" => "driver_timeout",
        "status_accepted" => false,
        "validation_accepted" => false,
        "solver_tolerance_accepted" => false,
        "validation_tolerance" => parse(Float64, options["--tolerance"]),
        "schema_version" => 1,
        "seed" => entry["seed"],
        "setup_seconds" => setup_seconds,
        "solve_wall_seconds" => solve_seconds,
        "solver" => solver,
        "termination_status" => status,
        "time_limit_seconds" => time_limit,
        "tolerance" => parse(Float64, options["--tolerance"]),
        "utility_nnz" => utility_nnz,
        "verbose_level" => parse(Int, options["--verbose-level"]),
    )
    result_path = options["--result"]
    mkpath(dirname(result_path))
    open(result_path, "w") do stream
        TOML.print(stream, result; sorted = true)
    end
    println(
        "FISHER_DRIVER_TIMEOUT_RESULT solver=$solver " *
        "instance=$instance_id status=$status result=$result_path",
    )
end

main()
