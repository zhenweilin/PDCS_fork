#!/usr/bin/env julia

using Dates
using SHA
using SparseArrays
using TOML

using CUDA
using PDCS

const SIBLING_COMMON = normpath(joinpath(
    @__DIR__, "..", "..", "..", "PDCS_fork", "benchmark",
    "large_scale_fisher_market", "fisher_market_common.jl",
))
isfile(SIBLING_COMMON) || error("missing Fisher generator: $SIBLING_COMMON")
include(SIBLING_COMMON)
using .FisherMarketCommon

include(joinpath(@__DIR__, "FisherDirectFormulation.jl"))
using .FisherDirectFormulation

function parse_bool(value::AbstractString)
    lowercase(value) == "true" && return true
    lowercase(value) == "false" && return false
    error("expected true or false, got $value")
end

function parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        option = arguments[index]
        startswith(option, "--") || error("unexpected argument: $option")
        index < length(arguments) || error("missing value for $option")
        haskey(values, option) && error("duplicate option: $option")
        values[option] = arguments[index + 1]
        index += 2
    end
    for required in ("--m", "--n", "--density", "--seed", "--result")
        haskey(values, required) || error("missing required option $required")
    end
    formulation = Symbol(get(values, "--formulation", "direct"))
    formulation in (:direct, :lifted) ||
        error("--formulation must be direct or lifted")
    rescaling = Symbol(get(values, "--rescaling", "diagonal"))
    rescaling in (:diagonal, :scalar_cone) ||
        error("--rescaling must be diagonal or scalar_cone")
    m = parse(Int, values["--m"])
    n = parse(Int, values["--n"])
    density = parse(Float64, values["--density"])
    seed = parse(Int, values["--seed"])
    tolerance = parse(Float64, get(values, "--tolerance", "1e-6"))
    time_limit = parse(Float64, get(values, "--time-limit", "3600"))
    print_frequency = parse(Int, get(values, "--print-frequency", "1000"))
    verbose = parse(Int, get(values, "--verbose", "1"))
    force = parse_bool(get(values, "--force", "false"))
    config = haskey(values, "--config") ?
        abspath(values["--config"]) : nothing
    m > 0 && n > 0 || error("m and n must be positive")
    0.0 < density <= 1.0 || error("density must be in (0,1]")
    tolerance > 0.0 || error("tolerance must be positive")
    time_limit > 0.0 || error("time limit must be positive")
    print_frequency > 0 || error("print frequency must be positive")
    verbose in 0:2 || error("verbose must be 0, 1, or 2")
    config === nothing || isfile(config) || error("config does not exist: $config")
    return (
        m = m,
        n = n,
        density = density,
        seed = seed,
        formulation = formulation,
        rescaling = rescaling,
        tolerance = tolerance,
        time_limit = time_limit,
        print_frequency = print_frequency,
        verbose = verbose,
        force = force,
        config = config,
        result = abspath(values["--result"]),
    )
end

function file_sha256(path::AbstractString)
    return open(path, "r") do stream
        bytes2hex(SHA.sha256(stream))
    end
end

function git_value(arguments...; directory)
    try
        return strip(read(`git -C $directory $(arguments)`, String))
    catch
        return "unavailable"
    end
end

function atomic_toml(path::AbstractString, result)
    mkpath(dirname(path))
    temporary, stream = mktemp(dirname(path))
    try
        TOML.print(stream, result; sorted = true)
        close(stream)
        mv(temporary, path; force = true)
    catch
        isopen(stream) && close(stream)
        isfile(temporary) && rm(temporary)
        rethrow()
    end
end

function normalize_status(status)
    value = lowercase(string(status))
    value == "optimal" && return "OPTIMAL"
    occursin("time", value) && return "TIME_LIMIT"
    occursin("iter", value) && return "ITERATION_LIMIT"
    return uppercase(value)
end

function main()
    options = parse_cli(ARGS)
    if isfile(options.result) && !options.force
        println("R35_FISHER_SKIP result=$(options.result)")
        return 0
    end
    CUDA.functional() || error("a functional CUDA device is required")

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    sibling_root = normpath(joinpath(repo_root, "..", "PDCS_fork"))
    result = Dict{String,Any}(
        "schema_version" => 1,
        "experiment" => "R3.5 Fisher diagonal-rescaling ablation",
        "formulation" => string(options.formulation),
        "rescaling" => string(options.rescaling),
        "rescaling_method" => "ruiz_pock_chambolle",
        "scalar_cone_rescaling" => options.rescaling == :scalar_cone,
        "use_preconditioner" => true,
        "m" => options.m,
        "n" => options.n,
        "density" => options.density,
        "seed" => options.seed,
        "tolerance" => options.tolerance,
        "time_limit_seconds" => options.time_limit,
        "value_type" => "Float64",
        "matrix_storage" => "in_memory_sparse_csc",
        "saved_instance_data" => false,
        "julia_version" => string(VERSION),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "cuda_device" => CUDA.name(CUDA.device()),
        "runner_sha256" => file_sha256(@__FILE__),
        "direct_formulation_sha256" => file_sha256(joinpath(
            @__DIR__, "FisherDirectFormulation.jl",
        )),
        "generator_sha256" => file_sha256(SIBLING_COMMON),
        "repo_git_commit" => git_value("rev-parse", "HEAD"; directory = repo_root),
        "generator_repo_git_commit" =>
            git_value("rev-parse", "HEAD"; directory = sibling_root),
        "started_utc" => string(now(UTC)),
        "status" => "runtime_error",
        "error" => "",
    )
    if options.config !== nothing
        result["config_sha256"] = file_sha256(options.config)
    end

    exit_code = 1
    started = time()
    try
        generation_started = time()
        instance = generate_instance(
            options.m, options.n, options.density, options.seed,
        )
        result["generation_seconds"] = time() - generation_started
        for name in propertynames(instance.summary)
            result[string(name)] = getproperty(instance.summary, name)
        end
        result["seed"] = options.seed

        setup_started = time()
        formulation = options.formulation == :direct ?
            build_direct_pdcs_formulation(instance) :
            build_pdcs_formulation(instance)
        result["setup_seconds"] = time() - setup_started
        result["variable_count"] = formulation.variable_count
        result["row_count"] = formulation.row_count
        result["matrix_nnz"] = nnz(formulation.A)
        result["zero_count"] = formulation.zero_count
        result["exponential_count"] = formulation.exponential_count

        GC.gc(true)
        CUDA.reclaim()
        CUDA.synchronize()
        solve_started = time()
        solution = PDCS.PDCS_GPU.rpdhg_gpu_solve(
            n = formulation.variable_count,
            m = formulation.row_count,
            nb = formulation.variable_count,
            c = formulation.c,
            G = formulation.A,
            h = formulation.b,
            mGzero = formulation.zero_count,
            mGnonnegative = 0,
            socG = Int[],
            rsocG = Int[],
            expG = formulation.exponential_count,
            dual_expG = 0,
            bl = formulation.lower_bounds,
            bu = formulation.upper_bounds,
            soc_x = Int[],
            rsoc_x = Int[],
            exp_x = 0,
            dual_exp_x = 0,
            print_freq = options.print_frequency,
            check_terminate_freq = options.print_frequency,
            use_preconditioner = true,
            rescaling_method = :ruiz_pock_chambolle,
            scalar_cone_rescaling = options.rescaling == :scalar_cone,
            method = :average,
            time_limit = options.time_limit,
            use_adaptive_restart = true,
            use_adaptive_step_size_weight = true,
            use_resolving = true,
            use_accelerated = false,
            use_aggressive = true,
            verbose = options.verbose,
            rel_tol = options.tolerance,
            abs_tol = options.tolerance,
            logfile_name = nothing,
            use_kkt_restart = false,
            use_duality_gap_restart = true,
        )
        CUDA.synchronize()
        result["solve_wall_seconds"] = time() - solve_started

        primal = Array(solution.x.recovered_primal.primal_sol.x)
        result["nonfinite_primal_count"] =
            count(value -> !isfinite(value), primal)
        metrics = options.formulation == :direct ?
            independent_direct_metrics(primal, instance) :
            independent_primal_metrics(primal, instance)
        for name in propertynames(metrics)
            result[string(name)] = getproperty(metrics, name)
        end

        converge = solution.info.convergeInfo[1]
        result["termination_status"] = normalize_status(solution.info.exit_status)
        result["raw_status"] = string(solution.info.exit_status)
        result["solver_exit_code"] = string(solution.info.exit_code)
        result["iterations"] = Int(solution.info.iter)
        result["solver_objective"] = solution.info.pObj
        result["solver_dual_objective"] = solution.info.dObj
        result["solver_reported_seconds"] = solution.info.time
        result["solver_primal_residual"] = converge.l_inf_rel_primal_res
        result["solver_dual_residual"] = converge.l_inf_rel_dual_res
        result["solver_relative_gap"] = converge.rel_gap

        independent_limit = max(20 * options.tolerance, 1e-10)
        verified = result["nonfinite_primal_count"] == 0 &&
            isfinite(result["objective_value"]) &&
            result["supply_rel_residual"] <= independent_limit &&
            result["nonnegative_violation"] <= independent_limit &&
            result["exponential_log_violation"] <= independent_limit
        if options.formulation == :lifted
            verified &= result["utility_rel_residual"] <= independent_limit
        end
        result["independent_verification_limit"] = independent_limit
        result["independently_verified"] = verified
        termination_accepted = result["termination_status"] in
            ("OPTIMAL", "ALMOST_OPTIMAL")
        result["termination_accepted"] = termination_accepted
        result["status"] = if verified && termination_accepted
            "solved_verified"
        elseif !termination_accepted
            lowercase(result["termination_status"])
        else
            "verification_failed"
        end
        exit_code = result["status"] == "solved_verified" ? 0 : 3

        primal = nothing
        formulation = nothing
        instance = nothing
        GC.gc(true)
    catch error_value
        result["error"] = sprint(showerror, error_value, catch_backtrace())
    finally
        result["elapsed_wall_seconds"] = time() - started
        result["finished_utc"] = string(now(UTC))
        atomic_toml(options.result, result)
        status_value = result["status"]
        println(
            "R35_FISHER_RESULT formulation=$(options.formulation) " *
            "rescaling=$(options.rescaling) status=$status_value " *
            "result=$(options.result)",
        )
    end
    return exit_code
end

exit(main())
