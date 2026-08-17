#!/usr/bin/env julia

include(joinpath(@__DIR__, "AE11Common.jl"))

using .AE11Common
using CUDA
using Dates
using PDCS
using Printf
using SparseArrays
using TOML

const GPU = PDCS.PDCS_GPU

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
    required = ("--profile", "--seed", "--kappa", "--tolerance",
                "--rescaling", "--output-dir")
    all(key -> haskey(values, key), required) ||
        error("required options: $(join(required, ", "))")
    profile = get(values, "--profile", "")
    profile in ("pilot", "medium", "large") ||
        error("--profile must be pilot, medium, or large")
    rescaling = lowercase(values["--rescaling"])
    rescaling in ("on", "off") || error("--rescaling must be on or off")
    panels = uppercase(get(values, "--panels", profile == "large" ? "A" : "A,B"))
    panels in ("A", "B", "A,B") || error("--panels must be A, B, or A,B")
    options = (
        config = abspath(get(values, "--config", AE11Common.config_path())),
        profile = profile,
        seed = parse(Int, values["--seed"]),
        kappa = parse(Float64, values["--kappa"]),
        tolerance = parse(Float64, values["--tolerance"]),
        solver_tolerance = haskey(values, "--solver-tolerance") ?
            parse(Float64, values["--solver-tolerance"]) :
            parse(Float64, values["--tolerance"]),
        rescaling = rescaling,
        panels = split(panels, ','),
        output_dir = abspath(values["--output-dir"]),
        time_limit = haskey(values, "--time-limit") ?
            parse(Float64, values["--time-limit"]) : nothing,
        warmup = lowercase(get(values, "--warmup", "true")) == "true",
        force = lowercase(get(values, "--force", "false")) == "true",
        verbose = parse(Int, get(values, "--verbose", "0")),
    )
    options.kappa >= 1 || error("--kappa must be at least one")
    options.tolerance > 0 || error("--tolerance must be positive")
    options.solver_tolerance > 0 ||
        error("--solver-tolerance must be positive")
    options.time_limit === nothing || options.time_limit > 0 ||
        error("--time-limit must be positive")
    options.verbose in 0:2 || error("--verbose must be 0, 1, or 2")
    return options
end

function git_value(arguments...; directory)
    try
        return strip(read(`git -C $directory $(arguments)`, String))
    catch
        return "unknown"
    end
end

function classify_exception(error_value)
    message = lowercase(sprint(showerror, error_value))
    occursin("out of", message) && occursin("memory", message) &&
        return "out_of_memory"
    occursin("cuda_error_out_of_memory", message) && return "out_of_memory"
    occursin("input", message) && return "input_or_conversion_failure"
    return "numerical_failure"
end

function result_path(options, panel, parameter)
    parameter_tag = panel == "A" ? "beta$(parameter)" : "alpha$(parameter)"
    tolerance_tag = replace(string(options.tolerance), "." => "p", "-" => "m")
    kappa_tag = replace(@sprintf("%.0e", options.kappa), "+" => "")
    filename = join((
        options.profile, "seed$(options.seed)", "K$kappa_tag", "panel$panel",
        parameter_tag, "tol$tolerance_tag", "rescaling$(options.rescaling)",
    ), "_") * ".toml"
    return joinpath(options.output_dir, filename)
end

function warmup_gpu!()
    # Use a 128-row instance to cover the regular SOC projection path while
    # keeping compilation and allocation outside every formal timing.
    design = make_design(128, 5, 16, 4, 99173)
    pattern = AE11Common.build_pattern(design)
    A, _ = assemble_matrix(design, pattern, 1.0)
    penalty = penalty_spec(A, design.b, "B", 0.1)
    conic = build_conic_data(A, design.b, penalty.lambda)
    GPU.rpdhg_gpu_solve(
        n = conic.n_conic, m = conic.m_conic, nb = conic.nb,
        c = conic.c, G = conic.G, h = conic.h,
        mGzero = conic.mGzero, mGnonnegative = conic.mGnonnegative,
        socG = conic.socG, rsocG = conic.rsocG,
        expG = conic.expG, dual_expG = conic.dual_expG,
        bl = conic.bl, bu = conic.bu,
        soc_x = conic.soc_x, rsoc_x = conic.rsoc_x,
        exp_x = conic.exp_x, dual_exp_x = conic.dual_exp_x,
        use_preconditioner = true,
        rescaling_method = :ruiz_pock_chambolle,
        max_outer_iter = 1, max_inner_iter = 4,
        check_terminate_freq = 2, print_freq = 100,
        abs_tol = 1e-3, rel_tol = 1e-3,
        time_limit = 30.0, verbose = 0,
    )
    CUDA.synchronize()
    return nothing
end

function solver_result(
    options, config, design, pattern, A, sigma, conic, penalty, panel,
    parameter, generation_seconds, conic_setup_seconds,
)
    path = result_path(options, panel, parameter)
    if isfile(path) && !options.force
        println("AE11_SKIP path=$path")
        return path
    end
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    sibling_root = normpath(joinpath(repo_root, "..", "PDCS_fork"))
    result = instance_metadata(
        design, pattern, A, sigma, options.kappa, penalty, panel,
    )
    merge!(result, Dict{String,Any}(
        "schema_version" => 1,
        "solver" => "cupdcs",
        "status" => "numerical_failure",
        "rescaling_on" => options.rescaling == "on",
        "tolerance" => options.tolerance,
        "solver_tolerance" => options.solver_tolerance,
        "time_limit_seconds" => options.time_limit,
        "generation_seconds" => generation_seconds,
        "setup_seconds" => conic_setup_seconds,
        "solve_seconds" => NaN,
        "end_to_end_seconds" => NaN,
        "iterations" => -1,
        "matvec_count" => -1,
        "matvec_count_source" => "not_exposed_by_solver",
        "projection_count" => -1,
        "peak_cpu_memory" => -1,
        "peak_gpu_memory" => -1,
        "primal_residual" => NaN,
        "dual_residual" => NaN,
        "relative_gap" => NaN,
        "independent_kkt" => NaN,
        "objective_value" => NaN,
        "tail_activation_ratio" => NaN,
        "nonfinite_count" => -1,
        "termination_status" => "EXCEPTION",
        "error" => "",
        "julia_version" => string(VERSION),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "cuda_device" => CUDA.name(CUDA.device()),
        "cuda_runtime" => string(CUDA.runtime_version()),
        "runner_sha256" => file_sha256(@__FILE__),
        "common_sha256" => file_sha256(joinpath(@__DIR__, "AE11Common.jl")),
        "config_sha256" => file_sha256(options.config),
        "generator_git_commit" => git_value("rev-parse", "HEAD"; directory = repo_root),
        "solver_environment_git_commit" => git_value("rev-parse", "HEAD"; directory = sibling_root),
        "started_utc" => string(now(UTC)),
    ))
    started = time()
    try
        GC.gc(true)
        CUDA.reclaim()
        CUDA.synchronize()
        solve_started = time()
        solution = GPU.rpdhg_gpu_solve(
            n = conic.n_conic, m = conic.m_conic, nb = conic.nb,
            c = conic.c, G = conic.G, h = conic.h,
            mGzero = conic.mGzero, mGnonnegative = conic.mGnonnegative,
            socG = conic.socG, rsocG = conic.rsocG,
            expG = conic.expG, dual_expG = conic.dual_expG,
            bl = conic.bl, bu = conic.bu,
            soc_x = conic.soc_x, rsoc_x = conic.rsoc_x,
            exp_x = conic.exp_x, dual_exp_x = conic.dual_exp_x,
            use_preconditioner = true,
            rescaling_method = options.rescaling == "on" ?
                :ruiz_pock_chambolle : :none,
            use_adaptive_restart = true,
            use_restart = true,
            use_adaptive_step = true,
            use_adaptive_step_size_weight = true,
            use_aggressive = true,
            use_reflection = true,
            use_resolving = true,
            use_halpern = false,
            use_kkt_restart = false,
            use_duality_gap_restart = true,
            max_outer_iter = Int(config["execution"]["max_outer_iterations"]),
            max_inner_iter = Int(config["execution"]["max_inner_iterations"]),
            check_terminate_freq = Int(config["execution"]["check_terminate_frequency"]),
            print_freq = Int(config["execution"]["print_frequency"]),
            abs_tol = options.solver_tolerance,
            rel_tol = options.solver_tolerance,
            time_limit = options.time_limit,
            warm_start = false,
            verbose = options.verbose,
        )
        CUDA.synchronize()
        result["solve_seconds"] = time() - solve_started
        primal = Array(solution.x.recovered_primal.primal_sol.x)
        x = recover_lasso_x(primal, design.n)
        verification = verify_lasso(
            A, design.b, penalty.lambda, x;
            tolerance = options.tolerance,
            design = design, sigma = sigma,
            tail_fraction = Float64(config["statistics"]["tail_fraction"]),
        )
        merge!(result, verification)
        result["iterations"] = solution.info.iter
        result["termination_status"] = string(solution.info.exit_status)
        result["solver_exit_code"] = string(solution.info.exit_code)
        result["solver_primal_objective"] = solution.info.pObj
        result["solver_dual_objective"] = solution.info.dObj
        result["solver_reported_seconds"] = solution.info.time
        converge = solution.info.convergeInfo[1]
        result["primal_residual"] = converge.l_inf_rel_primal_res
        result["dual_residual"] = converge.l_inf_rel_dual_res
        if verification["verified_solved"]
            result["status"] = "solved_verified"
        elseif occursin("time", lowercase(string(solution.info.exit_status)))
            result["status"] = "timeout"
        elseif occursin("optimal", lowercase(string(solution.info.exit_status))) ||
               occursin("solved", lowercase(string(solution.info.exit_status)))
            result["status"] = "solver_claimed_solved_but_inaccurate"
        else
            result["status"] = "numerical_failure"
        end
    catch error_value
        result["status"] = classify_exception(error_value)
        result["error"] = sprint(showerror, error_value, catch_backtrace())
    finally
        result["end_to_end_seconds"] = time() - started
        # Keep successful scalar records unambiguous: panel B has no beta,
        # and this runner does not use a reference objective.  Omitting these
        # inapplicable fields avoids serializing NaN as an N/A sentinel.
        if result["panel"] == "B"
            delete!(result, "beta")
        elseif result["panel"] == "A"
            delete!(result, "alpha")
        end
        if haskey(result, "relative_objective_error") &&
           isnan(Float64(result["relative_objective_error"]))
            delete!(result, "relative_objective_error")
        end
        # Sys.maxrss() is reported in bytes by Julia on this Linux host.  GPU
        # peak allocation is not exposed by the direct solver API and remains
        # -1 rather than substituting an end-of-run snapshot for a peak.
        result["peak_cpu_memory"] = Sys.maxrss()
        result["finished_utc"] = string(now(UTC))
        write_toml_atomic(path, result)
        println(
            "AE11_RESULT path=$path status=$(result["status"]) " *
            "iterations=$(result["iterations"]) " *
            "kkt=$(result["independent_kkt"])",
        )
    end
    return path
end

function main()
    options = parse_cli(ARGS)
    config = load_config(options.config)
    profile_config = config[options.profile]
    m = Int(profile_config["m"])
    n = Int(profile_config["n"])
    q = Int(config["q"])
    n == q * m || error("profile n must equal q*m")
    options.seed in Int.(profile_config["seeds"]) ||
        error("seed is outside the configured profile")
    options.kappa in Float64.(config["kappas"]) ||
        error("kappa is outside the configured grid")
    options.tolerance in Float64.(config["tolerances"]) ||
        error("tolerance is outside the configured grid")
    time_limit = options.time_limit === nothing ? Float64(profile_config[
        options.tolerance == 1e-3 ?
            "time_limit_1e_3_seconds" : "time_limit_1e_6_seconds"
    ]) : options.time_limit
    options = merge(options, (; time_limit))
    CUDA.functional() || error("a functional CUDA device is required")
    options.warmup && warmup_gpu!()

    generation_started = time()
    design = make_design(
        m, q, Int(config["u_block_size"]), Int(config["v_block_size"]),
        options.seed,
    )
    pattern = AE11Common.build_pattern(design)
    A, sigma = assemble_matrix(design, pattern, options.kappa)
    generation_seconds = time() - generation_started
    matrix_pattern_hash(A) == matrix_pattern_hash(pattern) ||
        error("assembled matrix pattern hash mismatch")
    eltype(A) == Float64 || error("matrix precision mismatch")

    penalty_runs = Tuple{String,Float64}[]
    "A" in options.panels && push!(
        penalty_runs, ("A", Float64(config["panel_a_beta"])),
    )
    if "B" in options.panels
        append!(penalty_runs, [
            ("B", Float64(alpha)) for alpha in config["panel_b_alphas"]
        ])
    end
    first_panel, first_parameter = first(penalty_runs)
    first_penalty = penalty_spec(
        A, design.b, first_panel, first_parameter, options.kappa,
    )
    setup_started = time()
    conic = build_conic_data(A, design.b, first_penalty.lambda)
    conic_setup_seconds = time() - setup_started
    for (run_index, (panel, parameter)) in enumerate(penalty_runs)
        penalty = run_index == 1 ? first_penalty : penalty_spec(
            A, design.b, panel, parameter, options.kappa,
        )
        # G and h do not depend on lambda. Only the first 2n objective
        # coefficients change, so every penalty reuses the same matrix object.
        conic.c[1:(2design.n)] .= penalty.lambda
        solver_result(
            options, config, design, pattern, A, sigma, conic, penalty,
            panel, parameter, generation_seconds, conic_setup_seconds,
        )
    end
    return 0
end

exit(main())
