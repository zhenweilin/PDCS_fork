#!/usr/bin/env julia

using Dates
using SparseArrays
using TOML

include(joinpath(@__DIR__, "fisher_market_common.jl"))
using .FisherMarketCommon

function parse_cli(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        option = arguments[index]
        startswith(option, "--") ||
            error("unexpected positional argument: $option")
        index == length(arguments) && error("missing value for $option")
        haskey(options, option) && error("duplicate option: $option")
        options[option] = arguments[index + 1]
        index += 2
    end
    for required in ("--solver", "--manifest", "--instance-id", "--result")
        haskey(options, required) || error("missing required option $required")
    end
    solver = Symbol(options["--solver"])
    solver in (:cupdcs, :scs_gpu, :cuclarabel) ||
        error("--solver must be cupdcs, scs_gpu, or cuclarabel")
    tolerance = parse(Float64, get(options, "--tolerance", "1e-6"))
    time_limit = parse(Float64, get(options, "--time-limit", "600"))
    print_frequency = parse(Int, get(options, "--print-frequency", "1000"))
    verbose_level = parse(Int, get(options, "--verbose-level", "2"))
    tolerance > 0 || error("--tolerance must be positive")
    time_limit > 0 || error("--time-limit must be positive")
    print_frequency > 0 || error("--print-frequency must be positive")
    verbose_level in 0:2 || error("--verbose-level must be 0, 1, or 2")
    return (
        solver = solver,
        manifest = abspath(options["--manifest"]),
        instance_id = options["--instance-id"],
        result = abspath(options["--result"]),
        tolerance = tolerance,
        time_limit = time_limit,
        print_frequency = print_frequency,
        verbose_level = verbose_level,
        required_julia_version = get(
            options,
            "--required-julia-version",
            "1.12.5",
        ),
    )
end

const OPTIONS = parse_cli(ARGS)

if OPTIONS.solver == :cupdcs
    @eval using CUDA
    @eval using PDCS
elseif OPTIONS.solver == :scs_gpu
    @eval using SCS
    @eval using SCS_GPU_jll
elseif OPTIONS.solver == :cuclarabel
    @eval using CUDA
    @eval using Clarabel
end

function load_entry(options)
    isfile(options.manifest) ||
        error("manifest does not exist: $(options.manifest)")
    string(VERSION) == options.required_julia_version ||
        error(
            "required Julia $(options.required_julia_version), got $VERSION",
        )
    manifest = TOML.parsefile(options.manifest)
    get(manifest, "schema_version", nothing) == 1 ||
        error("manifest schema_version must be 1")
    get(manifest, "replicates", nothing) == 5 ||
        error("manifest replicates must be 5")
    get(manifest, "julia_version", "") == string(VERSION) ||
        error("manifest Julia version does not match the running Julia")

    selected = Dict{String,Any}[]
    for group in ("smoke_instances", "instances")
        for entry in get(manifest, group, Dict{String,Any}[])
            entry["id"] == options.instance_id && push!(selected, entry)
        end
    end
    length(selected) == 1 ||
        error(
            "instance ID $(options.instance_id) was not found exactly once",
        )
    return selected[1]
end

function atomic_toml(path, result)
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
    return path
end

function normalize_pdcs_status(status)
    value = lowercase(string(status))
    value == "optimal" && return "OPTIMAL"
    value == "time_limit" && return "TIME_LIMIT"
    value == "max_iter" && return "ITERATION_LIMIT"
    return uppercase(value)
end

function normalize_scs_status(status)
    value = lowercase(status)
    value == "solved" && return "OPTIMAL"
    value == "solved (inaccurate - reached max_iters)" &&
        return "ALMOST_OPTIMAL"
    occursin("time", value) && return "TIME_LIMIT"
    occursin("infeasible", value) && return "INFEASIBLE"
    occursin("unbounded", value) && return "DUAL_INFEASIBLE"
    return uppercase(replace(value, ' ' => '_'))
end

function normalize_clarabel_status(status)
    value = uppercase(string(status))
    value == "SOLVED" && return "OPTIMAL"
    value == "ALMOSTSOLVED" && return "ALMOST_OPTIMAL"
    value == "MAXTIME" && return "TIME_LIMIT"
    value == "MAXITERATIONS" && return "ITERATION_LIMIT"
    return value
end

function solve_cupdcs(instance, options)
    CUDA.functional() || error("CUDA is not functional for cuPDCS")
    setup_started = time()
    formulation = build_pdcs_formulation(instance)
    setup_seconds = time() - setup_started
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
        method = :average,
        time_limit = options.time_limit,
        use_adaptive_restart = true,
        use_adaptive_step_size_weight = true,
        use_resolving = true,
        use_accelerated = false,
        use_aggressive = true,
        verbose = options.verbose_level,
        rel_tol = options.tolerance,
        abs_tol = options.tolerance,
        logfile_name = nothing,
        use_kkt_restart = false,
        use_duality_gap_restart = true,
    )
    CUDA.synchronize()
    solve_wall_seconds = time() - solve_started
    primal = Array(solution.x.recovered_primal.primal_sol.x)
    converge_info = solution.info.convergeInfo[1]
    metadata = Dict{String,Any}(
        "termination_status" =>
            normalize_pdcs_status(solution.info.exit_status),
        "raw_status" => string(solution.info.exit_status),
        "iterations" => Int(solution.info.iter),
        "solver_objective" => solution.info.pObj,
        "solver_dual_objective" => solution.info.dObj,
        "solver_solve_seconds" => solution.info.time,
        "solver_primal_residual" =>
            converge_info.l_inf_rel_primal_res,
        "solver_dual_residual" =>
            converge_info.l_inf_rel_dual_res,
        "solver_relative_gap" => converge_info.rel_gap,
        "gpu_backend" => "PDCS.PDCS_GPU.rpdhg_gpu_solve",
        "cuda_device" => CUDA.name(CUDA.device()),
    )
    return (
        primal = primal,
        metadata = metadata,
        setup_seconds = setup_seconds,
        solve_wall_seconds = solve_wall_seconds,
    )
end

function solve_scs_gpu(instance, options)
    SCS.is_available(SCS.GpuIndirectSolver) ||
        error("SCS.GpuIndirectSolver is unavailable")
    setup_started = time()
    formulation = build_standard_formulation(instance)
    quadratic = empty_quadratic(formulation.variable_count)
    primal = zeros(Float64, formulation.variable_count)
    dual = zeros(Float64, formulation.row_count)
    slack = zeros(Float64, formulation.row_count)
    setup_seconds = time() - setup_started

    solve_started = time()
    solution = SCS.scs_solve(
        SCS.GpuIndirectSolver,
        formulation.row_count,
        formulation.variable_count,
        formulation.A,
        quadratic,
        formulation.b,
        formulation.c,
        formulation.zero_count,
        formulation.nonnegative_count,
        Float64[],
        Float64[],
        Int[],
        Int[],
        formulation.exponential_count,
        0,
        Float64[],
        primal,
        dual,
        slack;
        warm_start = false,
        time_limit_secs = options.time_limit,
        eps_abs = options.tolerance,
        eps_rel = options.tolerance,
        eps_infeas = options.tolerance,
        max_iters = typemax(Int32),
        verbose = options.verbose_level > 0,
    )
    solve_wall_seconds = time() - solve_started
    raw_status = SCS.raw_status(solution.info)
    metadata = Dict{String,Any}(
        "termination_status" => normalize_scs_status(raw_status),
        "raw_status" => raw_status,
        "iterations" => Int(solution.info.iter),
        "solver_objective" => solution.info.pobj,
        "solver_dual_objective" => solution.info.dobj,
        "solver_solve_seconds" => solution.info.solve_time / 1_000.0,
        "solver_setup_seconds" => solution.info.setup_time / 1_000.0,
        "solver_primal_residual" => solution.info.res_pri,
        "solver_dual_residual" => solution.info.res_dual,
        "solver_relative_gap" =>
            abs(solution.info.gap) /
            max(1.0, abs(solution.info.pobj), abs(solution.info.dobj)),
        "gpu_backend" => "SCS.GpuIndirectSolver",
        "gpu_index_type" => string(SCS.scsint_t(SCS.GpuIndirectSolver)),
    )
    return (
        primal = solution.x,
        metadata = metadata,
        setup_seconds = setup_seconds,
        solve_wall_seconds = solve_wall_seconds,
    )
end

function solve_cuclarabel(instance, options)
    CUDA.functional() || error("CUDA is not functional for cuClarabel")
    setup_started = time()
    formulation = build_standard_formulation(instance)
    quadratic = empty_quadratic(formulation.variable_count)
    cones = Clarabel.SupportedCone[
        Clarabel.ZeroConeT(formulation.zero_count),
        Clarabel.NonnegativeConeT(formulation.nonnegative_count),
    ]
    sizehint!(cones, formulation.exponential_count + 2)
    for _ in 1:formulation.exponential_count
        push!(cones, Clarabel.ExponentialConeT())
    end
    settings = Clarabel.Settings(
        max_iter = typemax(UInt32),
        time_limit = options.time_limit,
        tol_gap_abs = options.tolerance,
        tol_gap_rel = options.tolerance,
        tol_feas = options.tolerance,
        tol_infeas_abs = options.tolerance,
        tol_infeas_rel = options.tolerance,
        verbose = options.verbose_level > 0,
        direct_solve_method = :cudss,
    )

    solver = Clarabel.Solver(
        quadratic,
        formulation.c,
        formulation.A,
        formulation.b,
        cones,
        settings,
    )
    CUDA.synchronize()
    setup_seconds = time() - setup_started
    solve_started = time()
    solution = Clarabel.solve!(solver)
    CUDA.synchronize()
    solve_wall_seconds = time() - solve_started
    metadata = Dict{String,Any}(
        "termination_status" =>
            normalize_clarabel_status(solution.status),
        "raw_status" => string(solution.status),
        "iterations" => Int(solution.iterations),
        "solver_objective" => solution.obj_val,
        "solver_dual_objective" => solution.obj_val_dual,
        "solver_solve_seconds" => solution.solve_phase_time,
        "solver_total_seconds" => solution.solve_time,
        "solver_setup_seconds" => solution.setup_phase_time,
        "solver_primal_residual" => solution.r_prim,
        "solver_dual_residual" => solution.r_dual,
        "solver_relative_gap" =>
            abs(solution.obj_val - solution.obj_val_dual) /
            max(1.0, abs(solution.obj_val), abs(solution.obj_val_dual)),
        "gpu_backend" => "Clarabel direct_solve_method=:cudss",
        "cuda_device" => CUDA.name(CUDA.device()),
    )
    return (
        primal = Array(solution.x),
        metadata = metadata,
        setup_seconds = setup_seconds,
        solve_wall_seconds = solve_wall_seconds,
    )
end

function solve_instance(instance, options)
    options.solver == :cupdcs && return solve_cupdcs(instance, options)
    options.solver == :scs_gpu && return solve_scs_gpu(instance, options)
    return solve_cuclarabel(instance, options)
end

function main()
    started = now(UTC)
    result = Dict{String,Any}(
        "schema_version" => 1,
        "instance_id" => OPTIONS.instance_id,
        "solver" => string(OPTIONS.solver),
        "tolerance" => OPTIONS.tolerance,
        "time_limit_seconds" => OPTIONS.time_limit,
        "verbose_level" => OPTIONS.verbose_level,
        "julia_version" => string(VERSION),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "started_utc" => string(started),
        "run_status" => "runtime_error",
        "termination_status" => "EXCEPTION",
        "error" => "",
    )
    exit_code = 1
    try
        entry = load_entry(OPTIONS)
        m = Int(entry["m"])
        n = Int(entry["n"])
        density = Float64(entry["density"])
        seed = parse(Int64, string(entry["seed"]))
        result["m"] = m
        result["n"] = n
        result["density"] = density
        result["seed"] = string(seed)
        result["replicate"] = Int(entry["replicate"])
        println(
            "FISHER_SOLVE_START solver=$(OPTIONS.solver) " *
            "instance=$(OPTIONS.instance_id) m=$m n=$n density=$density " *
            "seed=$seed tolerance=$(OPTIONS.tolerance) " *
            "time_limit=$(OPTIONS.time_limit)",
        )

        generation_started = time()
        instance = generate_instance(m, n, density, seed)
        result["generation_seconds"] = time() - generation_started
        for name in propertynames(instance.summary)
            result[string(name)] = getproperty(instance.summary, name)
        end
        result["seed"] = string(seed)
        println(
            "FISHER_INSTANCE_READY instance=$(OPTIONS.instance_id) " *
            "allocation_count=$(instance.summary.allocation_count) " *
            "utility_nnz=$(instance.summary.utility_nnz) " *
            "digest=$(instance.summary.numerical_digest) " *
            "generation_seconds=$(result["generation_seconds"])",
        )

        solved = solve_instance(instance, OPTIONS)
        merge!(result, solved.metadata)
        result["setup_seconds"] = solved.setup_seconds
        result["solve_wall_seconds"] = solved.solve_wall_seconds

        metrics = independent_primal_metrics(solved.primal, instance)
        for name in propertynames(metrics)
            result[string(name)] = getproperty(metrics, name)
        end
        instance = nothing
        GC.gc()

        result["run_status"] =
            result["termination_status"] in ("OPTIMAL", "ALMOST_OPTIMAL") ?
            "completed" : "solver_stopped"
        result["finished_utc"] = string(now(UTC))
        result["elapsed_wall_seconds"] =
            Dates.value(now(UTC) - started) / 1_000.0
        println(
            "FISHER_SOLVE_RESULT solver=$(OPTIONS.solver) " *
            "instance=$(OPTIONS.instance_id) " *
            "termination=$(result["termination_status"]) " *
            "iterations=$(result["iterations"]) " *
            "objective=$(result["objective_value"]) " *
            "supply_rel=$(result["supply_rel_residual"]) " *
            "utility_rel=$(result["utility_rel_residual"]) " *
            "exp_log_violation=$(result["exponential_log_violation"])",
        )
        exit_code = result["run_status"] == "completed" ? 0 : 3
    catch error_value
        result["error"] = sprint(showerror, error_value, catch_backtrace())
        result["finished_utc"] = string(now(UTC))
        result["elapsed_wall_seconds"] =
            Dates.value(now(UTC) - started) / 1_000.0
        showerror(stderr, error_value, catch_backtrace())
        println(stderr)
    finally
        atomic_toml(OPTIONS.result, result)
    end
    return exit_code
end

exit(main())
