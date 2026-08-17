#!/usr/bin/env julia

include(joinpath(@__DIR__, "AE11Common.jl"))

using .AE11Common
using Dates
using JuMP
using LinearAlgebra
using Printf
using SparseArrays
using TOML

import MathOptInterface as MOI

const AE11_BLAS_THREADS = parse(Int, get(ENV, "AE11_BLAS_THREADS", "32"))
AE11_BLAS_THREADS > 0 || error("AE11_BLAS_THREADS must be positive")
BLAS.set_num_threads(AE11_BLAS_THREADS)

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
    required = ("--solver", "--profile", "--seed", "--kappa",
                "--tolerance", "--output-dir")
    all(key -> haskey(values, key), required) ||
        error("required options: $(join(required, ", "))")
    solver = lowercase(values["--solver"])
    solver in ("pdcs_cpu", "scs_indirect", "scs_gpu", "clarabel_cpu",
               "cuclarabel", "mosek") || error("unsupported solver: $solver")
    profile = lowercase(values["--profile"])
    profile in ("pilot", "medium", "large") ||
        error("--profile must be pilot, medium, or large")
    panels = uppercase(get(values, "--panels", profile == "large" ? "A" : "A,B"))
    panels in ("A", "B", "A,B") || error("--panels must be A, B, or A,B")
    return (
        solver = solver,
        profile = profile,
        seed = parse(Int, values["--seed"]),
        kappa = parse(Float64, values["--kappa"]),
        tolerance = parse(Float64, values["--tolerance"]),
        time_limit = haskey(values, "--time-limit") ?
            parse(Float64, values["--time-limit"]) : nothing,
        output_dir = abspath(values["--output-dir"]),
        config = abspath(get(values, "--config", AE11Common.config_path())),
        panels = split(panels, ','),
        warmup = lowercase(get(values, "--warmup", "true")) == "true",
        force = lowercase(get(values, "--force", "false")) == "true",
        verbose = parse(Int, get(values, "--verbose", "0")),
    )
end

const OPTIONS = parse_cli(ARGS)
const SOLVER_IMPORT_ERROR = Ref("")

try
    if OPTIONS.solver == "pdcs_cpu"
        @eval using PDCS
    elseif OPTIONS.solver in ("scs_indirect", "scs_gpu")
        @eval using SCS
        OPTIONS.solver == "scs_gpu" && @eval using SCS_GPU_jll
    elseif OPTIONS.solver == "clarabel_cpu"
        # Clarabel is already resolved in the repository Manifest as a
        # transitive dependency, but is not listed as a direct dependency in
        # Project.toml.  Load that exact recorded package without mutating or
        # instantiating the environment.
        @eval const Clarabel = Base.require(Base.PkgId(
            Base.UUID("61c947e1-3e6d-4ee4-985a-eec8c727bd6e"), "Clarabel",
        ))
    elseif OPTIONS.solver == "cuclarabel"
        @eval using Clarabel
        @eval using CUDA
    elseif OPTIONS.solver == "mosek"
        @eval using MosekTools
    end
catch error_value
    SOLVER_IMPORT_ERROR[] = sprint(showerror, error_value, catch_backtrace())
end

function scs_gpu_optimizer()
    SCS.is_available(SCS.GpuIndirectSolver) ||
        error("SCS.GpuIndirectSolver is unavailable")
    raw = SCS.Optimizer()
    MOI.set(raw, MOI.RawOptimizerAttribute("linear_solver"),
            SCS.GpuIndirectSolver)
    cache = MOI.default_cache(raw, Float64)
    cached = MOI.Utilities.CachingOptimizer(cache, raw)
    optimizer = MOI.Bridges.LazyBridgeOptimizer(cached)
    MOI.Bridges.Variable.add_all_bridges(optimizer, Float64)
    MOI.Bridges.Constraint.add_all_bridges(optimizer, Float64)
    MOI.Bridges.Objective.add_all_bridges(optimizer, Float64)
    return optimizer
end

function optimizer_factory(solver)
    solver == "pdcs_cpu" && return PDCS.PDCS_CPU.Optimizer
    solver == "scs_indirect" && return SCS.Optimizer
    solver == "scs_gpu" && return scs_gpu_optimizer
    solver in ("clarabel_cpu", "cuclarabel") && return Clarabel.Optimizer
    solver == "mosek" && return MosekTools.Optimizer
    error("unsupported solver")
end

function configure!(model, options, tolerance, time_limit; warmup = false)
    effective_tolerance = warmup ? max(tolerance, 1e-3) : tolerance
    effective_limit = warmup ? min(time_limit, 1.0) : time_limit
    attributes = if options.solver == "pdcs_cpu"
        (
            "verbose" => (warmup ? 0 : options.verbose),
            "time_limit_secs" => effective_limit,
            "abs_tol" => effective_tolerance,
            "rel_tol" => effective_tolerance,
            "check_terminate_freq" => (warmup ? 100 : 2000),
            "print_freq" => 20000,
            "use_scaling" => true,
            "rescaling_method" => :ruiz_pock_chambolle,
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
            "max_outer_iter" => (warmup ? 1 : 3_000_000_000),
            "max_inner_iter" => (warmup ? 4 : 3_000_000_000),
            "logfile" => nothing,
        )
    elseif options.solver in ("scs_indirect", "scs_gpu")
        common = Any[
            "eps_abs" => effective_tolerance,
            "eps_rel" => effective_tolerance,
            "time_limit_secs" => effective_limit,
            "max_iters" => (warmup ? 4 : 2_000_000_000),
            "verbose" => (!warmup && options.verbose > 0),
        ]
        push!(common, "linear_solver" => (
            options.solver == "scs_gpu" ?
                SCS.GpuIndirectSolver : SCS.IndirectSolver
        ))
        Tuple(common)
    elseif options.solver in ("clarabel_cpu", "cuclarabel")
        (
            "direct_solve_method" => (
                options.solver == "cuclarabel" ? :cudss : :qdldl
            ),
            "tol_gap_abs" => effective_tolerance,
            "tol_gap_rel" => effective_tolerance,
            "tol_feas" => effective_tolerance,
            "tol_infeas_abs" => effective_tolerance,
            "tol_infeas_rel" => effective_tolerance,
            "time_limit" => effective_limit,
            "max_iter" => (warmup ? 4 : 2_000_000_000),
            "verbose" => (!warmup && options.verbose > 0),
        )
    else
        (
            "MSK_DPAR_OPTIMIZER_MAX_TIME" => effective_limit,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => effective_tolerance,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => effective_tolerance,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS" => effective_tolerance,
            "MSK_IPAR_LOG" => (!warmup && options.verbose > 0 ? 1 : 0),
        )
    end
    for (name, value) in attributes
        set_optimizer_attribute(model, name, value)
    end
    return model
end

function build_model(A, b, lambda, options, tolerance, time_limit;
                     warmup = false)
    m, n = size(A)
    optimizer = optimizer_factory(options.solver)
    model = options.solver == "scs_gpu" ?
        Model(optimizer; add_bridges = false) : Model(optimizer)
    configure!(model, options, tolerance, time_limit; warmup)
    @variable(model, x_one)
    @variable(model, residual_epigraph)
    @variable(model, residual[1:m])
    @variable(model, x_positive[1:n] >= 0.0)
    @variable(model, x_negative[1:n] >= 0.0)
    @objective(
        model, Min,
        2residual_epigraph + lambda * sum(
            x_positive[index] + x_negative[index] for index in 1:n
        ),
    )
    @constraint(
        model,
        [x_one; residual + A * x_positive - A * x_negative] .== [1.0; b],
    )
    @variable(model, t)
    @variable(model, u)
    @constraint(model, t == (x_one + residual_epigraph) / sqrt(2.0))
    @constraint(model, u == (x_one - residual_epigraph) / sqrt(2.0))
    @constraint(model, [t; u; residual] in SecondOrderCone())
    return (; model, x_positive, x_negative)
end

function maybe_get(model, attribute, default)
    try
        return MOI.get(backend(model), attribute)
    catch
        return default
    end
end

function synchronize_solver(solver)
    solver == "cuclarabel" && CUDA.synchronize()
    return nothing
end

function iteration_count(model, solver)
    attribute = if solver == "pdcs_cpu"
        PDCS.PDCS_CPU.PDHGIterations()
    elseif solver in ("scs_indirect", "scs_gpu")
        SCS.ADMMIterations()
    else
        MOI.BarrierIterations()
    end
    return maybe_get(model, attribute, -1)
end

function result_path(options, panel, parameter)
    parameter_tag = panel == "A" ? "beta$(parameter)" : "alpha$(parameter)"
    tolerance_tag = replace(string(options.tolerance), "." => "p", "-" => "m")
    kappa_tag = replace(@sprintf("%.0e", options.kappa), "+" => "")
    filename = join((
        options.profile, "seed$(options.seed)", "K$kappa_tag", "panel$panel",
        parameter_tag, "tol$tolerance_tag", options.solver,
    ), "_") * ".toml"
    return joinpath(options.output_dir, filename)
end

function classify_exception(error_value)
    message = lowercase(sprint(showerror, error_value))
    occursin("license", message) && return "license_failure"
    occursin("out of", message) && occursin("memory", message) &&
        return "out_of_memory"
    occursin("cuda_error_out_of_memory", message) && return "out_of_memory"
    (occursin("not installed", message) || occursin("unavailable", message) ||
     occursin("missing source", message) || occursin("required but", message)) &&
        return "input_or_conversion_failure"
    # A JLL extension can load while its platform artifact binding remains
    # undefined.  This is an environment/artifact availability failure, not a
    # numerical breakdown of the algorithm on the generated instance.
    occursin("undefvarerror", message) && occursin("libscsgpuindir", message) &&
        return "input_or_conversion_failure"
    occursin("copy", message) && occursin("constraint", message) &&
        return "input_or_conversion_failure"
    return "numerical_failure"
end

function environment_metadata(options)
    cpu_affinity_list = try
        affinity_line = only(filter(
            line -> startswith(line, "Cpus_allowed_list:"),
            readlines("/proc/self/status"),
        ))
        strip(split(affinity_line, ':'; limit = 2)[2])
    catch
        "unavailable"
    end
    result = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "active_project" => something(Base.active_project(), ""),
        "active_project_sha256" => begin
            project = Base.active_project()
            project === nothing || !isfile(project) ? "" : file_sha256(project)
        end,
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "cpu_affinity_list" => cpu_affinity_list,
    )
    if options.solver in ("scs_gpu", "cuclarabel") &&
       isempty(SOLVER_IMPORT_ERROR[])
        result["cuda_device"] = options.solver == "cuclarabel" ?
            CUDA.name(CUDA.device()) : "selected by SCS GPU runtime"
    end
    return result
end

function warmup!(options, time_limit)
    isempty(SOLVER_IMPORT_ERROR[]) || return nothing
    design = make_design(32, 5, 16, 4, 91973)
    pattern = AE11Common.build_pattern(design)
    A, _ = assemble_matrix(design, pattern, 1.0)
    penalty = penalty_spec(A, design.b, "B", 0.1)
    built = build_model(
        A, design.b, penalty.lambda, options, 1e-3, time_limit;
        warmup = true,
    )
    try
        optimize!(built.model)
        synchronize_solver(options.solver)
    catch error_value
        # A license or availability failure must remain visible to the formal
        # run rather than aborting before it can write a classified record.
        println(stderr, "AE11_WARMUP_WARNING ", sprint(showerror, error_value))
    end
    return nothing
end

function run_one(
    options, config, design, pattern, A, sigma, panel, parameter,
    penalty, generation_seconds, time_limit,
)
    path = result_path(options, panel, parameter)
    if isfile(path) && !options.force
        println("AE11_SKIP path=$path")
        return path
    end
    result = instance_metadata(
        design, pattern, A, sigma, options.kappa, penalty, panel,
    )
    merge!(result, Dict{String,Any}(
        "schema_version" => 1,
        "solver" => options.solver,
        "status" => "numerical_failure",
        "rescaling_on" => options.solver in ("pdcs_cpu",) ? true : "solver_default",
        "tolerance" => options.tolerance,
        "time_limit_seconds" => time_limit,
        "generation_seconds" => generation_seconds,
        "setup_seconds" => NaN,
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
        "relative_objective_error" => NaN,
        "tail_activation_ratio" => NaN,
        "nonfinite_count" => -1,
        "termination_status" => "EXCEPTION",
        "primal_status" => "NO_SOLUTION",
        "dual_status" => "NO_SOLUTION",
        "error" => "",
        "runner_sha256" => file_sha256(@__FILE__),
        "common_sha256" => file_sha256(joinpath(@__DIR__, "AE11Common.jl")),
        "config_sha256" => file_sha256(options.config),
        "started_utc" => string(now(UTC)),
    ))
    merge!(result, environment_metadata(options))
    started = time()
    try
        isempty(SOLVER_IMPORT_ERROR[]) || error(SOLVER_IMPORT_ERROR[])
        setup_started = time()
        built = build_model(
            A, design.b, penalty.lambda, options, options.tolerance,
            time_limit,
        )
        result["setup_seconds"] = time() - setup_started
        GC.gc(true)
        synchronize_solver(options.solver)
        solve_started = time()
        optimize!(built.model)
        synchronize_solver(options.solver)
        result["solve_seconds"] = time() - solve_started
        result["termination_status"] = string(termination_status(built.model))
        result["primal_status"] = string(primal_status(built.model))
        result["dual_status"] = string(dual_status(built.model))
        result["iterations"] = iteration_count(built.model, options.solver)
        result["solver_reported_seconds"] = maybe_get(
            built.model, MOI.SolveTimeSec(), NaN,
        )
        has_primal = primal_status(built.model) in (
            MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT,
        )
        if has_primal
            x = value.(built.x_positive) .- value.(built.x_negative)
            verification = verify_lasso(
                A, design.b, penalty.lambda, x;
                tolerance = options.tolerance,
                design = design, sigma = sigma,
                tail_fraction = Float64(config["statistics"]["tail_fraction"]),
            )
            merge!(result, verification)
            if verification["verified_solved"]
                result["status"] = "solved_verified"
            elseif termination_status(built.model) in (
                MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL,
            )
                result["status"] = "solver_claimed_solved_but_inaccurate"
            elseif termination_status(built.model) == MOI.TIME_LIMIT
                result["status"] = "timeout"
            else
                result["status"] = "numerical_failure"
            end
        elseif termination_status(built.model) == MOI.TIME_LIMIT
            result["status"] = "timeout"
        else
            result["status"] = "numerical_failure"
        end
    catch error_value
        result["status"] = classify_exception(error_value)
        result["error"] = sprint(showerror, error_value, catch_backtrace())
    finally
        result["end_to_end_seconds"] = time() - started
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
    options = OPTIONS
    config = load_config(options.config)
    profile = config[options.profile]
    m = Int(profile["m"])
    n = Int(profile["n"])
    q = Int(config["q"])
    n == q * m || error("profile n must equal q*m")
    options.seed in Int.(profile["seeds"]) || error("seed outside profile")
    options.kappa in Float64.(config["kappas"]) || error("kappa outside grid")
    time_limit = options.time_limit === nothing ? Float64(profile[
        options.tolerance == 1e-3 ?
            "time_limit_1e_3_seconds" : "time_limit_1e_6_seconds"
    ]) : options.time_limit
    options.warmup && warmup!(options, time_limit)

    generation_started = time()
    design = make_design(
        m, q, Int(config["u_block_size"]), Int(config["v_block_size"]),
        options.seed,
    )
    pattern = AE11Common.build_pattern(design)
    A, sigma = assemble_matrix(design, pattern, options.kappa)
    generation_seconds = time() - generation_started
    matrix_pattern_hash(A) == matrix_pattern_hash(pattern) ||
        error("matrix pattern hash mismatch")

    penalty_runs = Tuple{String,Float64}[]
    "A" in options.panels && push!(
        penalty_runs, ("A", Float64(config["panel_a_beta"])),
    )
    if "B" in options.panels
        append!(penalty_runs, [
            ("B", Float64(alpha)) for alpha in config["panel_b_alphas"]
        ])
    end
    for (panel, parameter) in penalty_runs
        penalty = penalty_spec(A, design.b, panel, parameter, options.kappa)
        run_one(
            options, config, design, pattern, A, sigma, panel, parameter,
            penalty, generation_seconds, time_limit,
        )
    end
    return 0
end

exit(main())
