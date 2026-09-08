#!/usr/bin/env julia

using JuMP
using LinearAlgebra
using Serialization
using TOML

import MathOptInterface as MOI

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ill_conditioned_lasso_cases.jl"))
using .RebuttalCommon
using .IllConditionedLassoCases

function atomic_toml(path::AbstractString, value::AbstractDict)
    mkpath(dirname(abspath(path)))
    temporary, stream = mktemp(dirname(abspath(path)))
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

function load_solver!(solver::Symbol, repo_root::AbstractString)
    if solver == :cupdcs
        @eval using CUDA
        Base.include(Main, joinpath(repo_root, "src", "pdcs_gpu", "PDCS_GPU.jl"))
    elseif solver == :cuscs
        @eval using SCS
        @eval using SCS_GPU_jll
    elseif solver == :cuclarabel
        @eval using CUDA
        @eval using Clarabel
    else
        error("solver must be cupdcs, cuscs, or cuclarabel")
    end
    return nothing
end

function visible_h100()
    output = read(`nvidia-smi --query-gpu=name --format=csv,noheader`, String)
    names = filter(!isempty, strip.(split(chomp(output), '\n')))
    length(names) == 1 || error(
        "expected exactly one Slurm-visible GPU, found $(length(names)): $names",
    )
    occursin("H100", only(names)) || error("expected H100, found $(only(names))")
    return only(names)
end

function scs_gpu_optimizer()
    SCS.is_available(SCS.GpuIndirectSolver) || error(
        "SCS.GpuIndirectSolver is unavailable",
    )
    raw = SCS.Optimizer()
    MOI.set(raw, MOI.RawOptimizerAttribute("linear_solver"), SCS.GpuIndirectSolver)
    cache = MOI.default_cache(raw, Float64)
    cached = MOI.Utilities.CachingOptimizer(cache, raw)
    optimizer = MOI.Bridges.LazyBridgeOptimizer(cached)
    MOI.Bridges.Variable.add_all_bridges(optimizer, Float64)
    MOI.Bridges.Constraint.add_all_bridges(optimizer, Float64)
    MOI.Bridges.Objective.add_all_bridges(optimizer, Float64)
    return optimizer
end

function configure_solver!(
    model::JuMP.Model,
    solver::Symbol,
    tolerance::Float64,
    time_limit::Float64,
    verbose::Int,
    solver_log::AbstractString,
)
    attributes = if solver == :cupdcs
        (
            "verbose" => verbose,
            "time_limit_secs" => time_limit,
            "abs_tol" => tolerance,
            "rel_tol" => tolerance,
            "check_terminate_freq" => 1_000,
            "print_freq" => 10_000,
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
            "logfile" => solver_log,
        )
    elseif solver == :cuscs
        (
            "linear_solver" => SCS.GpuIndirectSolver,
            "eps_abs" => tolerance,
            "eps_rel" => tolerance,
            "time_limit_secs" => time_limit,
            "max_iters" => 2_000_000_000,
            "verbose" => verbose > 0,
        )
    else
        (
            "direct_solve_method" => :cudss,
            "tol_gap_abs" => tolerance,
            "tol_gap_rel" => tolerance,
            "tol_feas" => tolerance,
            "tol_infeas_abs" => tolerance,
            "tol_infeas_rel" => tolerance,
            "time_limit" => time_limit,
            "max_iter" => 2_000_000_000,
            "verbose" => verbose > 0,
        )
    end
    for (name, value) in attributes
        set_optimizer_attribute(model, name, value)
    end
    return model
end

function build_model(
    instance::LassoInstance,
    solver::Symbol,
    tolerance::Float64,
    time_limit::Float64,
    verbose::Int,
    solver_log::AbstractString,
)
    model = if solver == :cupdcs
        Model(PDCS_GPU.Optimizer)
    elseif solver == :cuscs
        Model(scs_gpu_optimizer; add_bridges = false)
    else
        Model(Clarabel.Optimizer)
    end
    configure_solver!(model, solver, tolerance, time_limit, verbose, solver_log)
    m, n = size(instance.A)
    @variable(model, x[1:n])
    @variable(model, u[1:n] >= 0.0)
    @variable(model, r)
    @objective(model, Min, 2.0 * r + instance.lambda * sum(u))
    @constraint(model, x .<= u)
    @constraint(model, -x .<= u)
    @constraint(
        model,
        vcat(
            (1.0 + r) / sqrt(2.0),
            (1.0 - r) / sqrt(2.0),
            instance.A * x - instance.b,
        ) in SecondOrderCone(),
    )
    return (; model, x, u, r, m, n)
end

synchronize_solver(solver::Symbol) =
    solver in (:cupdcs, :cuclarabel) ? CUDA.synchronize() : nothing

function maybe_get(model::JuMP.Model, attribute, default)
    try
        return MOI.get(JuMP.unsafe_backend(model), attribute)
    catch
        return default
    end
end

function solver_iterations(model::JuMP.Model, solver::Symbol)
    attribute = if solver == :cupdcs
        PDCS_GPU.PDHGIterations()
    elseif solver == :cuscs
        SCS.ADMMIterations()
    else
        MOI.BarrierIterations()
    end
    return maybe_get(model, attribute, -1)
end

function record_cupdcs_metrics!(result::Dict{String,Any}, model::JuMP.Model)
    metrics = maybe_get(model, MOI.RawOptimizerAttribute("result_metrics"), nothing)
    metrics === nothing && return result
    for field in (
        :exit_code,
        :exit_status,
        :solve_time_sec,
        :projection_time_sec,
        :primal_projection_time_sec,
        :dual_slack_projection_time_sec,
        :preprocessing_time_sec,
        :iterations,
        :l_inf_rel_primal_res,
        :l_inf_rel_dual_res,
        :l_2_rel_primal_res,
        :l_2_rel_dual_res,
        :relative_gap,
        :restart_count,
    )
        hasproperty(metrics, field) || continue
        value = getproperty(metrics, field)
        result["solver_$(field)"] = value isa Symbol ? string(value) : value
    end
    return result
end

function comparison_metrics(instance::LassoInstance, built)
    x = value.(built.x)
    u = value.(built.u)
    r = value(built.r)
    all(isfinite, x) || error("solver x contains nonfinite values")
    all(isfinite, u) || error("solver u contains nonfinite values")
    isfinite(r) || error("solver r is nonfinite")

    lasso = lasso_metrics(instance, x)
    residual = instance.A * x - instance.b
    epigraph_violation = max(
        maximum(abs.(x) .- u; init = 0.0),
        maximum(-u; init = 0.0),
        0.0,
    )
    epigraph_scale = 1.0 + max(
        maximum(abs, x; init = 0.0),
        maximum(abs, u; init = 0.0),
    )
    cone_head = (1.0 + r) / sqrt(2.0)
    cone_tail_norm = hypot((1.0 - r) / sqrt(2.0), norm(residual))
    soc_violation = max(cone_tail_norm - cone_head, 0.0)
    soc_scale = 1.0 + abs(cone_head) + cone_tail_norm
    primal_abs = max(epigraph_violation, soc_violation)
    primal_rel = max(
        epigraph_violation / epigraph_scale,
        soc_violation / soc_scale,
    )
    primal_objective = try
        objective_value(built.model)
    catch
        2.0 * r + instance.lambda * sum(u)
    end
    dual_objective = try
        dual_objective_value(built.model)
    catch
        NaN
    end
    gap_abs = isfinite(dual_objective) ?
        abs(primal_objective - dual_objective) : NaN
    gap_rel = isfinite(gap_abs) ? gap_abs /
        (1.0 + abs(primal_objective) + abs(dual_objective)) : NaN
    kkt_max = all(isfinite, (primal_rel, lasso.normalized_stationarity, gap_rel)) ?
        max(primal_rel, lasso.normalized_stationarity, gap_rel) : NaN
    original_objective_mismatch = abs(lasso.objective - primal_objective) /
        (1.0 + abs(lasso.objective) + abs(primal_objective))
    return (;
        comparison_primal_infeasibility_abs = primal_abs,
        comparison_primal_infeasibility_rel = primal_rel,
        comparison_dual_infeasibility_abs = lasso.stationarity,
        comparison_dual_infeasibility_rel = lasso.normalized_stationarity,
        comparison_gap_abs = gap_abs,
        comparison_relative_gap = gap_rel,
        comparison_kkt_max = kkt_max,
        primal_objective,
        dual_objective,
        original_objective = lasso.objective,
        original_objective_mismatch,
        residual_norm = lasso.residual_norm,
        xstar_relative_error = lasso.xstar_relative_error,
    )
end

function solve_case_main(args = ARGS)
    solver = Symbol(lowercase(option("solver", ""; args)))
    solver in (:cupdcs, :cuscs, :cuclarabel) || error(
        "--solver must be cupdcs, cuscs, or cuclarabel",
    )
    manifest_option = option("manifest", ""; args)
    instance_id = option("instance-id", ""; args)
    output_option = option("output", ""; args)
    isempty(manifest_option) && error("--manifest is required")
    isempty(output_option) && error("--output is required")
    manifest_path = abspath(manifest_option)
    output_path = abspath(output_option)
    solver_log = abspath(option(
        "solver-log",
        replace(output_path, r"\.toml$" => ".solver.log");
        args,
    ))
    repo_root = abspath(option(
        "pdcs-root",
        normpath(joinpath(@__DIR__, "..", ".."));
        args,
    ))
    tolerance = parse(Float64, option("tolerance", "1e-6"; args))
    time_limit = parse(Float64, option("time-limit", "3600"; args))
    verbose = parse(Int, option("verbose", "2"; args))
    VERSION == v"1.10.4" || error("formal campaign requires Julia 1.10.4")
    isfile(manifest_path) || error("missing --manifest: $manifest_path")
    isempty(instance_id) && error("--instance-id is required")
    tolerance > 0 || error("tolerance must be positive")
    time_limit > 0 || error("time limit must be positive")
    verbose in 0:2 || error("verbose must be 0, 1, or 2")

    result = Dict{String,Any}(
        "schema_version" => 1,
        "solver" => string(solver),
        "instance_id" => instance_id,
        "manifest" => manifest_path,
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "tolerance" => tolerance,
        "time_limit_seconds" => time_limit,
        "solution_vectors_saved" => false,
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "run_status" => "runtime_error",
        "termination_status" => "EXCEPTION",
        "error" => "",
    )
    exit_code = 1
    started = time()
    try
        manifest = TOML.parsefile(manifest_path)
        manifest["julia_version"] == string(VERSION) || error(
            "manifest Julia version mismatch",
        )
        entries = filter(
            entry -> entry["id"] == instance_id,
            manifest["instances"],
        )
        length(entries) == 1 || error("instance id is not unique in manifest")
        entry = only(entries)
        cache_path = joinpath(dirname(manifest_path), entry["cache"])
        isfile(cache_path) || error("missing serialized instance: $cache_path")
        instance = deserialize(cache_path)
        instance isa LassoInstance || error("cache has unexpected type")
        hashes = instance_hashes(instance)
        hashes.matrix == entry["matrix_sha256"] || error("matrix hash mismatch")
        hashes.b == entry["b_sha256"] || error("b hash mismatch")
        hashes.xstar == entry["xstar_sha256"] || error("xstar hash mismatch")
        hashes.eigenvalues == entry["eigenvalues_sha256"] || error(
            "eigenvalue hash mismatch",
        )
        verification = verify_instance(instance; compute_spectrum = false)
        verification.dense_nonzeros == 1_000_000 || error(
            "formal A is not a dense 1000-by-1000 matrix",
        )
        result["condition_number"] = instance.condition_number
        result["dimension"] = size(instance.A, 1)
        result["dense_nonzeros"] = verification.dense_nonzeros
        result["matrix_sha256"] = hashes.matrix
        result["b_sha256"] = hashes.b
        result["xstar_sha256"] = hashes.xstar
        result["eigenvalues_sha256"] = hashes.eigenvalues
        result["minimum_eigenvalue"] = last(instance.eigenvalues)
        result["maximum_eigenvalue"] = first(instance.eigenvalues)
        result["eigenvalue_spacing"] = "arithmetic_descending"
        result["lambda"] = instance.lambda
        result["lambda_ratio"] = instance.lambda_ratio

        load_solver!(solver, repo_root)
        result["gpu_name"] = visible_h100()
        mkpath(dirname(solver_log))
        Base.invokelatest() do
            built = nothing
            result["setup_seconds"] = @elapsed built = build_model(
                instance,
                solver,
                tolerance,
                time_limit,
                verbose,
                solver_log,
            )
            synchronize_solver(solver)
            result["optimize_wall_seconds"] = @elapsed begin
                optimize!(built.model)
                synchronize_solver(solver)
            end
            result["termination_status"] = string(termination_status(built.model))
            result["primal_status"] = string(primal_status(built.model))
            result["dual_status"] = string(dual_status(built.model))
            result["iterations"] = solver_iterations(built.model, solver)
            result["native_solve_seconds"] = maybe_get(
                built.model,
                MOI.SolveTimeSec(),
                NaN,
            )
            solver == :cupdcs && record_cupdcs_metrics!(result, built.model)
            if has_values(built.model)
                metrics = comparison_metrics(instance, built)
                for field in fieldnames(typeof(metrics))
                    result[string(field)] = getfield(metrics, field)
                end
                result["criteria_accepted"] =
                    isfinite(metrics.comparison_kkt_max) &&
                    metrics.comparison_kkt_max <= tolerance
                result["run_status"] = "returned"
            else
                result["criteria_accepted"] = false
                result["run_status"] = "no_solution"
            end
        end
        exit_code = 0
    catch error_value
        result["error"] = sprint(showerror, error_value, catch_backtrace())
        showerror(stderr, error_value, catch_backtrace())
        println(stderr)
    finally
        result["wall_seconds"] = time() - started
        atomic_toml(output_path, result)
        println(
            "ILL_CONDITION_LASSO_FINISH solver=$solver instance=$instance_id " *
            "status=$(result["run_status"]) " *
            "termination=$(result["termination_status"]) result=$output_path",
        )
    end
    return exit_code
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(solve_case_main())
end
