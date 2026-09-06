#!/usr/bin/env julia

using Dates
using JuMP
using LinearAlgebra
using SparseArrays
using TOML

import MathOptInterface as MOI

const BENCHMARK_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_DIR, "..", ".."))

function parse_cli(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        key = arguments[index]
        startswith(key, "--") || error("unexpected positional argument: $key")
        index == length(arguments) && error("missing value for $key")
        haskey(options, key) && error("duplicate option: $key")
        options[key] = arguments[index + 1]
        index += 2
    end
    for key in ("--solver", "--config", "--instance-id", "--result")
        haskey(options, key) || error("missing required option $key")
    end
    solver = Symbol(lowercase(options["--solver"]))
    solver in (:cupdcs, :cuscs, :cuclarabel) ||
        error("--solver must be cupdcs, cuscs, or cuclarabel")
    tolerance = parse(Float64, get(options, "--tolerance", "1e-6"))
    time_limit = parse(Float64, get(options, "--time-limit", "3600"))
    workers = parse(Int, get(options, "--workers", string(Threads.nthreads())))
    verbose = parse(Int, get(options, "--verbose", "1"))
    tolerance > 0 || error("--tolerance must be positive")
    time_limit > 0 || error("--time-limit must be positive")
    workers > 0 || error("--workers must be positive")
    verbose in 0:2 || error("--verbose must be 0, 1, or 2")
    return (
        solver,
        config = abspath(options["--config"]),
        instance_id = options["--instance-id"],
        result = abspath(options["--result"]),
        tolerance,
        time_limit,
        workers,
        verbose,
    )
end

const OPTIONS = parse_cli(ARGS)

include(joinpath(BENCHMARK_DIR, "large_scale_lasso.jl"))

if OPTIONS.solver == :cupdcs
    @eval using CUDA
    Base.include(Main, joinpath(REPO_ROOT, "src", "pdcs_gpu", "PDCS_GPU.jl"))
    include(joinpath(BENCHMARK_DIR, "..", "libsvm_lasso", "realistic_lasso.jl"))
    @eval using .RealisticLasso
elseif OPTIONS.solver == :cuscs
    @eval using SCS
    @eval using SCS_GPU_jll
else
    @eval using CUDA
    @eval using Clarabel
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

function selected_entry(config_path::AbstractString, instance_id::AbstractString)
    manifest = TOML.parsefile(config_path)
    get(manifest, "generator_version", nothing) == GENERATOR_VERSION ||
        error("generator version mismatch")
    entries = filter(entry -> entry["id"] == instance_id, manifest["instances"])
    length(entries) == 1 || error("instance '$instance_id' was not found exactly once")
    return only(entries), manifest
end

function visible_h100()
    output = read(
        `nvidia-smi --query-gpu=name --format=csv,noheader`,
        String,
    )
    names = filter(!isempty, strip.(split(chomp(output), '\n')))
    length(names) == 1 ||
        error("expected exactly one Slurm-visible GPU, found $(length(names)): $names")
    name = only(names)
    occursin("H100", name) || error("expected an H100, found '$name'")
    return name
end

function scs_gpu_optimizer()
    SCS.is_available(SCS.GpuIndirectSolver) ||
        error("SCS.GpuIndirectSolver is unavailable")
    raw = SCS.Optimizer()
    MOI.set(
        raw,
        MOI.RawOptimizerAttribute("linear_solver"),
        SCS.GpuIndirectSolver,
    )
    cache = MOI.default_cache(raw, Float64)
    cached = MOI.Utilities.CachingOptimizer(cache, raw)
    optimizer = MOI.Bridges.LazyBridgeOptimizer(cached)
    MOI.Bridges.Variable.add_all_bridges(optimizer, Float64)
    MOI.Bridges.Constraint.add_all_bridges(optimizer, Float64)
    MOI.Bridges.Objective.add_all_bridges(optimizer, Float64)
    return optimizer
end

function configure!(model::JuMP.Model, options)
    attributes = if options.solver == :cupdcs
        (
            "verbose" => options.verbose,
            "time_limit_secs" => options.time_limit,
            "abs_tol" => options.tolerance,
            "rel_tol" => options.tolerance,
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
            "logfile" => nothing,
        )
    elseif options.solver == :cuscs
        (
            "linear_solver" => SCS.GpuIndirectSolver,
            "eps_abs" => options.tolerance,
            "eps_rel" => options.tolerance,
            "time_limit_secs" => options.time_limit,
            "max_iters" => 2_000_000_000,
            "verbose" => options.verbose > 0,
        )
    else
        (
            "direct_solve_method" => :cudss,
            "tol_gap_abs" => options.tolerance,
            "tol_gap_rel" => options.tolerance,
            "tol_feas" => options.tolerance,
            "tol_infeas_abs" => options.tolerance,
            "tol_infeas_rel" => options.tolerance,
            "time_limit" => options.time_limit,
            "max_iter" => 2_000_000_000,
            "verbose" => options.verbose > 0,
        )
    end
    for (name, value) in attributes
        set_optimizer_attribute(model, name, value)
    end
    return model
end

function build_external_model(A, b, lambda, options)
    m, n = size(A)
    model = options.solver == :cuscs ?
        Model(scs_gpu_optimizer; add_bridges = false) :
        Model(Clarabel.Optimizer)
    configure!(model, options)
    @variable(model, x[1:n])
    @variable(model, u[1:n] >= 0.0)
    @variable(model, r)
    @objective(model, Min, 2.0 * r + lambda * sum(u))
    @constraint(model, x .<= u)
    @constraint(model, -x .<= u)
    @constraint(
        model,
        vcat(
            (1.0 + r) / sqrt(2.0),
            (1.0 - r) / sqrt(2.0),
            A * x - b,
        ) in SecondOrderCone(),
    )
    return (; model, x, u, r, optimizer = nothing, conic = nothing)
end

function build_cupdcs_model(A, b, lambda, instance_id, options)
    lasso_data = RealisticLasso.LassoData(
        instance_id,
        A,
        b,
        lambda,
        size(A, 2),
        nothing,
    )
    conic = RealisticLasso.build_lasso_conic_data(
        lasso_data;
        penalty = lambda,
        index_type = Int32,
        workers = options.workers,
    )
    optimizer = PDCS_GPU.Optimizer()
    model = PDCS_GPU.model_from_conic_data(conic; optimizer)
    configure!(model, options)
    return (; model, x = nothing, u = nothing, r = nothing, optimizer, conic)
end

function synchronize_solver(solver)
    solver in (:cupdcs, :cuclarabel) && CUDA.synchronize()
    return nothing
end

function solution_vectors(built, n, solver)
    if solver == :cupdcs
        primal = built.optimizer.sol.primal
        length(primal) == 2n + 1 || error(
            "cuPDCS returned $(length(primal)) primal values; expected $(2n + 1)",
        )
        return (
            x = copy(@view primal[1:n]),
            u = copy(@view primal[(n + 1):(2n)]),
            r = primal[end],
        )
    end
    return (
        x = value.(built.x),
        u = value.(built.u),
        r = value(built.r),
    )
end

function validate_solution(A, b, lambda, solution)
    x, u, r = solution.x, solution.u, solution.r
    all(isfinite, x) || error("solution x contains nonfinite values")
    all(isfinite, u) || error("solution u contains nonfinite values")
    isfinite(r) || error("solution r is nonfinite")

    residual = A * x - b
    all(isfinite, residual) || error("solution residual contains nonfinite values")
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
    normalized_violation = max(
        epigraph_violation / epigraph_scale,
        soc_violation / soc_scale,
    )
    original_objective = dot(residual, residual) + lambda * norm(x, 1)
    conic_objective = 2.0 * r + lambda * sum(u)
    objective_mismatch = abs(original_objective - conic_objective) /
        (1.0 + abs(original_objective) + abs(conic_objective))
    return (;
        epigraph_violation,
        soc_violation,
        normalized_violation,
        original_objective,
        conic_objective,
        objective_mismatch,
    )
end

function maybe_get(model, attribute, default)
    try
        return MOI.get(backend(model), attribute)
    catch
        return default
    end
end

function iteration_count(model, solver)
    attribute = if solver == :cupdcs
        PDCS_GPU.PDHGIterations()
    elseif solver == :cuscs
        SCS.ADMMIterations()
    else
        MOI.BarrierIterations()
    end
    return maybe_get(model, attribute, -1)
end

function acceptable_status(model)
    termination = termination_status(model)
    primal = primal_status(model)
    good_termination = termination in (
        MOI.OPTIMAL,
        MOI.ALMOST_OPTIMAL,
        MOI.LOCALLY_SOLVED,
        MOI.ALMOST_LOCALLY_SOLVED,
    )
    good_primal = primal in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
    return good_termination && good_primal
end

function main()
    started = now(UTC)
    result = Dict{String,Any}(
        "schema_version" => 1,
        "solver" => string(OPTIONS.solver),
        "instance_id" => OPTIONS.instance_id,
        "config" => OPTIONS.config,
        "tolerance" => OPTIONS.tolerance,
        "validation_tolerance" => OPTIONS.tolerance,
        "time_limit_seconds" => OPTIONS.time_limit,
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "slurm_array_task_id" => get(ENV, "SLURM_ARRAY_TASK_ID", ""),
        "started_utc" => string(started),
        "run_status" => "failed",
        "termination_status" => "EXCEPTION",
        "primal_status" => "NO_SOLUTION",
        "dual_status" => "NO_SOLUTION",
        "generation_seconds" => NaN,
        "setup_seconds" => NaN,
        "solve_seconds" => NaN,
        "iterations" => -1,
        "error" => "",
    )
    exit_code = 1
    try
        gpu_name = visible_h100()
        result["gpu_name"] = gpu_name
        entry, manifest = selected_entry(OPTIONS.config, OPTIONS.instance_id)
        result["manifest_julia_version"] = get(manifest, "julia_version", "")
        result["m"] = entry["m"]
        result["n"] = entry["n"]
        result["density"] = entry["density"]
        result["replicate"] = entry["replicate"]
        result["seed"] = entry["seed"]
        println(
            "LASSO_CASE_START solver=$(OPTIONS.solver) id=$(OPTIONS.instance_id) " *
            "m=$(entry["m"]) n=$(entry["n"]) gpu=$gpu_name " *
            "tolerance=$(OPTIONS.tolerance) " *
            "validation_tolerance=$(OPTIONS.tolerance)",
        )

        generated = nothing
        result["generation_seconds"] = @elapsed generated = generate_instance(entry)
        A = generated.A
        b = generated.b
        lambda = generated.lambda
        result["nnz"] = nnz(A)
        result["lambda"] = lambda
        result["numerical_digest"] = numerical_digest(generated)
        generated = nothing
        GC.gc(true)

        built = nothing
        result["setup_seconds"] = @elapsed built = if OPTIONS.solver == :cupdcs
            build_cupdcs_model(A, b, lambda, OPTIONS.instance_id, OPTIONS)
        else
            build_external_model(A, b, lambda, OPTIONS)
        end
        synchronize_solver(OPTIONS.solver)
        result["solve_seconds"] = @elapsed begin
            optimize!(built.model)
            synchronize_solver(OPTIONS.solver)
        end
        result["termination_status"] = string(termination_status(built.model))
        result["primal_status"] = string(primal_status(built.model))
        result["dual_status"] = string(dual_status(built.model))
        result["iterations"] = iteration_count(built.model, OPTIONS.solver)
        result["solver_solve_seconds"] = maybe_get(
            built.model,
            MOI.SolveTimeSec(),
            NaN,
        )
        if OPTIONS.solver == :cupdcs
            solver_solution = built.optimizer.sol
            result["solver_l_inf_rel_primal_res"] =
                solver_solution.l_inf_rel_primal_res
            result["solver_l_inf_rel_dual_res"] =
                solver_solution.l_inf_rel_dual_res
            result["solver_relative_gap"] = solver_solution.relative_gap
            result["solver_relative_kkt_max"] = max(
                solver_solution.l_inf_rel_primal_res,
                solver_solution.l_inf_rel_dual_res,
                solver_solution.relative_gap,
            )
        end

        solution = solution_vectors(built, size(A, 2), OPTIONS.solver)
        validation = validate_solution(A, b, lambda, solution)
        for field in fieldnames(typeof(validation))
            result[string(field)] = getfield(validation, field)
        end
        result["objective_value"] = try
            objective_value(built.model)
        catch
            validation.conic_objective
        end
        result["status_accepted"] = acceptable_status(built.model)
        result["validation_accepted"] =
            validation.normalized_violation <= result["validation_tolerance"]
        result["solver_tolerance_accepted"] =
            OPTIONS.solver != :cupdcs ||
            result["solver_relative_kkt_max"] <= OPTIONS.tolerance
        passed = result["status_accepted"] && result["validation_accepted"] &&
            result["solver_tolerance_accepted"] && isfinite(result["objective_value"])
        result["run_status"] = passed ? "passed" : "failed"
        exit_code = passed ? 0 : 1
        println(
            "LASSO_CASE_RESULT solver=$(OPTIONS.solver) id=$(OPTIONS.instance_id) " *
            "status=$(result["run_status"]) " *
            "termination=$(result["termination_status"]) " *
            "normalized_violation=$(result["normalized_violation"]) " *
            "solve_seconds=$(result["solve_seconds"])",
        )
    catch error_value
        result["error"] = sprint(showerror, error_value, catch_backtrace())
        println(stderr, "LASSO_CASE_EXCEPTION solver=$(OPTIONS.solver) id=$(OPTIONS.instance_id)")
        showerror(stderr, error_value, catch_backtrace())
        println(stderr)
    finally
        finished = now(UTC)
        result["finished_utc"] = string(finished)
        result["wall_seconds"] = Dates.value(finished - started) / 1_000.0
        atomic_toml(OPTIONS.result, result)
        println(
            "LASSO_CASE_FINISH solver=$(OPTIONS.solver) id=$(OPTIONS.instance_id) " *
            "result=$(OPTIONS.result)",
        )
    end
    return exit_code
end

exit(main())
