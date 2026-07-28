#!/usr/bin/env julia

"""
Run SCS's sparse GPU-indirect solver on one cached Lasso instance.

The official SCS_GPU_jll artifact currently targets CUDA 11.x.  This runner is
therefore isolated in benchmark/scs_gpu_env, whose LocalPreferences.toml pins
CUDA_Runtime_jll to 11.8 without changing the PDCS project's CUDA runtime.
"""

using JuMP
using LinearAlgebra
using Printf
using Serialization
using SparseArrays
using SCS
using SCS_GPU_jll
import MathOptInterface as MOI

include("rebuttal/common.jl")
include("rebuttal/ill_conditioned_lasso_cases.jl")
using .RebuttalCommon
using .IllConditionedLassoCases

const CACHE = abspath(option("cache", ""))
const WARMUP_CACHE = option("warmup-cache", "")
const OUTPUT = abspath(option(
    "output-dir",
    "benchmark/results/rebuttal/ill_conditioned_lasso/scs_gpu",
))
const TOL = parse(Float64, option("tol", "1e-6"))
const TIME_LIMIT = parse(Float64, option("time-limit", "600"))
const VERBOSE_LEVEL = parse(Int, option("verbose-level", "2"))

isfile(CACHE) || error("--cache must name an existing serialized instance")
TOL > 0 || error("--tol must be positive")
TIME_LIMIT > 0 || error("--time-limit must be positive")
VERBOSE_LEVEL in 0:2 || error("--verbose-level must be 0, 1, or 2")
SCS.is_available(SCS.GpuIndirectSolver) ||
    error("SCS.GpuIndirectSolver is unavailable in this environment")

function csv_field(value)
    text = replace(string(value), '"' => "\"\"")
    return string('"', text, '"')
end

function write_result(path, header, values)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(csv_field.(header), ','))
        println(io, join(csv_field.(values), ','))
    end
end

function maybe_get(model, attribute, default)
    try
        return MOI.get(backend(model), attribute)
    catch
        return default
    end
end

"""Create SCS's MOI cache after selecting the GPU solver.

GpuIndirectSolver uses Cint/Int32 indices, unlike SCS's default direct solver.
Selecting it after `MOI.default_cache` would make the cache Int64 and trigger
SCS's `T == scsint_t(linear_solver)` assertion.
"""
function gpu_optimizer()
    raw = SCS.Optimizer()
    MOI.set(
        raw,
        MOI.RawOptimizerAttribute("linear_solver"),
        SCS.GpuIndirectSolver,
    )
    cache = MOI.default_cache(raw, Float64)
    cached = MOI.Utilities.CachingOptimizer(cache, raw)
    bridged = MOI.Bridges.LazyBridgeOptimizer(cached)
    MOI.Bridges.Variable.add_all_bridges(bridged, Float64)
    MOI.Bridges.Constraint.add_all_bridges(bridged, Float64)
    MOI.Bridges.Objective.add_all_bridges(bridged, Float64)
    return bridged
end

function build_jump_model(inst)
    m, n = size(inst.A)
    model = Model(gpu_optimizer; add_bridges = false)
    set_optimizer_attribute(model, "linear_solver", SCS.GpuIndirectSolver)
    set_optimizer_attribute(model, "eps_abs", TOL)
    set_optimizer_attribute(model, "eps_rel", TOL)
    set_optimizer_attribute(model, "time_limit_secs", TIME_LIMIT)
    VERBOSE_LEVEL == 0 && set_silent(model)

    @variable(model, xpos[1:n] >= 0)
    @variable(model, xneg[1:n] >= 0)
    @variable(model, y[1:m])
    @variable(model, r >= 0)
    @objective(
        model,
        Min,
        2r + inst.lambda * sum(xpos[j] + xneg[j] for j in 1:n),
    )
    @constraint(model, y .== inst.A * (xpos - xneg) - inst.b)
    @constraint(
        model,
        [(1 + r) / sqrt(2); (1 - r) / sqrt(2); y] in SecondOrderCone(),
    )
    return model, xpos, xneg
end

function solve_warmup(inst)
    println("WARMUP_START solver=scs_gpu cache=$WARMUP_CACHE")
    started = time()
    model, _, _ = build_jump_model(inst)
    optimize!(model)
    println(
        "WARMUP_COMPLETE solver=scs_gpu status=$(termination_status(model)) " *
        "seconds=$(time() - started)",
    )
end

function main()
    if !isempty(WARMUP_CACHE)
        isfile(WARMUP_CACHE) ||
            error("--warmup-cache does not exist: $WARMUP_CACHE")
        solve_warmup(deserialize(WARMUP_CACHE))
    end

    input_started = time()
    inst = deserialize(CACHE)
    input_seconds = time() - input_started
    hashes = instance_hashes(inst)
    println(
        "SCS_GPU_ENV scs_version=$(pkgversion(SCS)) " *
        "gpu_available=$(SCS.is_available(SCS.GpuIndirectSolver)) " *
        "index_type=$(SCS.scsint_t(SCS.GpuIndirectSolver))",
    )
    println(
        "INSTANCE cache=$CACHE seed=$(inst.seed) K=$(inst.K) " *
        "m=$(size(inst.A, 1)) n=$(size(inst.A, 2)) nnz=$(nnz(inst.A)) " *
        "matrix_hash=$(hashes.matrix) b_hash=$(hashes.b) " *
        "xstar_hash=$(hashes.xstar) input_seconds=$input_seconds",
    )

    status = "numerical_failure"
    message = ""
    termination = "EXCEPTION"
    objective = NaN
    iterations = missing
    native_solve_seconds = NaN
    setup_seconds = NaN
    optimize_wall_seconds = NaN
    x = fill(NaN, size(inst.A, 2))

    cold_started = time()
    try
        model, xpos, xneg = build_jump_model(inst)
        objective_type =
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
        println(
            "JUMP_MODEL solver=scs_gpu backend=$(typeof(backend(model))) " *
            "affine_objective=$(MOI.supports(backend(model), objective_type)) " *
            "linear_solver=GpuIndirectSolver tolerance=$TOL",
        )
        setup_seconds = time() - cold_started
        optimize_started = time()
        optimize!(model)
        optimize_wall_seconds = time() - optimize_started
        term = termination_status(model)
        termination = string(term)
        message = termination
        iterations = maybe_get(model, SCS.ADMMIterations(), missing)
        native_solve_seconds = maybe_get(model, MOI.SolveTimeSec(), NaN)
        if term == MOI.TIME_LIMIT
            if has_values(model)
                x = value.(xpos) .- value.(xneg)
                objective = objective_value(model)
            end
            status = "timeout"
        elseif has_values(model)
            x = value.(xpos) .- value.(xneg)
            objective = objective_value(model)
            status = term in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ?
                "returned" : "returned_nonoptimal"
        else
            status = "no_solution"
        end
    catch err
        bt = catch_backtrace()
        message = sprint(showerror, err)
        showerror(stderr, err, bt)
        println(stderr)
    end
    cold_seconds = time() - cold_started

    metric = all(isfinite, x) ? lasso_metrics(inst, x) : nothing
    if status in ("returned", "returned_nonoptimal")
        status = (
            metric.normalized_stationarity <= TOL &&
            metric.x_error <= max(10TOL, 1e-4)
        ) ? "solved_verified" : "solver_claimed_solved_but_inaccurate"
    end

    result_path = joinpath(OUTPUT, "scs_gpu_results.csv")
    write_result(
        result_path,
        (
            "seed", "panel", "K", "solver", "backend", "tolerance",
            "status", "cold_seconds", "setup_seconds",
            "optimize_wall_seconds", "native_solve_seconds", "iterations",
            "objective", "normalized_kkt", "x_error", "objective_error",
            "precision", "recall", "termination_status", "scs_version",
            "matrix_hash", "b_hash", "xstar_hash", "message",
            "input_seconds",
        ),
        (
            inst.seed, inst.panel, inst.K, "scs_gpu", "gpu_indirect", TOL,
            status, cold_seconds, setup_seconds, optimize_wall_seconds,
            native_solve_seconds, iterations, objective,
            metric === nothing ? NaN : metric.normalized_stationarity,
            metric === nothing ? NaN : metric.x_error,
            metric === nothing ? NaN : metric.objective_error,
            metric === nothing ? NaN : metric.precision,
            metric === nothing ? NaN : metric.recall,
            termination, pkgversion(SCS), hashes.matrix, hashes.b,
            hashes.xstar, message, input_seconds,
        ),
    )
    println(
        "SOLVE_COMPLETE solver=scs_gpu backend=gpu_indirect status=$status " *
        "cold_seconds=$cold_seconds setup_seconds=$setup_seconds " *
        "optimize_wall_seconds=$optimize_wall_seconds " *
        "native_solve_seconds=$native_solve_seconds iterations=$iterations " *
        "objective=$objective normalized_kkt=" *
        "$(metric === nothing ? NaN : metric.normalized_stationarity) " *
        "x_error=$(metric === nothing ? NaN : metric.x_error) " *
        "result=$result_path",
    )
    status == "numerical_failure" && exit(1)
end

main()
