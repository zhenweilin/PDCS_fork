#!/usr/bin/env julia

"""
Run the CuClarabel branch of Clarabel.jl on one cached Lasso instance.

CuClarabel is distributed as the `CuClarabel` branch of Clarabel.jl, but its
Julia module is still named `Clarabel`.  It must therefore run in the isolated
`benchmark/cuclarabel_env` project instead of the main PDCS project, which
contains the registered CPU release of Clarabel.
"""

using CUDA
using Clarabel
using JuMP
using LinearAlgebra
using Printf
using Serialization
using SparseArrays
import MathOptInterface as MOI

include("rebuttal/common.jl")
include("rebuttal/ill_conditioned_lasso_cases.jl")
using .RebuttalCommon
using .IllConditionedLassoCases

const CACHE = abspath(option("cache", ""))
const WARMUP_CACHE = option("warmup-cache", "")
const OUTPUT = abspath(option(
    "output-dir",
    "benchmark/results/rebuttal/ill_conditioned_lasso/cuclarabel",
))
const TOL = parse(Float64, option("tol", "1e-6"))
const TIME_LIMIT = parse(Float64, option("time-limit", "600"))
const VERBOSE_LEVEL = parse(Int, option("verbose-level", "2"))
const DIRECT_SOLVE_METHOD = Symbol(option("direct-solve-method", "cudss"))

isfile(CACHE) || error("--cache must name an existing serialized instance")
TOL > 0 || error("--tol must be positive")
TIME_LIMIT > 0 || error("--time-limit must be positive")
VERBOSE_LEVEL in 0:2 || error("--verbose-level must be 0, 1, or 2")
DIRECT_SOLVE_METHOD in (:cudss, :cudssmixed) ||
    error("--direct-solve-method must be cudss or cudssmixed")
CUDA.functional() || error("CUDA is not functional")

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

function cuda_runtime_string()
    try
        return string(CUDA.runtime_version())
    catch
        return "unknown"
    end
end

function cuda_driver_string()
    try
        return string(CUDA.driver_version())
    catch
        return "unknown"
    end
end

function cuda_device_name()
    try
        return CUDA.name(CUDA.device())
    catch
        return string(CUDA.device())
    end
end

function build_jump_model(inst)
    m, n = size(inst.A)
    model = Model(Clarabel.Optimizer)

    # Do not silently ignore these attributes.  In particular,
    # direct_solve_method=:cudss is what selects the GPU implementation.
    set_optimizer_attribute(model, "direct_solve_method", DIRECT_SOLVE_METHOD)
    set_optimizer_attribute(model, "tol_gap_abs", TOL)
    set_optimizer_attribute(model, "tol_gap_rel", TOL)
    set_optimizer_attribute(model, "tol_feas", TOL)
    set_optimizer_attribute(model, "time_limit", TIME_LIMIT)
    set_optimizer_attribute(model, "verbose", VERBOSE_LEVEL > 0)

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

function warmup_solver(inst)
    println("WARMUP_START solver=cuclarabel cache=$WARMUP_CACHE")
    started = time()
    model, _, _ = build_jump_model(inst)
    CUDA.@sync optimize!(model)
    println(
        "WARMUP_COMPLETE solver=cuclarabel status=$(termination_status(model)) " *
        "seconds=$(time() - started)",
    )
    return nothing
end

function main()
if !isempty(WARMUP_CACHE)
    isfile(WARMUP_CACHE) ||
        error("--warmup-cache does not exist: $WARMUP_CACHE")
    warmup_solver(deserialize(WARMUP_CACHE))
end
input_started = time()
inst = deserialize(CACHE)
input_seconds = time() - input_started
hashes = instance_hashes(inst)
device_name = cuda_device_name()
cuda_runtime = cuda_runtime_string()
cuda_driver = cuda_driver_string()

println(
    "CUCLARABEL_ENV module=Clarabel package_version=$(pkgversion(Clarabel)) " *
    "direct_solve_method=$DIRECT_SOLVE_METHOD cuda_functional=$(CUDA.functional()) " *
    "device=$(repr(device_name)) runtime=$cuda_runtime driver=$cuda_driver",
)
println(
    "INSTANCE cache=$CACHE seed=$(inst.seed) K=$(inst.K) m=$(size(inst.A, 1)) " *
    "n=$(size(inst.A, 2)) nnz=$(nnz(inst.A)) matrix_hash=$(hashes.matrix) " *
    "b_hash=$(hashes.b) xstar_hash=$(hashes.xstar) " *
    "input_seconds=$input_seconds",
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
model = nothing

cold_started = time()
try
    model, xpos, xneg = build_jump_model(inst)
    objective_type = MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    println(
        "JUMP_MODEL solver=cuclarabel backend=$(typeof(backend(model))) " *
        "affine_objective=$(MOI.supports(backend(model), objective_type)) " *
        "direct_solve_method=$DIRECT_SOLVE_METHOD tolerance=$TOL",
    )
    setup_seconds = time() - cold_started
    optimize_started = time()
    CUDA.@sync optimize!(model)
    optimize_wall_seconds = time() - optimize_started
    term = termination_status(model)
    termination = string(term)
    message = termination
    iterations = maybe_get(model, MOI.BarrierIterations(), missing)
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

result_path = joinpath(OUTPUT, "cuclarabel_results.csv")
write_result(
    result_path,
    (
        "seed", "panel", "K", "solver", "backend", "tolerance", "status",
        "cold_seconds", "setup_seconds", "optimize_wall_seconds",
        "native_solve_seconds", "iterations", "objective",
        "normalized_kkt", "x_error", "objective_error", "precision", "recall",
        "termination_status", "device", "cuda_runtime", "cuda_driver",
        "clarabel_version", "matrix_hash", "b_hash", "xstar_hash", "message",
        "input_seconds",
    ),
    (
        inst.seed, inst.panel, inst.K, "cuclarabel", DIRECT_SOLVE_METHOD, TOL,
        status, cold_seconds, setup_seconds, optimize_wall_seconds,
        native_solve_seconds, iterations, objective,
        metric === nothing ? NaN : metric.normalized_stationarity,
        metric === nothing ? NaN : metric.x_error,
        metric === nothing ? NaN : metric.objective_error,
        metric === nothing ? NaN : metric.precision,
        metric === nothing ? NaN : metric.recall,
        termination,
        device_name, cuda_runtime, cuda_driver, pkgversion(Clarabel),
        hashes.matrix, hashes.b, hashes.xstar, message,
        input_seconds,
    ),
)

println(
    "SOLVE_COMPLETE solver=cuclarabel backend=$DIRECT_SOLVE_METHOD " *
    "status=$status cold_seconds=$cold_seconds " *
    "setup_seconds=$setup_seconds optimize_wall_seconds=$optimize_wall_seconds " *
    "native_solve_seconds=$native_solve_seconds iterations=$iterations " *
    "objective=$objective normalized_kkt=" *
    "$(metric === nothing ? NaN : metric.normalized_stationarity) " *
    "x_error=$(metric === nothing ? NaN : metric.x_error) " *
    "result=$result_path",
)

status == "numerical_failure" && exit(1)
end

main()
