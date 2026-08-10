#!/usr/bin/env julia

using CUDA
using Dates
using JSON
using Logging
using SHA
import MathOptInterface as MOI


function cli_value(name::String; default = nothing)
    position = findfirst(==(name), ARGS)
    if position === nothing
        default === nothing && error("missing required option $name")
        return default
    end
    position == length(ARGS) && error("missing value for $name")
    return ARGS[position + 1]
end


const INPUT_PATH = abspath(cli_value("--input"))
const OUTPUT_DIR = abspath(cli_value("--output-dir"))
const TIME_LIMIT = parse(Float64, cli_value("--time-limit"; default = "3600"))
const TOLERANCE = parse(Float64, cli_value("--tolerance"; default = "1e-6"))
const PRINT_FREQUENCY = parse(Int, cli_value("--print-frequency"; default = "20000"))
const DEVICE = lowercase(cli_value("--device"; default = "gpu"))
const USE_AGGRESSIVE =
    lowercase(cli_value("--use-aggressive"; default = "true")) == "true"
const MODEL_LOADER = lowercase(cli_value("--model-loader"; default = "moi"))
const ROOT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const OLD_SOURCE_DIR = abspath(
    cli_value(
        "--old-source-dir";
        default = joinpath(ROOT_DIR, "external", "PDCS-main"),
    ),
)

isfile(INPUT_PATH) || error("missing input CBF: $INPUT_PATH")
DEVICE in ("cpu", "gpu") || error("--device must be cpu or gpu")
MODEL_LOADER in ("moi", "cbf") || error("--model-loader must be moi or cbf")
mkpath(OUTPUT_DIR)

# Load the archived implementation directly. This avoids importing the current
# PDCS package and leaves the archived source tree unchanged.
const COMBINED_MODULE_PATH = joinpath(OLD_SOURCE_DIR, "src", "PDCS.jl")
if isfile(COMBINED_MODULE_PATH)
    include(COMBINED_MODULE_PATH)
    const OLD_SOLVER = DEVICE == "gpu" ? Main.PDCS.PDCS_GPU : Main.PDCS.PDCS_CPU
else
    const MODERN_STANDALONE_MODULE_PATH = joinpath(
        OLD_SOURCE_DIR,
        "src",
        DEVICE == "gpu" ? "pdcs_gpu" : "pdcs_cpu",
        DEVICE == "gpu" ? "PDCS_GPU.jl" : "PDCS_CPU.jl",
    )
    const LEGACY_STANDALONE_MODULE_PATH = joinpath(
        OLD_SOURCE_DIR,
        "code",
        "src",
        DEVICE == "gpu" ? "rpdhg_clp_gpu" : "rpdhg_clp_cpu",
        DEVICE == "gpu" ? "RPDHG_CLP_GPU.jl" : "RPDHG_CLP_CPU.jl",
    )
    const STANDALONE_MODULE_PATH =
        isfile(MODERN_STANDALONE_MODULE_PATH) ?
        MODERN_STANDALONE_MODULE_PATH : LEGACY_STANDALONE_MODULE_PATH
    isfile(STANDALONE_MODULE_PATH) ||
        error("missing archived solver module: $STANDALONE_MODULE_PATH")
    include(STANDALONE_MODULE_PATH)
    if STANDALONE_MODULE_PATH == MODERN_STANDALONE_MODULE_PATH
        const OLD_SOLVER = DEVICE == "gpu" ? Main.PDCS_GPU : Main.PDCS_CPU
    else
        const OLD_SOLVER =
            DEVICE == "gpu" ? Main.RPDHG_CLP_GPU : Main.RPDHG_CLP_CPU
    end
end

if DEVICE == "gpu" &&
   isdefined(OLD_SOLVER, :few_block_proj_ptr) &&
   OLD_SOLVER.few_block_proj_ptr[] == C_NULL &&
   isdefined(OLD_SOLVER, :__init__)
    OLD_SOLVER.__init__()
end


function load_cbf(path::String)
    if MODEL_LOADER == "moi"
        # Match the archived benchmark scripts exactly. In particular, this
        # preserves their MOI bridge/copy path instead of copying a CBF.Model
        # directly into the optimizer.
        model = MOI.Utilities.Model{Float64}()
        MOI.read_from_file(model, path)
        return model
    else
        model = MOI.FileFormats.CBF.Model()
        open(`gzip -dc $path`) do stream
            read!(stream, model)
        end
        return model
    end
end


function pdcs_backend(optimizer)
    bridge_model = getfield(optimizer, :model)
    backend = getfield(bridge_model, :optimizer)
    backend isa OLD_SOLVER.Optimizer ||
        error("unexpected optimizer backend $(typeof(backend))")
    return backend
end


function final_metric(text::String, label::String)
    expression = Regex(label * raw":\s*([+\-0-9.eE]+)")
    matches = collect(eachmatch(expression, text))
    isempty(matches) && return nothing
    return tryparse(Float64, matches[end].captures[1])
end


function json_safe(value)
    value isa AbstractFloat && !isfinite(value) && return nothing
    value isa Symbol && return string(value)
    value isa MOI.TerminationStatusCode && return string(value)
    return value
end


function archived_git_commit(path::String)
    try
        return strip(read(Cmd(["git", "-C", path, "rev-parse", "HEAD"]), String))
    catch
        return nothing
    end
end


raw_log = joinpath(OUTPUT_DIR, "solver.raw.log")
result_path = joinpath(OUTPUT_DIR, "result.json")
started_utc = string(now(UTC))
result = Dict{String,Any}(
    "case" => replace(basename(INPUT_PATH), ".cbf.gz" => ""),
    "input" => INPUT_PATH,
    "input_sha256" => bytes2hex(open(sha256, INPUT_PATH)),
    "old_source_dir" => OLD_SOURCE_DIR,
    "old_source_commit" => archived_git_commit(OLD_SOURCE_DIR),
    "time_limit_seconds" => TIME_LIMIT,
    "tolerance" => TOLERANCE,
    "print_frequency" => PRINT_FREQUENCY,
    "verbose" => 2,
    "device" => DEVICE,
    "use_aggressive" => USE_AGGRESSIVE,
    "model_loader" => MODEL_LOADER,
    "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
    "gpu_name" => DEVICE == "gpu" ? CUDA.name(CUDA.device()) : nothing,
    "julia_threads" => Threads.nthreads(),
    "julia_version" => string(VERSION),
    "started_utc" => started_utc,
    "raw_log" => raw_log,
)

wall_start = time()
open(raw_log, "w") do stream
    logger = SimpleLogger(stream, Logging.Info)
    redirect_stdout(stream) do
        redirect_stderr(stream) do
            with_logger(logger) do
                println("OLD_PDCS_RAW_LOG_VERSION=1")
                println("OLD_SOURCE_DIR=$OLD_SOURCE_DIR")
                println("OLD_SOURCE_COMMIT=" * something(result["old_source_commit"], "unavailable"))
                println("INPUT=$INPUT_PATH")
                println("INPUT_SHA256=" * result["input_sha256"])
                println("TIME_LIMIT_SECONDS=$TIME_LIMIT")
                println("TOLERANCE=$TOLERANCE")
                println("PRINT_FREQUENCY=$PRINT_FREQUENCY")
                println("DEVICE=$DEVICE")
                println("USE_AGGRESSIVE=$USE_AGGRESSIVE")
                println("MODEL_LOADER=$MODEL_LOADER")
                println("JULIA_THREADS=$(Threads.nthreads())")
                println("CUDA_VISIBLE_DEVICES=" * get(ENV, "CUDA_VISIBLE_DEVICES", ""))
                println("STARTED_UTC=$started_utc")
                flush(stream)
                try
                    cbf_model = load_cbf(INPUT_PATH)
                    optimizer = MOI.instantiate(
                        OLD_SOLVER.Optimizer;
                        with_bridge_type = Float64,
                    )
                    MOI.copy_to(optimizer, cbf_model)
                    for (name, value) in (
                        "verbose" => 2,
                        "time_limit_secs" => TIME_LIMIT,
                        "abs_tol" => TOLERANCE,
                        "rel_tol" => TOLERANCE,
                        "print_freq" => PRINT_FREQUENCY,
                        "use_aggressive" => USE_AGGRESSIVE,
                        "use_accelerated" => false,
                        "use_resolving" => true,
                        "logfile" => nothing,
                    )
                        MOI.set(optimizer, MOI.RawOptimizerAttribute(name), value)
                    end
                    MOI.optimize!(optimizer)
                    DEVICE == "gpu" && CUDA.synchronize()
                    backend = pdcs_backend(optimizer)
                    termination = MOI.get(optimizer, MOI.TerminationStatus())
                    result["run_status"] = "COMPLETED"
                    result["termination_status"] = string(termination)
                    result["old_exit_status"] = string(backend.sol.exit_status)
                    result["old_exit_code"] = backend.sol.exit_code
                    result["iterations"] = backend.sol.iterations
                    result["solver_seconds"] = backend.sol.solve_time_sec
                    result["objective_value"] = json_safe(backend.sol.objective_value)
                    result["dual_objective_value"] =
                        json_safe(backend.sol.dual_objective_value)
                    result["self_reported_optimal"] = termination == MOI.OPTIMAL
                catch error_value
                    result["run_status"] = "RUNTIME_ERROR"
                    result["error"] = sprint(
                        showerror,
                        error_value,
                        catch_backtrace(),
                    )
                    println("OLD_PDCS_EXCEPTION")
                    showerror(stream, error_value, catch_backtrace())
                    println(stream)
                end
                flush(stream)
            end
        end
    end
end

raw_text = read(raw_log, String)
metrics = Dict{String,Any}(
    "l_inf_rel_primal_res" => final_metric(raw_text, "l_inf_rel_primal_res"),
    "l_inf_rel_dual_res" => final_metric(raw_text, "l_inf_rel_dual_res"),
    "relative_gap" => final_metric(raw_text, "rel_gap"),
)
finite_metrics = Float64[
    value for value in values(metrics) if value isa Float64 && isfinite(value)
]
metric_max = length(finite_metrics) == 3 ? maximum(finite_metrics) : nothing
result["metrics"] = metrics
result["verification_metric_max_from_printed_summary"] = metric_max
result["verified_solved"] =
    get(result, "self_reported_optimal", false) &&
    metric_max !== nothing && metric_max <= TOLERANCE
result["wall_seconds"] = time() - wall_start
result["finished_utc"] = string(now(UTC))

temporary = result_path * ".tmp"
open(temporary, "w") do stream
    JSON.print(stream, result, 2)
    write(stream, '\n')
end
mv(temporary, result_path; force = true)

case_name = result["case"]
run_status = result["run_status"]
termination_status = get(result, "termination_status", "NA")
verified_solved = result["verified_solved"]
println(
    "OLD_CASE_COMPLETE case=$case_name " *
    "status=$run_status " *
    "termination=$termination_status " *
    "verified=$verified_solved " *
    "result=$result_path",
)

get(result, "run_status", "RUNTIME_ERROR") == "COMPLETED" || exit(2)
