#!/usr/bin/env julia

using JuMP
using SparseArrays
using TOML

include(joinpath(@__DIR__, "realistic_lasso.jl"))
using .RealisticLasso

const ALPHAS = Float64[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
const DEFAULT_DATASET_IDS = ("news20", "E2006-log1p", "rcv1-train")
const DEFAULT_PDCS_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function parse_options(args::Vector{String})
    any(arg -> arg in ("-h", "--help", "help"), args) && return nothing
    options = Dict{String, String}()
    index = 1
    while index <= length(args)
        startswith(args[index], "--") || error("Unexpected argument: $(args[index])")
        index == length(args) && error("Option $(args[index]) requires a value.")
        options[args[index][3:end]] = args[index + 1]
        index += 2
    end
    allowed = Set([
        "dataset",
        "mode",
        "modeling",
        "workers",
        "pdcs-root",
        "raw-dir",
        "output-dir",
        "max-rows",
        "compact-zero-columns",
        "index-type",
        "time-limit",
        "rel-tol",
        "abs-tol",
        "verbose",
        "alpha",
    ])
    unknown = setdiff(Set(keys(options)), allowed)
    isempty(unknown) || error("Unknown options: $(join(sort!(collect(unknown)), ", "))")
    return options
end

function print_help()
    println("""
Usage:
  julia --project=. run_penalty_sweep.jl \\
      --dataset DATASET_ID|all \\
      [--mode build|pdcs-cpu|pdcs-gpu] \\
      [--modeling bulk|jump] [--workers N] \\
      [--pdcs-root /path/to/PDCS] [--raw-dir raw] [--output-dir results] \\
      [--max-rows N] [--compact-zero-columns auto|true|false] \\
      [--index-type int32|int64] [--time-limit 3600] \\
      [--rel-tol 1e-6] [--abs-tol 1e-6] [--verbose 2]

With --dataset all, the default reproducibility set is:
  news20, E2006-log1p, rcv1-train

The penalty grid is fixed to:
  alpha = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}
  lambda = alpha * ||A'b||_inf

The default `bulk` path constructs finalized CSC conic data directly and
reuses it for every alpha. Use `--modeling jump` only to compare against the
scalar JuMP formulation. No CBF file is created.
""")
end

required(options, name) = haskey(options, name) ? options[name] : error("Missing --$name.")

function parse_bool(value::AbstractString)
    lowercase(value) == "true" && return true
    lowercase(value) == "false" && return false
    error("Expected true or false, got '$value'.")
end

function resolve_compact_zero_columns(value::AbstractString)
    lowercase(value) == "auto" && return true
    return parse_bool(value)
end

function parse_index_type(value::AbstractString)
    lowercase(value) == "int32" && return Int32
    lowercase(value) == "int64" && return Int64
    error("--index-type must be int32 or int64.")
end

function atomic_toml_write(path::AbstractString, value::AbstractDict)
    directory = dirname(abspath(path))
    mkpath(directory)
    temporary, io = mktemp(directory)
    try
        TOML.print(io, value; sorted = true)
        close(io)
        mv(temporary, path; force = true)
    catch
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
        rethrow()
    end
    return path
end

function load_pdcs_optimizer(mode::String, pdcs_root::AbstractString)
    module_name, relative_path = if mode == "pdcs-cpu"
        (:PDCS_CPU, joinpath("src", "pdcs_cpu", "PDCS_CPU.jl"))
    elseif mode == "pdcs-gpu"
        (:PDCS_GPU, joinpath("src", "pdcs_gpu", "PDCS_GPU.jl"))
    else
        error("No optimizer exists for mode '$mode'.")
    end
    source = joinpath(abspath(pdcs_root), relative_path)
    isfile(source) || error("Cannot find PDCS optimizer source: $source")
    if !isdefined(Main, module_name)
        Base.include(Main, source)
    end
    return Base.invokelatest() do
        getfield(getfield(Main, module_name), :Optimizer)
    end
end

function build_experiment_representation(
    data::LassoData;
    modeling::String,
    penalty_ratio::Real,
    workers::Integer,
)
    modeling == "bulk" && return build_lasso_conic_data(
        data;
        penalty_ratio,
        workers,
    )
    modeling == "jump" && return build_lasso_socp(data; penalty_ratio)
    throw(ArgumentError("modeling must be bulk or jump, got '$modeling'"))
end

function materialize_bulk_model(
    representation::LassoConicData,
    optimizer_factory,
)
    pdcs_module = parentmodule(optimizer_factory)
    isdefined(pdcs_module, :model_from_conic_data) || error(
        "--modeling bulk requires $(nameof(pdcs_module)).model_from_conic_data; " *
        "use the PDCS bulk-cache branch or pass --modeling jump",
    )
    model_factory = getfield(pdcs_module, :model_from_conic_data)
    optimizer = Base.invokelatest(optimizer_factory)
    return Base.invokelatest(model_factory, representation; optimizer)
end

function set_pdcs_options!(
    model::JuMP.Model,
    output_dir::AbstractString,
    dataset_id::String,
    alpha::Float64;
    time_limit::Float64,
    rel_tol::Float64,
    abs_tol::Float64,
    verbose::Int,
)
    alpha_tag = replace(string(alpha), "." => "p")
    log_dir = joinpath(output_dir, "logs")
    mkpath(log_dir)
    set_time_limit_sec(model, time_limit)
    set_optimizer_attribute(model, "use_scaling", true)
    set_optimizer_attribute(model, "use_adaptive_restart", true)
    set_optimizer_attribute(model, "use_adaptive_step_size_weight", true)
    set_optimizer_attribute(model, "use_resolving", true)
    set_optimizer_attribute(model, "use_accelerated", false)
    set_optimizer_attribute(model, "use_aggressive", true)
    set_optimizer_attribute(model, "rel_tol", rel_tol)
    set_optimizer_attribute(model, "abs_tol", abs_tol)
    set_optimizer_attribute(model, "verbose", verbose)
    set_optimizer_attribute(
        model,
        "logfile",
        joinpath(log_dir, "$(dataset_id)_alpha_$(alpha_tag).log"),
    )
    return
end

function add_if_available!(f::Function, entry::Dict{String, Any}, key::String)
    try
        value = f()
        value === nothing || (entry[key] = value)
    catch
        # A solver is not required to implement every optional MOI result.
    end
    return entry
end

function base_result(
    data::LassoData,
    mode::String,
    modeling::String,
    compact_zero_columns::Bool,
    load_seconds::Float64,
    build_seconds::Float64,
    representation,
    alphas,
)
    result = Dict{String, Any}(
        "schema_version" => 1,
        "dataset" => data.dataset_id,
        "mode" => mode,
        "modeling" => modeling,
        "rows" => size(data.A, 1),
        "columns" => size(data.A, 2),
        "original_columns" => data.original_features,
        "nnz" => nnz(data.A),
        "compact_zero_columns" => compact_zero_columns,
        "lambda_reference" => data.lambda_reference,
        "lambda_zero_threshold" => 2.0 * data.lambda_reference,
        "alphas" => copy(alphas),
        "load_seconds" => load_seconds,
        "build_seconds" => build_seconds,
        "objective" => "norm(A*x-b, 2)^2 + lambda*norm(x, 1)",
        "runs" => Any[],
    )
    if representation isa LassoConicData
        result["formulation"] = string(representation.formulation)
        result["canonical_rows"] = representation.num_rows
        result["canonical_variables"] = representation.num_variables
        result["canonical_nnz"] = length(representation.nzval)
        result["model_assembly_seconds"] = representation.timings.assembly_seconds
    end
    data.row_limit === nothing || (result["max_rows"] = data.row_limit)
    return result
end

function run_dataset(
    dataset_id::String;
    mode::String,
    modeling::String,
    optimizer_factory,
    raw_dir::String,
    output_dir::String,
    max_rows::Union{Nothing, Int64},
    compact_zero_columns::Bool,
    index_type::Type{<:Integer},
    time_limit::Float64,
    rel_tol::Float64,
    abs_tol::Float64,
    verbose::Int,
    workers::Int,
    alphas::Vector{Float64} = ALPHAS,
)
    spec = dataset_spec(dataset_id; raw_dir)
    println("load $dataset_id from $(spec.path)")
    data = nothing
    load_seconds = @elapsed data = load_libsvm(
        spec;
        max_rows,
        label_mode = spec.label_mode,
        value_type = Float32,
        index_type,
        compact_zero_columns,
    )
    println(
        "loaded $dataset_id: m=$(size(data.A, 1)) n=$(size(data.A, 2)) " *
        "nnz=$(nnz(data.A)) lambda_ref=$(data.lambda_reference) " *
        "time=$(round(load_seconds; digits=3)) s",
    )

    representation = nothing
    build_seconds = @elapsed representation = build_experiment_representation(
        data;
        modeling,
        penalty_ratio = first(alphas),
        workers,
    )
    result = base_result(
        data,
        mode,
        modeling,
        compact_zero_columns,
        load_seconds,
        build_seconds,
        representation,
        alphas,
    )
    alpha_tag = length(alphas) == 1 ? replace(string(first(alphas)), "." => "p") : "sweep"
    output_path = joinpath(output_dir, "$(dataset_id)_penalty_$(alpha_tag).toml")
    model = representation isa LassoSOCPModel ? representation.model : nothing
    if mode != "build" && representation isa LassoConicData
        cache_build_seconds = @elapsed model = materialize_bulk_model(
            representation,
            optimizer_factory,
        )
        result["cache_build_seconds"] = cache_build_seconds
    end
    optimizer_attached = false

    for (alpha_index, alpha) in enumerate(alphas)
        if optimizer_attached && representation isa LassoSOCPModel
            # Keep the JuMP constraint cache but discard all solver state so
            # every alpha is an independent run without a warm start.
            JuMP.MOI.Utilities.drop_optimizer(model)
            optimizer_attached = false
        end
        lambda = alpha * data.lambda_reference
        penalty_update_seconds = 0.0
        if alpha_index > 1
            penalty_update_seconds = @elapsed if representation isa LassoConicData
                model === nothing ?
                    set_penalty!(representation, lambda) :
                    set_penalty!(model, representation, lambda)
            else
                set_penalty!(representation, lambda)
            end
        end
        entry = Dict{String, Any}(
            "alpha" => alpha,
            "lambda" => lambda,
            "penalty_update_seconds" => penalty_update_seconds,
            "zero_solution_theory" => alpha >= 2.0,
        )

        if mode == "build"
            entry["status"] = "MODEL_BUILT"
            println("$dataset_id alpha=$alpha lambda=$lambda $modeling model built")
        else
            if representation isa LassoSOCPModel
                set_optimizer(model, optimizer_factory)
                optimizer_attached = true
            end
            set_pdcs_options!(
                model,
                output_dir,
                dataset_id,
                alpha;
                time_limit,
                rel_tol,
                abs_tol,
                verbose,
            )
            println("solve $dataset_id alpha=$alpha lambda=$lambda")
            try
                wall_seconds = @elapsed optimize!(model)
                entry["wall_seconds"] = wall_seconds
                entry["termination_status"] = string(termination_status(model))
                entry["primal_status"] = string(primal_status(model))
                entry["dual_status"] = string(dual_status(model))
                add_if_available!(entry, "solver_seconds") do
                    solve_time(model)
                end
                if haskey(entry, "solver_seconds")
                    entry["handoff_seconds"] = max(
                        0.0,
                        wall_seconds - entry["solver_seconds"],
                    )
                end
                add_if_available!(entry, "objective_value") do
                    objective_value(model)
                end
                add_if_available!(entry, "dual_objective_value") do
                    dual_objective_value(model)
                end
                println(
                    "finished $dataset_id alpha=$alpha status=$(entry["termination_status"]) " *
                    "wall=$(round(wall_seconds; digits=3)) s",
                )
            catch error
                entry["termination_status"] = "SCRIPT_ERROR"
                entry["error"] = sprint(showerror, error)
                push!(result["runs"], entry)
                atomic_toml_write(output_path, result)
                rethrow()
            end
        end

        push!(result["runs"], entry)
        atomic_toml_write(output_path, result)
    end
    println("wrote $output_path")
    return output_path
end

function main(args = ARGS)
    options = parse_options(collect(args))
    options === nothing && return print_help()
    dataset_option = required(options, "dataset")
    available_datasets = dataset_specs()
    dataset_option == "all" || haskey(available_datasets, dataset_option) ||
        error("Unknown dataset '$dataset_option'; see datasets.toml.")
    mode = get(options, "mode", "build")
    mode in ("build", "pdcs-cpu", "pdcs-gpu") || error(
        "--mode must be build, pdcs-cpu, or pdcs-gpu."
    )
    modeling = get(options, "modeling", "bulk")
    modeling in ("bulk", "jump") || error("--modeling must be bulk or jump.")
    workers = parse(Int, get(options, "workers", string(Threads.nthreads())))
    workers > 0 || error("--workers must be positive.")
    max_rows = haskey(options, "max-rows") ? parse(Int64, options["max-rows"]) : nothing
    max_rows === nothing || max_rows > 0 || error("--max-rows must be positive.")
    compact_option = get(options, "compact-zero-columns", "auto")
    compact_zero_columns = resolve_compact_zero_columns(compact_option)
    raw_dir = abspath(get(options, "raw-dir", joinpath(@__DIR__, "raw")))
    output_dir = abspath(get(options, "output-dir", joinpath(@__DIR__, "results")))
    index_type = parse_index_type(get(options, "index-type", "int32"))
    time_limit = parse(Float64, get(options, "time-limit", "3600"))
    rel_tol = parse(Float64, get(options, "rel-tol", "1e-6"))
    abs_tol = parse(Float64, get(options, "abs-tol", "1e-6"))
    verbose = parse(Int, get(options, "verbose", "2"))
    alphas = haskey(options, "alpha") ?
        Float64[parse(Float64, options["alpha"])] : ALPHAS
    time_limit > 0 || error("--time-limit must be positive.")
    rel_tol > 0 || error("--rel-tol must be positive.")
    abs_tol > 0 || error("--abs-tol must be positive.")

    optimizer_factory = mode == "build" ? nothing : load_pdcs_optimizer(
        mode,
        get(options, "pdcs-root", DEFAULT_PDCS_ROOT),
    )
    dataset_ids = dataset_option == "all" ? DEFAULT_DATASET_IDS : (dataset_option,)
    mkpath(output_dir)
    for dataset_id in dataset_ids
        # The PDCS module and its MOI methods may have just been included at
        # runtime. Enter the latest Julia world for all downstream dispatch.
        Base.invokelatest(
            run_dataset,
            dataset_id;
            mode,
            modeling,
            optimizer_factory,
            raw_dir,
            output_dir,
            max_rows,
            compact_zero_columns,
            index_type,
            time_limit,
            rel_tol,
            abs_tol,
            verbose,
            workers,
            alphas,
        )
        GC.gc()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
