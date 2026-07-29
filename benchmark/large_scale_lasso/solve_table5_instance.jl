#!/usr/bin/env julia

using Dates
using JuMP
using LinearAlgebra
using Random
using SHA
using SparseArrays
using TOML

import MathOptInterface as MOI

const CHUNK_LENGTH = 1_000_000

function parse_cli(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        option = arguments[index]
        startswith(option, "--") || error("unexpected positional argument: $option")
        index == length(arguments) && error("missing value for $option")
        haskey(options, option) && error("duplicate option: $option")
        options[option] = arguments[index + 1]
        index += 2
    end
    for option in (
        "--solver",
        "--manifest",
        "--instance-id",
        "--generator-script",
        "--generator-project",
        "--generator-manifest",
        "--result",
    )
        haskey(options, option) || error("missing required option $option")
    end
    solver = Symbol(options["--solver"])
    solver in (:cupdcs, :scs_gpu, :cuclarabel) ||
        error("--solver must be cupdcs, scs_gpu, or cuclarabel")
    tolerance = parse(Float64, get(options, "--tolerance", "1e-6"))
    time_limit = parse(Float64, get(options, "--time-limit", "3600"))
    print_frequency = parse(Int, get(options, "--print-frequency", "1000"))
    verbose_level = parse(Int, get(options, "--verbose-level", "2"))
    pdcs_profile = Symbol(get(options, "--pdcs-profile", "full"))
    tolerance > 0 || error("--tolerance must be positive")
    time_limit > 0 || error("--time-limit must be positive")
    print_frequency > 0 || error("--print-frequency must be positive")
    verbose_level in 0:2 || error("--verbose-level must be 0, 1, or 2")
    pdcs_profile in (:full, :baseline) ||
        error("--pdcs-profile must be full or baseline")
    paths = Dict(
        name => abspath(options[name])
        for name in (
            "--manifest",
            "--generator-script",
            "--generator-project",
            "--generator-manifest",
        )
    )
    for (name, path) in paths
        isfile(path) || error("$name does not exist: $path")
    end
    return (
        solver = solver,
        manifest = paths["--manifest"],
        instance_id = options["--instance-id"],
        generator_script = paths["--generator-script"],
        generator_project = paths["--generator-project"],
        generator_manifest = paths["--generator-manifest"],
        result = abspath(options["--result"]),
        tolerance = tolerance,
        time_limit = time_limit,
        print_frequency = print_frequency,
        verbose_level = verbose_level,
        pdcs_profile = pdcs_profile,
        required_julia_version = get(
            options,
            "--required-julia-version",
            "1.10.4",
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

function file_sha256(path)
    return open(path, "r") do stream
        bytes2hex(SHA.sha256(stream))
    end
end

function load_and_validate_entry(options)
    string(VERSION) == options.required_julia_version || error(
        "required Julia $(options.required_julia_version), got $VERSION",
    )
    manifest = TOML.parsefile(options.manifest)
    get(manifest, "schema_version", nothing) == 1 ||
        error("manifest schema_version must be 1")
    get(manifest, "generator_version", nothing) == 1 ||
        error("manifest generator_version must be 1")
    get(manifest, "preset", nothing) in ("table5", "pilot") ||
        error("manifest preset must be table5 or pilot")
    get(manifest, "master_seed", nothing) == "20260728" ||
        error("manifest master_seed must be 20260728")
    get(manifest, "replicates", nothing) == 5 ||
        error("manifest replicates must be 5")
    environment_hashes = (
        "script_sha256" => file_sha256(options.generator_script),
        "project_sha256" => file_sha256(options.generator_project),
        "manifest_sha256" => file_sha256(options.generator_manifest),
    )
    for (field, observed) in environment_hashes
        expected = get(manifest, field, "")
        expected == observed || error(
            "strict reproduction mismatch for $field: " *
            "expected '$expected', got '$observed'",
        )
    end
    get(manifest, "julia_version", "") == string(VERSION) ||
        error("manifest Julia version does not match the running Julia")
    selected = [
        entry for entry in manifest["instances"]
        if entry["id"] == options.instance_id
    ]
    length(selected) == 1 ||
        error("instance ID $(options.instance_id) was not found exactly once")
    return selected[1]
end

function update_tag!(context, tag)
    SHA.update!(context, Vector{UInt8}(codeunits(tag)))
    SHA.update!(context, UInt8[0x00])
    return context
end

function update_array_header!(context, tag, length_value)
    update_tag!(context, tag)
    SHA.update!(
        context,
        reinterpret(UInt8, [Int64(length_value)]),
    )
    return context
end

function update_raw!(context, values::AbstractVector)
    isempty(values) || SHA.update!(context, reinterpret(UInt8, values))
    return context
end

function update_array!(context, tag, values::AbstractVector)
    update_array_header!(context, tag, length(values))
    update_raw!(context, values)
    return context
end

function update_filled!(
    context,
    value::T,
    count::Integer,
) where {T}
    remaining = count
    while remaining > 0
        length_value = min(remaining, CHUNK_LENGTH)
        update_raw!(context, fill(value, length_value))
        remaining -= length_value
    end
    return context
end

function update_shifted_indices!(context, indices, shift)
    first_index = firstindex(indices)
    final_index = lastindex(indices)
    start = first_index
    while start <= final_index
        stop = min(start + CHUNK_LENGTH - 1, final_index)
        update_raw!(context, Int64.(view(indices, start:stop)) .+ shift)
        start = stop + 1
    end
    return context
end

function update_negated_values!(context, values)
    start = firstindex(values)
    final_index = lastindex(values)
    while start <= final_index
        stop = min(start + CHUNK_LENGTH - 1, final_index)
        update_raw!(context, .-view(values, start:stop))
        start = stop + 1
    end
    return context
end

function numerical_digest(A, x_feature, b, lambda)
    context = SHA.SHA256_CTX()
    update_array!(context, "A.colptr", A.colptr)
    update_array!(context, "A.rowval", A.rowval)
    update_array!(context, "A.nzval", A.nzval)
    update_array!(context, "x_feat", x_feature)
    update_array!(context, "b", b)
    update_array!(context, "lambda", [lambda])
    return bytes2hex(SHA.digest!(context))
end

function model_colptr(A, m, n)
    c_length = 2 + m + 2n
    colptr = Vector{Int64}(undef, c_length + 1)
    colptr[1] = 1
    colptr[2] = 2
    colptr[3] = 2
    for column in 1:m
        colptr[3 + column] = colptr[2 + column] + 1
    end
    offset = 3 + m
    for column in 1:n
        column_nnz = A.colptr[column + 1] - A.colptr[column]
        colptr[offset + column] = colptr[offset + column - 1] + column_nnz
    end
    offset += n
    for column in 1:n
        column_nnz = A.colptr[column + 1] - A.colptr[column]
        colptr[offset + column] = colptr[offset + column - 1] + column_nnz
    end
    return colptr
end

"""
Compute the exact digest produced by `large_scale_lasso.jl:model_digest`
without materializing its very large duplicated sparse constraint matrix.
"""
function model_digest(A, b, lambda)
    m, n = size(A)
    c_length = 2 + m + 2n
    matrix_nnz = 1 + m + 2nnz(A)
    context = SHA.SHA256_CTX()

    update_array_header!(context, "c", c_length)
    update_raw!(context, [0.0, 2.0])
    update_filled!(context, 0.0, m)
    update_filled!(context, lambda, 2n)

    colptr = model_colptr(A, m, n)
    update_array!(context, "G.colptr", colptr)
    colptr = nothing

    update_array_header!(context, "G.rowval", matrix_nnz)
    update_raw!(context, Int64[1])
    start = 2
    while start <= m + 1
        stop = min(start + CHUNK_LENGTH - 1, m + 1)
        update_raw!(context, collect(Int64(start):Int64(stop)))
        start = stop + 1
    end
    update_shifted_indices!(context, A.rowval, 1)
    update_shifted_indices!(context, A.rowval, 1)

    update_array_header!(context, "G.nzval", matrix_nnz)
    update_raw!(context, [1.0])
    update_filled!(context, 1.0, m)
    update_raw!(context, A.nzval)
    update_negated_values!(context, A.nzval)

    update_array_header!(context, "rhs", 1 + m)
    update_raw!(context, [1.0])
    update_raw!(context, b)
    return bytes2hex(SHA.digest!(context))
end

function generate_instance(entry)
    m = Int(entry["m"])
    n = Int(entry["n"])
    density = Float64(entry["density"])
    rng = MersenneTwister(parse(Int64, entry["seed"]))
    A = sprand(rng, m, n, density)
    x_feature = randn(rng, n)
    x_feature ./= sqrt(n)
    zero_indices = randperm(rng, n)[1:div(n, 2)]
    x_feature[zero_indices] .= 0.0
    b = A * x_feature
    b .+= 1e-6
    lambda = norm(transpose(A) * b, Inf)
    digest = numerical_digest(A, x_feature, b, lambda)
    formulation_digest = model_digest(A, b, lambda)
    summary = Dict{String,Any}(
        "m" => m,
        "n" => n,
        "density" => density,
        "replicate" => Int(entry["replicate"]),
        "seed" => entry["seed"],
        "nnz" => nnz(A),
        "lambda" => lambda,
        "lambda_bits" => string(reinterpret(UInt64, lambda)),
        "numerical_digest" => digest,
        "model_digest" => formulation_digest,
    )
    x_feature = zero_indices = nothing
    GC.gc()
    return A, b, lambda, summary
end

function scs_gpu_optimizer()
    SCS.is_available(SCS.GpuIndirectSolver) ||
        error("SCS.GpuIndirectSolver is unavailable in this environment")
    raw = SCS.Optimizer()
    # SCS GPU uses Int32 indices, so choose the linear solver before creating
    # MOI's cache.
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

function optimizer_factory(solver)
    if solver == :cupdcs
        CUDA.functional() || error("CUDA is not functional for cuPDCS")
        return PDCS.PDCS_GPU.Optimizer
    elseif solver == :scs_gpu
        return scs_gpu_optimizer
    else
        CUDA.functional() || error("CUDA is not functional for cuClarabel")
        return Clarabel.Optimizer
    end
end

function configure!(model, options)
    if options.solver == :cupdcs
        full = options.pdcs_profile == :full
        attributes = (
            "verbose" => options.verbose_level,
            "time_limit_secs" => options.time_limit,
            "abs_tol" => options.tolerance,
            "rel_tol" => options.tolerance,
            "check_terminate_freq" => options.print_frequency,
            "print_freq" => options.print_frequency,
            "use_scaling" => true,
            "use_adaptive_restart" => full,
            "use_restart" => full,
            "use_adaptive_step" => true,
            "use_adaptive_step_size_weight" => true,
            "use_aggressive" => full,
            "use_reflection" => full,
            "use_halpern" => full,
            "use_resolving" => full,
            "use_kkt_restart" => false,
            "use_duality_gap_restart" => full,
            "logfile" => nothing,
        )
    elseif options.solver == :scs_gpu
        attributes = (
            "linear_solver" => SCS.GpuIndirectSolver,
            "eps_abs" => options.tolerance,
            "eps_rel" => options.tolerance,
            "time_limit_secs" => options.time_limit,
            "verbose" => options.verbose_level,
        )
    else
        attributes = (
            "direct_solve_method" => :cudss,
            "tol_gap_abs" => options.tolerance,
            "tol_gap_rel" => options.tolerance,
            "tol_feas" => options.tolerance,
            "time_limit" => options.time_limit,
            "verbose" => options.verbose_level > 0,
        )
    end
    for (name, value) in attributes
        set_optimizer_attribute(model, name, value)
    end
    return model
end

function build_model(A, b, lambda, options)
    m, n = size(A)
    optimizer = optimizer_factory(options.solver)
    model = if options.solver == :scs_gpu
        Model(optimizer; add_bridges = false)
    else
        Model(optimizer)
    end
    configure!(model, options)

    # Preserve the variable and constraint ordering of large_scale_lasso.jl.
    @variable(model, x_one)
    @variable(model, residual_epigraph)
    @variable(model, residual[1:m])
    @variable(model, x_positive[1:n])
    @variable(model, x_negative[1:n])
    @objective(
        model,
        Min,
        2residual_epigraph +
        lambda * sum(x_positive[index] + x_negative[index] for index in 1:n),
    )
    @constraint(model, [x_positive; x_negative] .>= 0.0)
    @constraint(
        model,
        [x_one; residual + A * x_positive - A * x_negative] .== [1.0; b],
    )
    @variable(model, t)
    @variable(model, u)
    @constraint(model, t == (x_one + residual_epigraph) / sqrt(2))
    @constraint(model, u == (x_one - residual_epigraph) / sqrt(2))
    @constraint(model, [t; u; residual] in SecondOrderCone())
    return model
end

function synchronize(solver)
    solver in (:cupdcs, :cuclarabel) && CUDA.synchronize()
    return nothing
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
        PDCS.PDCS_GPU.PDHGIterations()
    elseif solver == :scs_gpu
        SCS.ADMMIterations()
    else
        MOI.BarrierIterations()
    end
    return maybe_get(model, attribute, -1)
end

function cuda_metadata(solver)
    solver == :scs_gpu && return Dict{String,Any}(
        "gpu_backend" => "SCS.GpuIndirectSolver",
        "gpu_index_type" => string(SCS.scsint_t(SCS.GpuIndirectSolver)),
    )
    return Dict{String,Any}(
        "cuda_functional" => CUDA.functional(),
        "cuda_runtime" => try
            string(CUDA.runtime_version())
        catch
            "unknown"
        end,
        "cuda_driver" => try
            string(CUDA.driver_version())
        catch
            "unknown"
        end,
        "cuda_device" => try
            CUDA.name(CUDA.device())
        catch
            string(CUDA.device())
        end,
    )
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

function main()
    started = now(UTC)
    result = Dict{String,Any}(
        "schema_version" => 1,
        "instance_id" => OPTIONS.instance_id,
        "manifest" => OPTIONS.manifest,
        "solver" => string(OPTIONS.solver),
        "tolerance" => OPTIONS.tolerance,
        "time_limit_seconds" => OPTIONS.time_limit,
        "verbose_level" => OPTIONS.verbose_level,
        "pdcs_profile" => string(OPTIONS.pdcs_profile),
        "julia_version" => string(VERSION),
        "required_julia_version" => OPTIONS.required_julia_version,
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "started_utc" => string(started),
        "run_status" => "runtime_error",
        "termination_status" => "EXCEPTION",
        "primal_status" => "UNKNOWN_RESULT_STATUS",
        "dual_status" => "UNKNOWN_RESULT_STATUS",
        "objective_value" => NaN,
        "iterations" => -1,
        "generation_seconds" => NaN,
        "setup_seconds" => NaN,
        "optimize_wall_seconds" => NaN,
        "solver_solve_seconds" => NaN,
        "error" => "",
    )
    exit_code = 1
    try
        entry = load_and_validate_entry(OPTIONS)
        println(
            "TABLE5_SOLVE_START solver=$(OPTIONS.solver) " *
            "instance=$(OPTIONS.instance_id) m=$(entry["m"]) n=$(entry["n"]) " *
            "density=$(entry["density"]) seed=$(entry["seed"]) " *
            "tolerance=$(OPTIONS.tolerance) time_limit=$(OPTIONS.time_limit)",
        )
        merge!(result, cuda_metadata(OPTIONS.solver))

        generation_started = time()
        A, b, lambda, generation_summary = generate_instance(entry)
        result["generation_seconds"] = time() - generation_started
        merge!(result, generation_summary)
        println(
            "TABLE5_INSTANCE_READY instance=$(OPTIONS.instance_id) " *
            "nnz=$(result["nnz"]) lambda=$(result["lambda"]) " *
            "numerical_digest=$(result["numerical_digest"]) " *
            "model_digest=$(result["model_digest"]) " *
            "generation_seconds=$(result["generation_seconds"])",
        )

        setup_started = time()
        model = build_model(A, b, lambda, OPTIONS)
        result["setup_seconds"] = time() - setup_started
        A = b = nothing
        GC.gc()

        optimize_started = time()
        optimize!(model)
        synchronize(OPTIONS.solver)
        result["optimize_wall_seconds"] = time() - optimize_started
        result["termination_status"] = string(termination_status(model))
        result["primal_status"] = string(primal_status(model))
        result["dual_status"] = string(dual_status(model))
        result["objective_value"] = try
            objective_value(model)
        catch
            NaN
        end
        result["iterations"] = iteration_count(model, OPTIONS.solver)
        result["solver_solve_seconds"] = maybe_get(
            model,
            MOI.SolveTimeSec(),
            NaN,
        )
        result["run_status"] = "completed"
        exit_code = 0
        println(
            "TABLE5_SOLVE_RESULT solver=$(OPTIONS.solver) " *
            "instance=$(OPTIONS.instance_id) " *
            "termination=$(result["termination_status"]) " *
            "primal_status=$(result["primal_status"]) " *
            "iterations=$(result["iterations"]) " *
            "objective=$(result["objective_value"]) " *
            "optimize_wall_seconds=$(result["optimize_wall_seconds"])",
        )
    catch error_value
        result["error"] = sprint(
            showerror,
            error_value,
            catch_backtrace(),
        )
        println(stderr, "TABLE5_SOLVE_EXCEPTION")
        showerror(stderr, error_value, catch_backtrace())
        println(stderr)
    finally
        finished = now(UTC)
        result["finished_utc"] = string(finished)
        result["wall_seconds"] = Dates.value(finished - started) / 1_000.0
        atomic_toml(OPTIONS.result, result)
        println(
            "TABLE5_SOLVE_FINISH solver=$(OPTIONS.solver) " *
            "instance=$(OPTIONS.instance_id) status=$(result["run_status"]) " *
            "result=$(OPTIONS.result)",
        )
    end
    return exit_code
end

exit(main())
