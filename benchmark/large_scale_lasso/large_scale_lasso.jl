using ArgParse
using JuMP
using LinearAlgebra
using Random
using SHA
using SparseArrays
using TOML

const GENERATOR_VERSION = 1
const MANIFEST_SCHEMA_VERSION = 1
const DEFAULT_REPLICATES = 5
const TABLE5_DIMENSIONS = [
    (10_000, 100_000, 1e-4),
    (70_000, 700_000, 1e-4),
    (400_000, 7_000_000, 1e-4),
    (700_000, 7_000_000, 1e-4),
    (750_000, 7_500_000, 1e-4),
]

function file_sha256(path::AbstractString)
    isfile(path) || return ""
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function active_environment()
    project_path = Base.active_project()
    project_path === nothing && error("No active Julia project. Launch with --project=/path/to/test_cpu_env.")
    manifest_path = joinpath(dirname(project_path), "Manifest.toml")
    return Dict(
        "julia_version" => string(VERSION),
        "project_sha256" => file_sha256(project_path),
        "manifest_sha256" => file_sha256(manifest_path),
    )
end

function preset_dimensions(preset::AbstractString)
    preset == "pilot" && return [(100, 1_000, 1e-2)]
    preset == "table5" && return TABLE5_DIMENSIONS
    error("Unsupported preset '$preset'; expected 'pilot' or 'table5'.")
end

function make_manifest(
    preset::AbstractString,
    master_seed::UInt64;
    replicates::Int = DEFAULT_REPLICATES,
)
    replicates > 0 || error("replicates must be positive")
    master_seed <= UInt64(typemax(Int64)) || error("master seed must fit in Int64")
    rng = MersenneTwister(Int64(master_seed))
    used_seeds = Set{Int64}()
    instances = Vector{Dict{String, Any}}()
    for (m, n, density) in preset_dimensions(preset)
        for replicate in 1:replicates
            seed = rand(rng, Int64(1):typemax(Int64))
            while seed in used_seeds
                seed = rand(rng, Int64(1):typemax(Int64))
            end
            push!(used_seeds, seed)
            id = preset == "pilot" ?
                "pilot-r$(lpad(replicate, 2, '0'))" :
                "table5-m$(m)-n$(n)-r$(lpad(replicate, 2, '0'))"
            push!(instances, Dict(
                "id" => id,
                "preset" => String(preset),
                "m" => m,
                "n" => n,
                "density" => density,
                "replicate" => replicate,
                # TOML integers are signed 64-bit. A string also avoids
                # accidental Float64 conversion in non-Julia readers.
                "seed" => string(seed),
            ))
        end
    end
    env = active_environment()
    return Dict{String, Any}(
        "schema_version" => MANIFEST_SCHEMA_VERSION,
        "generator_version" => GENERATOR_VERSION,
        "preset" => String(preset),
        "master_seed" => string(master_seed),
        "replicates" => replicates,
        "julia_version" => env["julia_version"],
        "project_sha256" => env["project_sha256"],
        "manifest_sha256" => env["manifest_sha256"],
        "script_sha256" => file_sha256(@__FILE__),
        "instances" => instances,
    )
end

function atomic_toml_write(path::AbstractString, value::AbstractDict)
    mkpath(dirname(abspath(path)))
    temporary, io = mktemp(dirname(abspath(path)))
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

write_manifest(path::AbstractString, manifest::AbstractDict) =
    atomic_toml_write(path, manifest)

read_manifest(path::AbstractString) = TOML.parsefile(path)

function validate_manifest(manifest::AbstractDict; strict_environment::Bool = true)
    get(manifest, "schema_version", nothing) == MANIFEST_SCHEMA_VERSION ||
        error("Unsupported or missing manifest schema_version.")
    get(manifest, "generator_version", nothing) == GENERATOR_VERSION ||
        error("Manifest generator version does not match this script.")
    instances = get(manifest, "instances", nothing)
    instances isa AbstractVector || error("Manifest has no instances array.")
    ids = String[]
    seeds = String[]
    for entry in instances
        m = entry["m"]
        n = entry["n"]
        density = entry["density"]
        m > 0 || error("Instance $(entry["id"]) has nonpositive m.")
        n > 0 || error("Instance $(entry["id"]) has nonpositive n.")
        0.0 < density <= 1.0 || error("Instance $(entry["id"]) has invalid density.")
        parse(Int64, entry["seed"]) > 0 || error("Instance $(entry["id"]) has invalid seed.")
        push!(ids, entry["id"])
        push!(seeds, entry["seed"])
    end
    length(unique(ids)) == length(ids) || error("Manifest contains duplicate instance IDs.")
    length(unique(seeds)) == length(seeds) || error("Manifest contains duplicate seeds.")
    if strict_environment
        env = active_environment()
        for field in ("julia_version", "project_sha256", "manifest_sha256")
            expected = get(manifest, field, "")
            actual = env[field]
            expected == actual || error(
                "Environment mismatch for $field: expected '$expected', got '$actual'. " *
                "Use the exact Julia Project/Manifest, or pass --allow-environment-mismatch."
            )
        end
        expected_script = get(manifest, "script_sha256", "")
        actual_script = file_sha256(@__FILE__)
        expected_script == actual_script || error(
            "Generator script mismatch: expected SHA-256 '$expected_script', " *
            "got '$actual_script'. Copy the exact large_scale_lasso.jl used to create the manifest."
        )
    end
    return true
end

function generate_instance(entry::AbstractDict)
    m = Int(entry["m"])
    n = Int(entry["n"])
    density = Float64(entry["density"])
    rng = MersenneTwister(parse(Int64, entry["seed"]))

    A = sprand(rng, m, n, density)
    x_feat = randn(rng, n)
    x_feat ./= sqrt(n)
    zero_indices = randperm(rng, n)[1:div(n, 2)]
    x_feat[zero_indices] .= 0.0
    b = A * x_feat
    b .+= 1e-6
    lambda = norm(transpose(A) * b, Inf)

    return (; A, x_feat, b, lambda)
end

function update_tag!(ctx, tag::AbstractString)
    SHA.update!(ctx, Vector{UInt8}(codeunits(tag)))
    SHA.update!(ctx, UInt8[0x00])
end

function update_array!(ctx, tag::AbstractString, values::AbstractVector{T}) where {T}
    update_tag!(ctx, tag)
    SHA.update!(ctx, reinterpret(UInt8, [Int64(length(values))]))
    SHA.update!(ctx, reinterpret(UInt8, values))
    return ctx
end

function numerical_digest(data)
    ctx = SHA.SHA256_CTX()
    update_array!(ctx, "A.colptr", data.A.colptr)
    update_array!(ctx, "A.rowval", data.A.rowval)
    update_array!(ctx, "A.nzval", data.A.nzval)
    update_array!(ctx, "x_feat", data.x_feat)
    update_array!(ctx, "b", data.b)
    update_array!(ctx, "lambda", [data.lambda])
    return bytes2hex(SHA.digest!(ctx))
end

function model_arrays(data)
    m, n = size(data.A)
    c_len = 2 + m + 2n
    c = zeros(c_len)
    c[2] = 2.0
    c[(3 + m):end] .= data.lambda

    constraint_matrix = spzeros(1 + m, c_len)
    constraint_matrix[1, 1] = 1.0
    constraint_matrix[2:end, 3:(2 + m)] = I(m)
    constraint_matrix[2:end, (3 + m):(2 + m + n)] = data.A
    constraint_matrix[2:end, (3 + m + n):end] = -data.A
    rhs = zeros(1 + m)
    rhs[1] = 1.0
    rhs[2:end] .= data.b
    return (; c, constraint_matrix, rhs)
end

function model_digest(arrays)
    ctx = SHA.SHA256_CTX()
    update_array!(ctx, "c", arrays.c)
    update_array!(ctx, "G.colptr", arrays.constraint_matrix.colptr)
    update_array!(ctx, "G.rowval", arrays.constraint_matrix.rowval)
    update_array!(ctx, "G.nzval", arrays.constraint_matrix.nzval)
    update_array!(ctx, "rhs", arrays.rhs)
    return bytes2hex(SHA.digest!(ctx))
end

function build_model(data, arrays = model_arrays(data))
    m, n = size(data.A)
    model = Model()
    @variable(model, x[1:length(arrays.c)])
    @objective(model, Min, arrays.c' * x)
    @constraint(model, x[(3 + m):end] .>= 0.0)
    @constraint(model, arrays.constraint_matrix * x .== arrays.rhs)
    @variable(model, t)
    @variable(model, u)
    @constraint(model, t == (x[1] + x[2]) / sqrt(2))
    @constraint(model, u == (x[1] - x[2]) / sqrt(2))
    @constraint(model, [t; u; x[3:(2 + m)]] in SecondOrderCone())
    return model
end

function output_filename(entry::AbstractDict)
    return "lasso_$(entry["id"])_seed$(entry["seed"]).cbf.gz"
end

function summary_for(entry::AbstractDict, data, arrays, output_path::AbstractString, elapsed::Real)
    return Dict{String, Any}(
        "id" => entry["id"],
        "m" => entry["m"],
        "n" => entry["n"],
        "density" => entry["density"],
        "replicate" => entry["replicate"],
        "seed" => entry["seed"],
        "nnz" => nnz(data.A),
        "lambda_bits" => string(reinterpret(UInt64, data.lambda)),
        "lambda" => data.lambda,
        "numerical_digest" => numerical_digest(data),
        "model_digest" => model_digest(arrays),
        "output_file" => basename(output_path),
        "elapsed_seconds" => Float64(elapsed),
    )
end

function read_results(path::AbstractString)
    isfile(path) || return Dict{String, Any}("schema_version" => 1, "results" => Any[])
    return TOML.parsefile(path)
end

function select_entries(manifest::AbstractDict, instance_id)
    entries = manifest["instances"]
    instance_id === nothing && return entries
    selected = [entry for entry in entries if entry["id"] == instance_id]
    length(selected) == 1 || error("Instance ID '$instance_id' was not found.")
    return selected
end

function build_instances(
    manifest::AbstractDict;
    output_dir::AbstractString,
    results_path::AbstractString,
    instance_id = nothing,
    overwrite::Bool = false,
    strict_environment::Bool = true,
)
    validate_manifest(manifest; strict_environment)
    mkpath(output_dir)
    old_results = read_results(results_path)
    by_id = Dict(String(result["id"]) => result for result in old_results["results"])

    for entry in select_entries(manifest, instance_id)
        output_path = joinpath(output_dir, output_filename(entry))
        if isfile(output_path) && !overwrite
            haskey(by_id, entry["id"]) || error(
                "Output exists but no result summary is available for $(entry["id"]); use --overwrite."
            )
            println("skip $(entry["id"]): $output_path")
            continue
        end
        started = time()
        println("build $(entry["id"]): m=$(entry["m"]) n=$(entry["n"]) " *
                "density=$(entry["density"]) seed=$(entry["seed"])")
        data = generate_instance(entry)
        arrays = model_arrays(data)
        model = build_model(data, arrays)
        write_to_file(model, output_path)
        summary = summary_for(entry, data, arrays, output_path, time() - started)
        by_id[entry["id"]] = summary
        println("wrote $output_path nnz=$(summary["nnz"]) lambda=$(summary["lambda"])")
        data = arrays = model = nothing
        GC.gc()
        atomic_toml_write(results_path, Dict(
            "schema_version" => 1,
            "preset" => manifest["preset"],
            "results" => sort!(collect(values(by_id)); by = result -> result["id"]),
        ))
    end
    return read_results(results_path)
end

function comparable_result(result::AbstractDict)
    return Dict(
        field => result[field]
        for field in (
            "id", "m", "n", "density", "replicate", "seed", "nnz",
            "lambda_bits", "numerical_digest", "model_digest",
        )
    )
end

function verify_instances(
    manifest::AbstractDict,
    reference::AbstractDict;
    output_dir::AbstractString,
    strict_environment::Bool = true,
)
    temporary_results = joinpath(output_dir, "verification_results.toml")
    actual = build_instances(
        manifest;
        output_dir,
        results_path = temporary_results,
        overwrite = true,
        strict_environment,
    )
    expected_by_id = Dict(result["id"] => result for result in reference["results"])
    for result in actual["results"]
        id = result["id"]
        haskey(expected_by_id, id) || error("Reference has no result for $id.")
        expected = comparable_result(expected_by_id[id])
        observed = comparable_result(result)
        for field in keys(expected)
            expected[field] == observed[field] || error(
                "Verification failed for $id field '$field': " *
                "expected $(expected[field]), got $(observed[field])."
            )
        end
        println("verified $id")
    end
    length(actual["results"]) == length(reference["results"]) ||
        error("Reference and regenerated result counts differ.")
    println("verified $(length(actual["results"])) instances")
    return true
end

function parse_cli(args::Vector{String})
    isempty(args) && return ("help", Dict{String, Any}())
    command = args[1]
    command in ("generate-config", "build-data", "verify", "help", "--help", "-h") ||
        error("Unknown command '$command'. Run with --help.")
    options = Dict{String, Any}()
    boolean_flags = Set(["--overwrite", "--allow-environment-mismatch"])
    i = 2
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("Unexpected argument '$arg'.")
        if arg in boolean_flags
            options[arg[3:end]] = true
            i += 1
        else
            i == length(args) && error("Option '$arg' requires a value.")
            options[arg[3:end]] = args[i + 1]
            i += 2
        end
    end
    return (command, options)
end

required_option(options, name) =
    haskey(options, name) ? options[name] : error("Missing required option --$name.")

function print_help()
    println("""
Usage:
  julia --project=/path/to/test_cpu_env large_scale_lasso.jl generate-config \\
    --preset pilot|table5 --master-seed INTEGER --config PATH

  julia --project=/path/to/test_cpu_env large_scale_lasso.jl build-data \\
    --config PATH --output-dir DIR --results PATH [--instance ID] [--overwrite] \\
    [--allow-environment-mismatch]

  julia --project=/path/to/test_cpu_env large_scale_lasso.jl verify \\
    --config PATH --reference PATH --output-dir DIR \\
    [--allow-environment-mismatch]
""")
end

function main(args = ARGS)
    command, options = parse_cli(collect(args))
    if command in ("help", "--help", "-h")
        print_help()
        return 0
    elseif command == "generate-config"
        preset = required_option(options, "preset")
        master_seed = parse(UInt64, required_option(options, "master-seed"))
        config = required_option(options, "config")
        manifest = make_manifest(preset, master_seed)
        write_manifest(config, manifest)
        println("wrote $config with $(length(manifest["instances"])) instances")
    elseif command == "build-data"
        manifest = read_manifest(required_option(options, "config"))
        build_instances(
            manifest;
            output_dir = required_option(options, "output-dir"),
            results_path = required_option(options, "results"),
            instance_id = get(options, "instance", nothing),
            overwrite = get(options, "overwrite", false),
            strict_environment = !get(options, "allow-environment-mismatch", false),
        )
    elseif command == "verify"
        manifest = read_manifest(required_option(options, "config"))
        reference = read_results(required_option(options, "reference"))
        verify_instances(
            manifest,
            reference;
            output_dir = required_option(options, "output-dir"),
            strict_environment = !get(options, "allow-environment-mismatch", false),
        )
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch exception
        showerror(stderr, exception)
        println(stderr)
        exit(1)
    end
end
