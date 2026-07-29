#!/usr/bin/env julia

using Random
using SHA
using TOML

const TABLE5_DIMENSIONS = [
    (10_000, 100_000, 1e-4),
    (70_000, 700_000, 1e-4),
    (400_000, 7_000_000, 1e-4),
    (700_000, 7_000_000, 1e-4),
    (750_000, 7_500_000, 1e-4),
]
const PILOT_DIMENSIONS = [(100, 1_000, 1e-2)]

function parse_cli(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        option = arguments[index]
        startswith(option, "--") || error("unexpected argument: $option")
        index == length(arguments) && error("missing value for $option")
        haskey(options, option) && error("duplicate option: $option")
        options[option] = arguments[index + 1]
        index += 2
    end
    for option in (
        "--output",
        "--generator-script",
        "--generator-project",
        "--generator-manifest",
    )
        haskey(options, option) || error("missing required option $option")
    end
    master_seed = parse(UInt64, get(options, "--master-seed", "20260728"))
    master_seed <= UInt64(typemax(Int64)) ||
        error("master seed must fit in Int64")
    return (
        output = abspath(options["--output"]),
        generator_script = abspath(options["--generator-script"]),
        generator_project = abspath(options["--generator-project"]),
        generator_manifest = abspath(options["--generator-manifest"]),
        master_seed = master_seed,
        preset = get(options, "--preset", "table5"),
    )
end

function file_sha256(path)
    isfile(path) || error("required source file does not exist: $path")
    return open(path, "r") do stream
        bytes2hex(SHA.sha256(stream))
    end
end

function make_manifest(options)
    options.preset in ("table5", "pilot") ||
        error("--preset must be table5 or pilot")
    dimensions =
        options.preset == "table5" ? TABLE5_DIMENSIONS : PILOT_DIMENSIONS
    rng = MersenneTwister(Int64(options.master_seed))
    seeds = Set{Int64}()
    instances = Vector{Dict{String,Any}}()
    for (m, n, density) in dimensions
        for replicate in 1:5
            seed = rand(rng, Int64(1):typemax(Int64))
            while seed in seeds
                seed = rand(rng, Int64(1):typemax(Int64))
            end
            push!(seeds, seed)
            instance_id = if options.preset == "table5"
                "table5-m$(m)-n$(n)-r$(lpad(replicate, 2, '0'))"
            else
                "pilot-r$(lpad(replicate, 2, '0'))"
            end
            push!(
                instances,
                Dict(
                    "id" => instance_id,
                    "preset" => options.preset,
                    "m" => m,
                    "n" => n,
                    "density" => density,
                    "replicate" => replicate,
                    "seed" => string(seed),
                ),
            )
        end
    end
    return Dict{String,Any}(
        "schema_version" => 1,
        "generator_version" => 1,
        "preset" => options.preset,
        "master_seed" => string(options.master_seed),
        "replicates" => 5,
        "julia_version" => string(VERSION),
        "project_sha256" => file_sha256(options.generator_project),
        "manifest_sha256" => file_sha256(options.generator_manifest),
        "script_sha256" => file_sha256(options.generator_script),
        "instances" => instances,
    )
end

function atomic_toml(path, document)
    mkpath(dirname(path))
    temporary, stream = mktemp(dirname(path))
    try
        TOML.print(stream, document; sorted = true)
        close(stream)
        mv(temporary, path; force = true)
    catch
        isopen(stream) && close(stream)
        isfile(temporary) && rm(temporary)
        rethrow()
    end
end

function main()
    options = parse_cli(ARGS)
    manifest = make_manifest(options)
    atomic_toml(options.output, manifest)
    println(
        "TABLE5_MANIFEST_WRITTEN path=$(options.output) " *
        "instances=$(length(manifest["instances"])) " *
        "julia_version=$(manifest["julia_version"])",
    )
end

main()
