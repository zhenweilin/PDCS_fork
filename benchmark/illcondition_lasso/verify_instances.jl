#!/usr/bin/env julia

using Serialization
using SHA
using TOML

include(joinpath(@__DIR__, "ill_conditioned_lasso_cases.jl"))
using .IllConditionedLassoCases

function main()
    VERSION == v"1.10.4" || error("verification requires Julia 1.10.4")
    length(ARGS) == 1 || error("Usage: verify_instances.jl GENERATED_MANIFEST")
    manifest_path = abspath(only(ARGS))
    manifest = TOML.parsefile(manifest_path)
    manifest["generator_version"] == GENERATOR_VERSION || error(
        "generator version mismatch",
    )
    manifest["julia_version"] == string(VERSION) || error("Julia version mismatch")
    source_config = manifest["source_config"]
    isfile(source_config) || error("missing source config: $source_config")
    bytes2hex(sha256(read(source_config))) == manifest["source_config_sha256"] ||
        error("source config changed after instance generation")
    entries = manifest["instances"]
    length(entries) == 6 || error("formal manifest must contain six instances")
    matrix_hashes = Set{String}()
    for entry in entries
        cache_path = joinpath(dirname(manifest_path), entry["cache"])
        filesize(cache_path) == entry["cache_bytes"] || error(
            "cache size mismatch: $cache_path",
        )
        instance = deserialize(cache_path)
        instance isa LassoInstance || error("unexpected cache type: $cache_path")
        verification = verify_instance(instance; compute_spectrum = false)
        verification.dimension == 1000 || error("dimension mismatch")
        verification.dense_nonzeros == 1_000_000 || error("A is not dense")
        hashes = instance_hashes(instance)
        hashes.matrix == entry["matrix_sha256"] || error("matrix hash mismatch")
        hashes.b == entry["b_sha256"] || error("b hash mismatch")
        hashes.xstar == entry["xstar_sha256"] || error("xstar hash mismatch")
        hashes.eigenvalues == entry["eigenvalues_sha256"] || error(
            "eigenvalue hash mismatch",
        )
        push!(matrix_hashes, hashes.matrix)
        println(
            "INSTANCE_OK id=$(entry["id"]) kappa=$(entry["condition_number"]) " *
            "matrix_sha256=$(hashes.matrix)",
        )
    end
    length(matrix_hashes) == length(entries) || error("matrix hashes are not unique")
    println("ILL_CONDITIONED_INSTANCE_MANIFEST_OK count=$(length(entries))")
end

main()
