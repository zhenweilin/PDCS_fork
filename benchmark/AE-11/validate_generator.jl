#!/usr/bin/env julia

include(joinpath(@__DIR__, "AE11Common.jl"))

using .AE11Common
using Dates
using LinearAlgebra
using SparseArrays

function parse_cli(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        key = arguments[index]
        startswith(key, "--") || error("unexpected positional argument: $key")
        index == length(arguments) && error("missing value for $key")
        values[key] = arguments[index + 1]
        index += 2
    end
    return (
        config = abspath(get(values, "--config", AE11Common.config_path())),
        output = abspath(get(
            values, "--output",
            joinpath(@__DIR__, "results", "generator_validation.toml"),
        )),
        quick = lowercase(get(values, "--quick", "false")) == "true",
        run_svd = lowercase(get(values, "--run-svd", "true")) == "true",
    )
end

function orthogonality_error(matrix)
    columns = size(matrix, 2)
    return norm(transpose(matrix) * matrix - I(columns), Inf)
end

function condition_measure(matrix, run_svd)
    run_svd || return (NaN, NaN, NaN)
    values = svdvals(Matrix(matrix))
    return (maximum(values) / minimum(values), maximum(values), minimum(values))
end

function main()
    options = parse_cli(ARGS)
    config = load_config(options.config)
    pilot = config["pilot"]
    m = options.quick ? 32 : Int(pilot["m"])
    q = Int(config["q"])
    seeds = options.quick ? [Int(first(pilot["seeds"]))] : Int.(pilot["seeds"])
    kappas = Float64.(config["kappas"])
    rows = Any[]
    accepted = true
    for seed in seeds
        design = make_design(
            m, q, Int(config["u_block_size"]),
            Int(config["v_block_size"]), seed,
        )
        pattern = AE11Common.build_pattern(design)
        pattern_hashes = String[]
        for kappa in kappas
            started = time()
            A, sigma = assemble_matrix(design, pattern, kappa)
            measured, measured_max, measured_min = condition_measure(
                A, options.run_svd,
            )
            relative_error = options.run_svd ?
                abs(measured - kappa) / kappa : NaN
            passed = eltype(A) == Float64 &&
                matrix_pattern_hash(A) == matrix_pattern_hash(pattern) &&
                (!options.run_svd || relative_error <=
                    (kappa <= 1e6 ? 1e-8 : 5e-7))
            accepted &= passed
            push!(pattern_hashes, matrix_pattern_hash(A))
            push!(rows, Dict{String,Any}(
                "generator" => "structured",
                "seed" => seed,
                "m" => m,
                "n" => q * m,
                "target_kappa" => kappa,
                "known_kappa" => maximum(sigma) / minimum(sigma),
                "measured_kappa" => measured,
                "measured_sigma_max" => measured_max,
                "measured_sigma_min" => measured_min,
                "relative_kappa_error" => relative_error,
                "pattern_hash" => matrix_pattern_hash(A),
                "value_type" => string(eltype(A)),
                "nnz" => nnz(A),
                "b_norm_2" => norm(design.b),
                "elapsed_seconds" => time() - started,
                "passed" => passed,
            ))
        end
        length(unique(pattern_hashes)) == 1 ||
            error("structured pattern changed across kappa for seed $seed")

        gaussian_started = time()
        gaussian = gaussian_qr_components(m, q, seed)
        u_error = orthogonality_error(gaussian.U)
        v_error = orthogonality_error(gaussian.V)
        for kappa in kappas
            started = time()
            A, sigma = gaussian_qr_matrix(gaussian, kappa)
            measured, measured_max, measured_min = condition_measure(
                A, options.run_svd,
            )
            relative_error = options.run_svd ?
                abs(measured - kappa) / kappa : NaN
            passed = eltype(A) == Float64 &&
                u_error <= 1e-11 && v_error <= 1e-11 &&
                (!options.run_svd || relative_error <=
                    (kappa <= 1e6 ? 1e-8 : 5e-7))
            accepted &= passed
            push!(rows, Dict{String,Any}(
                "generator" => "gaussian_qr",
                "seed" => seed,
                "m" => m,
                "n" => q * m,
                "target_kappa" => kappa,
                "known_kappa" => maximum(sigma) / minimum(sigma),
                "measured_kappa" => measured,
                "measured_sigma_max" => measured_max,
                "measured_sigma_min" => measured_min,
                "relative_kappa_error" => relative_error,
                "u_orthogonality_error_inf" => u_error,
                "v_orthogonality_error_inf" => v_error,
                "value_type" => string(eltype(A)),
                "b_norm_2" => norm(gaussian.b),
                "component_generation_seconds" => time() - gaussian_started,
                "elapsed_seconds" => time() - started,
                "passed" => passed,
            ))
        end
    end
    output = Dict{String,Any}(
        "schema_version" => 1,
        "accepted" => accepted,
        "quick" => options.quick,
        "svd_executed" => options.run_svd,
        "generated_utc" => string(now(UTC)),
        "config_sha256" => file_sha256(options.config),
        "common_sha256" => file_sha256(joinpath(@__DIR__, "AE11Common.jl")),
        "rows" => rows,
    )
    write_toml_atomic(options.output, output)
    println("AE11_GENERATOR_VALIDATION accepted=$accepted rows=$(length(rows)) output=$(options.output)")
    return accepted ? 0 : 1
end

exit(main())

