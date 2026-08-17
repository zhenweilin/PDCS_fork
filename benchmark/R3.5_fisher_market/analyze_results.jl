#!/usr/bin/env julia

using Printf
using Statistics
using TOML

length(ARGS) == 3 || error(
    "usage: analyze_results.jl CONFIG RESULTS_DIR OUTPUT_DIR",
)
config_path, results_dir, output_dir = abspath.(ARGS)
config = TOML.parsefile(config_path)
formal = config["formal"]
solver = config["solver"]
verification = config["verification"]
seeds = Int.(formal["seeds"])
modes = String.(solver["compared_modes"])
expected = Set((seed, mode) for seed in seeds for mode in modes)

paths = sort(filter(
    path -> endswith(path, ".toml"),
    readdir(results_dir; join = true),
))
rows = [TOML.parsefile(path) for path in paths]
key(row) = (Int(row["seed"]), String(row["rescaling"]))
actual = key.(rows)
length(unique(actual)) == length(actual) || error("duplicate seed/mode rows")
isempty(setdiff(expected, Set(actual))) || error("missing configured rows")
isempty(setdiff(Set(actual), expected)) || error("unexpected result rows")
length(rows) == Int(formal["expected_records"]) ||
    error("wrong result count")

limit = Float64(verification["maximum_independent_primal_violation"])
expected_config_hashes = unique(getindex.(rows, "config_sha256"))
length(expected_config_hashes) == 1 || error("mixed config hashes")
length(unique(getindex.(rows, "runner_sha256"))) == 1 ||
    error("mixed runner hashes")
length(unique(getindex.(rows, "direct_formulation_sha256"))) == 1 ||
    error("mixed formulation hashes")
length(unique(getindex.(rows, "generator_sha256"))) == 1 ||
    error("mixed generator hashes")

for row in rows
    Int(row["m"]) == Int(formal["m"]) || error("m mismatch")
    Int(row["n"]) == Int(formal["n"]) || error("n mismatch")
    Float64(row["density"]) == Float64(formal["density"]) ||
        error("density mismatch")
    row["formulation"] == "direct" || error("not the direct formulation")
    row["rescaling_method"] == "ruiz_pock_chambolle" ||
        error("wrong rescaling pipeline")
    expected_scalar = row["rescaling"] == "scalar_cone"
    Bool(row["scalar_cone_rescaling"]) == expected_scalar ||
        error("mode/Boolean mismatch")
    Bool(row["use_preconditioner"]) || error("preconditioner disabled")
    Int(row["nonfinite_primal_count"]) == 0 || error("nonfinite primal")
    all(isfinite(Float64(row[name])) for name in (
        "objective_value", "supply_rel_residual",
        "nonnegative_violation", "exponential_log_violation",
        "solver_reported_seconds",
    )) || error("nonfinite scalar metric")
    independent = Float64(row["supply_rel_residual"]) <= limit &&
        Float64(row["nonnegative_violation"]) <= limit &&
        Float64(row["exponential_log_violation"]) <= limit
    Bool(row["independently_verified"]) == independent ||
        error("stored independent decision mismatch")
end

for seed in seeds
    pair = filter(row -> Int(row["seed"]) == seed, rows)
    length(unique(getindex.(pair, "numerical_digest"))) == 1 ||
        error("seed $seed modes used different instances")
    length(unique(getindex.(pair, "cuda_visible_devices"))) == 1 ||
        error("seed $seed pair did not run on the same physical GPU")
end

function geometric_mean(values)
    all(value -> value > 0.0, values) || error("nonpositive GM input")
    return exp(mean(log.(values)))
end

diagonal = sort(filter(row -> row["rescaling"] == "diagonal", rows);
    by = row -> Int(row["seed"]))
scalar = sort(filter(row -> row["rescaling"] == "scalar_cone", rows);
    by = row -> Int(row["seed"]))
time_ratios = Float64[
    scalar[index]["solver_reported_seconds"] /
    diagonal[index]["solver_reported_seconds"] for index in eachindex(seeds)
]
iteration_ratios = Float64[
    scalar[index]["iterations"] / diagonal[index]["iterations"]
    for index in eachindex(seeds)
]
objective_relative_differences = Float64[
    abs(scalar[index]["objective_value"] - diagonal[index]["objective_value"]) /
    max(1.0, abs(diagonal[index]["objective_value"]))
    for index in eachindex(seeds)
]

summary = Dict{String,Any}(
    "schema_version" => 1,
    "records" => length(rows),
    "diagonal_verified" => count(row -> row["status"] == "solved_verified", diagonal),
    "scalar_cone_verified" => count(row -> row["status"] == "solved_verified", scalar),
    "diagonal_median_iterations" => median(Int.(getindex.(diagonal, "iterations"))),
    "scalar_cone_median_iterations" => median(Int.(getindex.(scalar, "iterations"))),
    "diagonal_median_solver_seconds" => median(Float64.(getindex.(diagonal, "solver_reported_seconds"))),
    "scalar_cone_median_solver_seconds" => median(Float64.(getindex.(scalar, "solver_reported_seconds"))),
    "geometric_mean_time_ratio_scalar_over_diagonal" => geometric_mean(time_ratios),
    "median_time_ratio_scalar_over_diagonal" => median(time_ratios),
    "geometric_mean_iteration_ratio_scalar_over_diagonal" => geometric_mean(iteration_ratios),
    "minimum_time_ratio_scalar_over_diagonal" => minimum(time_ratios),
    "maximum_time_ratio_scalar_over_diagonal" => maximum(time_ratios),
    "maximum_diagonal_exp_violation" => maximum(Float64.(getindex.(diagonal, "exponential_log_violation"))),
    "minimum_scalar_cone_exp_violation" => minimum(Float64.(getindex.(scalar, "exponential_log_violation"))),
    "maximum_scalar_cone_exp_violation" => maximum(Float64.(getindex.(scalar, "exponential_log_violation"))),
    "maximum_paired_relative_objective_difference" => maximum(objective_relative_differences),
)

mkpath(output_dir)
open(joinpath(output_dir, "summary.toml"), "w") do stream
    TOML.print(stream, summary; sorted = true)
end
open(joinpath(output_dir, "pairs.csv"), "w") do stream
    println(stream, "seed,diagonal_status,scalar_cone_status,diagonal_iterations,scalar_cone_iterations,diagonal_solver_seconds,scalar_cone_solver_seconds,iteration_ratio,time_ratio,diagonal_exp_violation,scalar_cone_exp_violation")
    for index in eachindex(seeds)
        d, s = diagonal[index], scalar[index]
        println(stream, join((
            seeds[index], d["status"], s["status"], d["iterations"],
            s["iterations"], d["solver_reported_seconds"],
            s["solver_reported_seconds"], iteration_ratios[index],
            time_ratios[index], d["exponential_log_violation"],
            s["exponential_log_violation"],
        ), ','))
    end
end
open(joinpath(output_dir, "RESULTS.md"), "w") do stream
    println(stream, "# R3.5 Fisher diagonal versus scalar-per-cone results\n")
    println(stream, "The direct Fisher formulation has 500,000 allocation variables and places `U_ij` directly in each exponential cone's third coordinate. Both modes use Ruiz--Pock--Chambolle rescaling; only `scalar_cone_rescaling=false/true` changes.\n")
    println(stream, "| Seed | Diagonal verified | Scalar verified | Diagonal iter. | Scalar iter. | Diagonal time (s) | Scalar time (s) | Time ratio | Diagonal EXP violation | Scalar EXP violation |")
    println(stream, "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
    for index in eachindex(seeds)
        d, s = diagonal[index], scalar[index]
        @printf(stream, "| %d | %s | %s | %d | %d | %.2f | %.2f | %.2f | %.3e | %.3e |\n",
            seeds[index], d["status"] == "solved_verified" ? "yes" : "no",
            s["status"] == "solved_verified" ? "yes" : "no",
            d["iterations"], s["iterations"], d["solver_reported_seconds"],
            s["solver_reported_seconds"], time_ratios[index],
            d["exponential_log_violation"], s["exponential_log_violation"])
    end
    @printf(stream, "\nDiagonal verifies **%d/5** rows versus **%d/5** for scalar-per-cone. The median iteration counts are %.0f and %.0f; median solver times are %.2f s and %.2f s. The geometric-mean scalar/diagonal time ratio is **%.2fx** (range %.2fx--%.2fx).\n",
        summary["diagonal_verified"], summary["scalar_cone_verified"],
        summary["diagonal_median_iterations"],
        summary["scalar_cone_median_iterations"],
        summary["diagonal_median_solver_seconds"],
        summary["scalar_cone_median_solver_seconds"],
        summary["geometric_mean_time_ratio_scalar_over_diagonal"],
        summary["minimum_time_ratio_scalar_over_diagonal"],
        summary["maximum_time_ratio_scalar_over_diagonal"])
end

println("R35_FISHER_AUDIT records=$(length(rows)) output=$output_dir")
