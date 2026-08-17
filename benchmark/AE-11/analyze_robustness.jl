#!/usr/bin/env julia

using Printf
using Statistics
using TOML

length(ARGS) == 3 || error(
    "usage: analyze_robustness.jl CONFIG RESULTS_DIR OUTPUT_DIR",
)
config_path, results_dir, output_dir = abspath.(ARGS)
config = TOML.parsefile(config_path)
protocol = config["protocol"]
medium = config["medium"]
seeds = Int.(medium["seeds"])
kappas = Float64.(config["kappas"])
expected = Set((seed, kappa) for seed in seeds for kappa in kappas)

paths = sort(filter(
    path -> endswith(path, ".toml"),
    readdir(results_dir; join = true),
))
rows = [TOML.parsefile(path) for path in paths]
rows = filter(row -> get(row, "solver", "") == "cupdcs", rows)

function logical_key(row)
    return (Int(row["seed"]), Float64(row["target_kappa"]))
end

actual = [logical_key(row) for row in rows]
length(unique(actual)) == length(actual) || error("duplicate seed/K records")
missing = setdiff(expected, Set(actual))
extra = setdiff(Set(actual), expected)
isempty(missing) || error("missing configured records: $(sort!(collect(missing)))")
isempty(extra) || error("unexpected records: $(sort!(collect(extra)))")
expected_records = Int(protocol["expected_records"])
length(rows) == expected_records || error(
    "expected $expected_records records, found $(length(rows))",
)

expected_alpha = Float64(protocol["alpha"])
verification_tolerance = Float64(protocol["verification_tolerance"])
solver_tolerance = Float64(protocol["solver_tolerance"])
m, n = Int(medium["m"]), Int(medium["n"])
for row in rows
    Int(row["m"]) == m && Int(row["n"]) == n || error("dimension mismatch")
    row["panel"] == "B" || error("unexpected penalty panel")
    Float64(row["alpha"]) == expected_alpha || error("alpha mismatch")
    row["rescaling_on"] === true || error("production rescaling is not on")
    Float64(row["tolerance"]) == verification_tolerance ||
        error("verification tolerance mismatch")
    Float64(row["solver_tolerance"]) == solver_tolerance ||
        error("solver tolerance mismatch")
    row["value_type"] == "Float64" || error("non-Float64 result")
    Bool(row["verified_solved"]) || error(
        "independent verifier failed for $(logical_key(row))",
    )
    row_status = row["status"]
    row_status == "solved_verified" || error(
        "non-verified status for $(logical_key(row)): $row_status",
    )
    isfinite(Float64(row["x_norm_1"])) && Float64(row["x_norm_1"]) > 0 ||
        error("zero or non-finite solution for $(logical_key(row))")
end

for seed in seeds
    seed_rows = filter(row -> Int(row["seed"]) == seed, rows)
    length(unique(getindex.(seed_rows, "matrix_pattern_hash"))) == 1 ||
        error("matrix pattern changed across K for seed $seed")
    length(unique(getindex.(seed_rows, "b_hash"))) == 1 ||
        error("response vector changed across K for seed $seed")
end
length(unique(getindex.(rows, "config_sha256"))) == 1 ||
    error("mixed configuration hashes")
length(unique(getindex.(rows, "common_sha256"))) == 1 ||
    error("mixed generator/verifier hashes")

summary_rows = Dict{String,Any}[]
for kappa in kappas
    group = filter(row -> Float64(row["target_kappa"]) == kappa, rows)
    push!(summary_rows, Dict{String,Any}(
        "target_kappa" => kappa,
        "verified" => count(row -> row["status"] == "solved_verified", group),
        "total" => length(group),
        "median_solve_seconds" => median(Float64.(getindex.(group, "solve_seconds"))),
        "max_solve_seconds" => maximum(Float64.(getindex.(group, "solve_seconds"))),
        "median_iterations" => median(Float64.(getindex.(group, "iterations"))),
        "max_independent_kkt" => maximum(Float64.(getindex.(group, "independent_kkt"))),
        "max_relative_gap" => maximum(Float64.(getindex.(group, "relative_gap"))),
        "min_x_norm_1" => minimum(Float64.(getindex.(group, "x_norm_1"))),
        "max_x_norm_1" => maximum(Float64.(getindex.(group, "x_norm_1"))),
        "max_relative_kappa_error" => maximum(
            abs(Float64(row["measured_kappa"]) - kappa) / kappa for row in group
        ),
    ))
end

mkpath(output_dir)
open(joinpath(output_dir, "summary.csv"), "w") do io
    println(io, "target_kappa,verified,total,median_solve_seconds,max_solve_seconds,median_iterations,max_independent_kkt,max_relative_gap,min_x_norm_1,max_x_norm_1,max_relative_kappa_error")
    for row in summary_rows
        println(io, join((
            row["target_kappa"], row["verified"], row["total"],
            row["median_solve_seconds"], row["max_solve_seconds"],
            row["median_iterations"], row["max_independent_kkt"],
            row["max_relative_gap"], row["min_x_norm_1"],
            row["max_x_norm_1"], row["max_relative_kappa_error"],
        ), ','))
    end
end

summary_toml = Dict{String,Any}(
    "schema_version" => 1,
    "experiment" => config["experiment"],
    "records" => length(rows),
    "verified" => count(row -> row["status"] == "solved_verified", rows),
    "all_verified" => all(row -> row["status"] == "solved_verified", rows),
    "nonzero_solutions" => all(Float64(row["x_norm_1"]) > 0 for row in rows),
    "solver_tolerance" => solver_tolerance,
    "verification_tolerance" => verification_tolerance,
    "by_condition_number" => summary_rows,
)
open(joinpath(output_dir, "summary.toml"), "w") do io
    TOML.print(io, summary_toml; sorted = true)
end

open(joinpath(output_dir, "RESULTS.md"), "w") do io
    println(io, "# AE-11 simplified robustness results\n")
    println(io, "cuPDCS passed the independent verifier on **$(length(rows))/$(length(rows))** runs over `K = 1` through `K = 1e8`. The fixed `alpha=$(expected_alpha)` solutions are all nonzero.\n")
    println(io, "| condition number K | verified | median solve (s) | max solve (s) | median iterations | max KKT | max relative gap | min ||x||_1 |")
    println(io, "|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows
        @printf(io, "| %.0e | %d/%d | %.2f | %.2f | %.0f | %.3e | %.3e | %.3f |\n",
            row["target_kappa"], row["verified"], row["total"],
            row["median_solve_seconds"], row["max_solve_seconds"],
            row["median_iterations"], row["max_independent_kkt"],
            row["max_relative_gap"], row["min_x_norm_1"])
    end
    println(io, "\nAcceptance uses normalized KKT `<= 1e-6`, normalized dual infeasibility `<= 1e-6`, and relative duality gap `<= 1e-5`. The solver is asked for `2e-7`; the final decision is made only by the independent verifier.")
end

println("AE11_ROBUSTNESS verified=$(length(rows))/$(length(rows)) output=$output_dir")
