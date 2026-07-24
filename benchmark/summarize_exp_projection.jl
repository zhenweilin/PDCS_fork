#!/usr/bin/env julia

using Printf
using Statistics

function option(name, default=nothing)
    prefix = "--$(name)="
    for (index, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix)+1:end]
        value == "--$(name)" && index < length(ARGS) && return ARGS[index+1]
    end
    return default
end

const RAW = abspath(option("raw", "benchmark/results/exp_projection/raw.csv"))
const SUMMARY = abspath(option("summary", "benchmark/results/exp_projection/summary.csv"))
const REPORT = abspath(option("report", "benchmark/results/exp_projection/report.md"))

unquote(value) = replace(strip(value), r"^\"|\"$" => "", "\"\"" => "\"")
lines = readlines(RAW)
columns = unquote.(split(lines[1], ','))
column = Dict(name => index for (index, name) in pairs(columns))
required = ("cone_count", "input_distribution", "input_sigma",
            "diagonal_distribution", "diagonal_sigma", "variant", "strategy",
            "runtime_ms", "max_error", "status")
all(haskey(column, name) for name in required) ||
    error("raw CSV is missing one of these columns: $(join(required, ", "))")
rows = [unquote.(split(line, ',')) for line in lines[2:end] if !isempty(strip(line))]
groups = Dict{Tuple{Int,String,String,String,String,String,String},Vector{Vector{String}}}()
for row in rows
    row[column["status"]] == "PASS" || continue
    key = (parse(Int, row[column["cone_count"]]), row[column["input_distribution"]],
           row[column["input_sigma"]], row[column["diagonal_distribution"]],
           row[column["diagonal_sigma"]], row[column["variant"]], row[column["strategy"]])
    push!(get!(groups, key, Vector{Vector{String}}()), row)
end

mkpath(dirname(SUMMARY))
open(SUMMARY, "w") do io
    println(io, "cone_count,cone_dimension,input_distribution,input_sigma,diagonal_distribution,diagonal_sigma,variant,strategy,completed_trials,mean_ms,std_ms,median_ms,min_ms,max_ms,max_error,status")
    for key in sort(collect(keys(groups)); by=x -> (x[1], x[2], parse(Float64, x[3]), x[4], parse(Float64, x[5]), x[6], x[7]))
        count, distribution, sigma, diagonal_distribution, diagonal_sigma, variant, strategy = key
        group = groups[key]
        times = parse.(Float64, getindex.(group, column["runtime_ms"]))
        errors = parse.(Float64, getindex.(group, column["max_error"]))
        deviation = length(times) > 1 ? std(times) : 0.0
        @printf(io, "%d,3,%s,%s,%s,%s,%s,%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,COMPLETE\n",
                count, distribution, sigma, diagonal_distribution, diagonal_sigma,
                variant, strategy, length(times), mean(times), deviation,
                median(times), minimum(times), maximum(times), maximum(errors))
    end
end

failures = count(row -> row[column["status"]] == "FAIL", rows)
skipped = count(row -> row[column["status"]] == "SKIPPED", rows)
passes = count(row -> row[column["status"]] == "PASS", rows)
open(REPORT, "w") do io
    println(io, "# Exponential-cone GPU projection results\n")
    println(io, "Exponential cones have fixed dimension 3. The experiment therefore varies cone count, input heterogeneity, projection variant, and GPU hierarchy strategy.\n")
    println(io, "- Passing timed records: $passes")
    println(io, "- Failed timed records: $failures")
    println(io, "- Deliberately skipped configurations: $skipped")
    println(io, "- Raw data: `$(relpath(RAW, dirname(REPORT)))`")
    println(io, "- Summary: `$(relpath(SUMMARY, dirname(REPORT)))`\n")
    println(io, "Every timed record contains its input-distribution label and a maximum absolute error against CPU projections sampled uniformly across the cone collection.")
end

@info "Exponential-cone results summarized" raw=RAW summary=SUMMARY report=REPORT
