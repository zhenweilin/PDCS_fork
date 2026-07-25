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

const ROOT = abspath(option("root", "benchmark/results/rebuttal/nsys"))
const OUTPUT = abspath(option("output", joinpath(ROOT, "nsys_summary.csv")))

function manifest(path)
    values = Dict{String,String}()
    isfile(path) || return values
    for line in eachline(path)
        occursin('=', line) || continue
        key, value = split(line, '='; limit=2)
        values[key] = value
    end
    return values
end

function parse_utilization(path)
    values = Float64[]
    isfile(path) || return values
    for line in eachline(path)
        fields = strip.(split(line, ','))
        length(fields) >= 4 || continue
        value = tryparse(Float64, fields[4])
        value === nothing || push!(values, value)
    end
    return values
end

function launch_data(path)
    isfile(path) || return (missing, missing, missing)
    text = read(path, String)
    found = match(r"launches=(\d+) elapsed_seconds=([0-9.eE+-]+)", text)
    found === nothing && return (missing, missing, missing)
    launches = parse(Int, found.captures[1])
    elapsed = parse(Float64, found.captures[2])
    return launches, elapsed, launches / elapsed
end

csv(value) = value === missing ? "" : '"' * replace(string(value), '"' => "\"\"") * '"'
q(values, probability) = quantile(values, probability)

run_dirs = isdir(ROOT) ? sort(filter(isdir, readdir(ROOT; join=true))) : String[]
mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    println(io, "run_id,kind,case,strategy,seed,warm_start,cone_count,cone_dimension,duration_seconds,sample_count,gpu_busy_mean,gpu_busy_median,gpu_busy_p10,gpu_busy_p90,gpu_busy_peak,launches,measured_seconds,launches_per_second,exit_status,nsys_report,gpu_metrics_complete")
    for run_dir in run_dirs
        env = manifest(joinpath(run_dir, "environment.txt"))
        isempty(env) && continue
        utilization = parse_utilization(joinpath(run_dir, "nvidia_smi.csv"))
        launches, elapsed, rate = launch_data(joinpath(run_dir, "application.log"))
        status = isfile(joinpath(run_dir, "exit_status.txt")) ?
                 strip(read(joinpath(run_dir, "exit_status.txt"), String)) : "missing"
        report = isfile(joinpath(run_dir, "trace.nsys-rep"))
        metrics_complete = !isfile(joinpath(run_dir, "PROFILE_INCOMPLETE.txt"))
        statistics = isempty(utilization) ?
            (missing, missing, missing, missing, missing) :
            (mean(utilization), median(utilization), q(utilization, 0.1),
             q(utilization, 0.9), maximum(utilization))
        fields = (
            get(env, "run_id", basename(run_dir)), get(env, "kind", ""),
            get(env, "case", ""), get(env, "strategy", ""), get(env, "seed", ""),
            get(env, "warm_start", ""), get(env, "cone_count", ""),
            get(env, "cone_dimension", ""), get(env, "duration_seconds", ""),
            length(utilization), statistics..., launches, elapsed, rate, status,
            report, metrics_complete,
        )
        println(io, join(csv.(fields), ','))
    end
end

@info "Nsight Systems runs summarized" root=ROOT output=OUTPUT runs=length(run_dirs)
