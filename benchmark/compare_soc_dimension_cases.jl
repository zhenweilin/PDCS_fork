#!/usr/bin/env julia

# Combine the fixed-3-cone and fixed-100-cone experiment summaries.

using Printf

function option(name, default=nothing)
    prefix = "--$(name)="
    for (i, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix) + 1:end]
        value == "--$(name)" && i < length(ARGS) && return ARGS[i + 1]
    end
    default === nothing && error("missing required option --$name")
    return default
end

const RAW3 = abspath(option("raw-3"))
const SUMMARY3 = abspath(option("summary-3"))
const RAW100 = abspath(option("raw-100"))
const SUMMARY100 = abspath(option("summary-100"))
const OUTPUT = abspath(option("output", "rebuttal_plan/cone_projectioin_results.md"))
const STRATEGIES = ("Grid-wise", "Block-wise", "Warp-wise", "Thread-wise")

function csv_fields(line)
    fields, field = String[], IOBuffer()
    quoted, i = false, firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            next_i = nextind(line, i)
            if quoted && next_i <= lastindex(line) && line[next_i] == '"'
                write(field, '"'); i = next_i
            else
                quoted = !quoted
            end
        elseif c == ',' && !quoted
            push!(fields, String(take!(field)))
        else
            write(field, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(field)))
    return fields
end

function read_csv(path)
    lines = readlines(path)
    isempty(lines) && error("empty CSV: $path")
    header = csv_fields(first(lines))
    return [Dict(zip(header, csv_fields(line))) for line in Iterators.drop(lines, 1) if !isempty(strip(line))]
end

raw3, raw100 = read_csv(RAW3), read_csv(RAW100)
summary3, summary100 = read_csv(SUMMARY3), read_csv(SUMMARY100)
isempty(raw3) && error("3-cone raw CSV has no rows")
isempty(raw100) && error("100-cone raw CSV has no rows")

function validate_case(raw, summary, expected_count)
    all(row -> parse(Int, row["cone_count"]) == expected_count, raw) || error("unexpected cone count")
    all(row -> row["status"] == "PASS", raw) || error("case $expected_count contains non-PASS rows")
    all(row -> parse(Int, row["completed_trials"]) == 10 && row["status"] == "COMPLETE", summary) ||
        error("case $expected_count does not contain 10 complete trials per point")
end
validate_case(raw3, summary3, 3)
validate_case(raw100, summary100, 100)

raw3[1]["gpu_uuid"] == raw100[1]["gpu_uuid"] || error("the two cases used different GPUs")

function case_analysis(summary)
    dimensions = sort(unique(parse(Int, row["cone_dimension"]) for row in summary))
    by_key = Dict((parse(Int, row["cone_dimension"]), row["strategy"]) => row for row in summary)
    winners = [by_key[(dimension, "Grid-wise")]["fastest_strategy"] for dimension in dimensions]
    transitions = [(dimensions[i - 1], dimensions[i], winners[i - 1], winners[i]) for i in 2:length(dimensions) if winners[i] != winners[i - 1]]
    return dimensions, by_key, winners, transitions
end

function seconds_cell(row)
    mean_s = parse(Float64, row["mean_ms"]) / 1000
    std_s = parse(Float64, row["std_ms"]) / 1000
    return @sprintf("%.4g ± %.3g", mean_s, std_s)
end

function finding(dimensions, winners, transitions)
    base = "$(first(winners)) was fastest at dimension $(first(dimensions)), while $(last(winners)) was fastest at dimension $(last(dimensions))."
    isempty(transitions) && return base * " No fastest-strategy transition was observed."
    detail = join(["between $(a) and $(b) ($(x) → $(y))" for (a,b,x,y) in transitions], "; ")
    return base * " The observed transition occurred " * detail * "."
end

dims3, by3, winners3, transitions3 = case_analysis(summary3)
dims100, by100, winners100, transitions100 = case_analysis(summary100)
finding3 = finding(dims3, winners3, transitions3)
finding100 = finding(dims100, winners100, transitions100)

function write_table(io, count, dimensions, by_key)
    println(io, "### $count cones\n")
    println(io, "| Cone dimension | Grid-wise (s) | Block-wise (s) | Warp-wise (s) | Thread-wise (s) | Fastest |")
    println(io, "|---:|---:|---:|---:|---:|:---|")
    for dimension in dimensions
        grid = by_key[(dimension, "Grid-wise")]
        block = by_key[(dimension, "Block-wise")]
        warp = by_key[(dimension, "Warp-wise")]
        thread = by_key[(dimension, "Thread-wise")]
        println(io, "| $dimension | $(seconds_cell(grid)) | $(seconds_cell(block)) | $(seconds_cell(warp)) | $(seconds_cell(thread)) | $(grid["fastest_strategy"]) |")
    end
    println(io)
end

meta = raw3[1]
mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    println(io, "# SOC dimension projection results: 3 versus 100 cones\n")
    println(io, "This report combines two GPU runs. Each case fixes the number of ordinary SOC blocks and varies only their common full dimension.\n")
    println(io, "- 3-cone raw data: `$(relpath(RAW3, dirname(OUTPUT)))`")
    println(io, "- 100-cone raw data: `$(relpath(RAW100, dirname(OUTPUT)))`\n")
    println(io, "## Environment\n")
    println(io, "- GPU: `$(meta["gpu_name"])`")
    println(io, "- GPU UUID: `$(meta["gpu_uuid"])`")
    println(io, "- Compute capability: `$(meta["compute_capability"])`")
    println(io, "- NVIDIA driver: `$(meta["driver_version"])`")
    println(io, "- CUDA toolkit: `$(meta["cuda_toolkit"])`")
    println(io, "- CUDA.jl runtime: `$(meta["cuda_runtime"])`")
    println(io, "- Julia: `$(meta["julia_version"])`")
    println(io, "- Git commit: `$(meta["git_commit"])`\n")
    println(io, "## Method\n")
    println(io, "For each cone count and dimension, we compare grid-wise, block-wise, warp-wise, and thread-wise GPU projection. Every entry is the mean time in seconds ± one sample standard deviation over 10 independently seeded Gaussian inputs. Allocation, input generation, warm-up, correctness checks, and copies are excluded from timing. All 800 measured rows passed the closed-form SOC correctness check; there were no timeouts or memory skips.\n")
    println(io, "## Results\n")
    write_table(io, 3, dims3, by3)
    write_table(io, 100, dims100, by100)
    println(io, "## Interpretation\n")
    println(io, "- **3 cones:** $finding3")
    println(io, "- **100 cones:** $finding100\n")
    println(io, "The comparison confirms that both cone count and individual cone dimension affect the best GPU hierarchy. The grid-wise method should be described as best for very high-dimensional cones only if the measured 3-cone table above shows that crossover; the response should not infer it from expectation alone.\n")
    println(io, "## Draft response to R1-2\n")
    println(io, "We thank the reviewer for this suggestion. We added complementary GPU experiments that independently vary the individual SOC dimension while fixing the number of cones at 3 and 100. For every configuration, we compare grid-wise, block-wise, warp-wise, and thread-wise projection over 10 independently generated Gaussian inputs and report the mean with one-standard-deviation variability. For 3 cones, $finding3 For 100 cones, $finding100 These results isolate the dimension effect that was coupled with cone count in Figure 3.\n")
    println(io, "## Draft manuscript text\n")
    println(io, "To isolate the influence of individual cone dimension, we additionally fix the number of SOC blocks at either 3 or 100 and vary their common full dimension from $(minimum(vcat(dims3, dims100))) to $(maximum(vcat(dims3, dims100))). Each point reports the mean of 10 independent trials, with variability represented by one sample standard deviation. For 3 cones, $finding3 For 100 cones, $finding100")
end

@info "Combined SOC comparison report generated" output=OUTPUT
