#!/usr/bin/env julia

using Printf
using Statistics

function option(name, default)
    prefix = "--$(name)="
    for (i, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix) + 1:end]
        value == "--$(name)" && i < length(ARGS) && return ARGS[i + 1]
    end
    return default
end

const RAW = abspath(option("raw", "benchmark/results/soc_dimension_raw.csv"))
const SUMMARY = abspath(option("summary", "benchmark/results/soc_dimension_summary.csv"))
const REPORT = abspath(option("report", "rebuttal_plan/cone_projectioin_results.md"))
const FIGURE_TEX = abspath(option("figure-tex", "rebuttal_plan/figures/soc_dimension.tex"))
const STRATEGIES = ("Grid-wise", "Block-wise", "Warp-wise", "Thread-wise")
const EXPECTED_TRIALS = parse(Int, get(ENV, "PDCS_SOC_TRIALS", "10"))

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

lines = readlines(RAW)
isempty(lines) && error("raw CSV is empty: $RAW")
header = csv_fields(first(lines))
rows = [Dict(zip(header, csv_fields(line))) for line in Iterators.drop(lines, 1) if !isempty(strip(line))]
isempty(rows) && error("raw CSV has no data rows: $RAW")

dimensions = sort(unique(parse(Int, row["cone_dimension"]) for row in rows))
cone_counts = unique(row["cone_count"] for row in rows)
length(cone_counts) == 1 || error("experiment does not have one fixed cone count")

summaries = NamedTuple[]
for dimension in dimensions, strategy in STRATEGIES
    selected = filter(row -> parse(Int, row["cone_dimension"]) == dimension && row["strategy"] == strategy, rows)
    passed = filter(row -> row["status"] == "PASS" && !isempty(row["runtime_ms"]), selected)
    values = parse.(Float64, getindex.(passed, "runtime_ms"))
    statuses = unique(getindex.(selected, "status"))
    status = "TIMEOUT" in statuses ? "TIMEOUT" :
             isempty(values) ? join(statuses, ";") :
             length(values) == EXPECTED_TRIALS ? "COMPLETE" : "PARTIAL"
    push!(summaries, (
        dimension=dimension, strategy=strategy, trials=length(values),
        mean=isempty(values) ? NaN : mean(values),
        std=length(values) <= 1 ? 0.0 : std(values),
        median=isempty(values) ? NaN : median(values),
        minimum=isempty(values) ? NaN : minimum(values),
        maximum=isempty(values) ? NaN : maximum(values), status=status,
    ))
end

fastest = Dict{Int,String}()
for dimension in dimensions
    candidates = filter(s -> s.dimension == dimension && isfinite(s.mean), summaries)
    fastest[dimension] = isempty(candidates) ? "none" : argmin(s -> s.mean, candidates).strategy
end
valid_dims = filter(d -> fastest[d] != "none", dimensions)
winners = [fastest[d] for d in valid_dims]
transitions = [(valid_dims[i - 1], valid_dims[i], winners[i - 1], winners[i]) for i in 2:length(valid_dims) if winners[i] != winners[i - 1]]
finding = if isempty(valid_dims)
    "No completed timing cases are available."
elseif isempty(transitions)
    "$(first(winners)) was fastest throughout the completed dimension range $(first(valid_dims))–$(last(valid_dims)); no fastest-strategy crossover was observed."
else
    transition_text = join(["between dimensions $(a) and $(b) ($(x) → $(y))" for (a,b,x,y) in transitions], "; ")
    "$(first(winners)) was fastest at dimension $(first(valid_dims)), while $(last(winners)) was fastest at dimension $(last(valid_dims)). The observed fastest-strategy transition occurred $transition_text."
end

mkpath(dirname(SUMMARY))
open(SUMMARY, "w") do io
    println(io, "cone_count,cone_dimension,strategy,completed_trials,mean_ms,std_ms,median_ms,min_ms,max_ms,fastest_strategy,status")
    for s in summaries
        @printf(io, "%s,%d,\"%s\",%d,%.9g,%.9g,%.9g,%.9g,%.9g,\"%s\",%s\n",
                only(cone_counts), s.dimension, s.strategy, s.trials, s.mean, s.std,
                s.median, s.minimum, s.maximum, fastest[s.dimension], s.status)
    end
end

function timing(s)
    isfinite(s.mean) || return s.status
    return @sprintf("%.4g ± %.3g", s.mean / 1000, s.std / 1000)
end

first_row = first(rows)
mkpath(dirname(REPORT))
open(REPORT, "w") do io
    println(io, "# Fixed-count SOC dimension projection results\n")
    println(io, "This report is generated from `$(relpath(RAW, dirname(REPORT)))`. Do not edit measured values by hand.\n")
    if EXPECTED_TRIALS != 10
        println(io, "> **Diagnostic run only.** This report uses $EXPECTED_TRIALS trials instead of the 10-trial publication preset. Do not cite these timings in the manuscript or reviewer response.\n")
    end
    println(io, "## Environment\n")
    println(io, "- GPU: `$(first_row["gpu_name"])`")
    println(io, "- GPU UUID: `$(first_row["gpu_uuid"])`")
    println(io, "- Compute capability: `$(first_row["compute_capability"])`")
    println(io, "- NVIDIA driver: `$(first_row["driver_version"])`")
    println(io, "- CUDA toolkit used to build native kernels: `$(first_row["cuda_toolkit"])`")
    println(io, "- CUDA runtime used by CUDA.jl: `$(first_row["cuda_runtime"])`")
    println(io, "- Julia: `$(first_row["julia_version"])`")
    println(io, "- Git commit: `$(first_row["git_commit"])`\n")
    println(io, "## Experiment\n")
    println(io, "We fixed the number of ordinary second-order cones at **$(only(cone_counts))** and varied only the full dimension of each cone. Entries are mean projection time in seconds ± one sample standard deviation over up to $EXPECTED_TRIALS independent trials. Input generation, allocation, warm-up, correctness checks, and host/device copies are excluded.\n")
    println(io, "| Cone dimension | Grid-wise (s) | Block-wise (s) | Warp-wise (s) | Thread-wise (s) | Fastest |")
    println(io, "|---:|---:|---:|---:|---:|:---|")
    for dimension in dimensions
        by_strategy = Dict(s.strategy => s for s in summaries if s.dimension == dimension)
        println(io, "| $dimension | $(timing(by_strategy["Grid-wise"])) | $(timing(by_strategy["Block-wise"])) | $(timing(by_strategy["Warp-wise"])) | $(timing(by_strategy["Thread-wise"])) | $(fastest[dimension]) |")
    end
    failures = count(row -> row["status"] ∉ ("PASS", "SKIPPED_MEMORY"), rows)
    skipped = count(row -> row["status"] == "SKIPPED_MEMORY", rows)
    println(io, "\nCorrectness/timing exceptions: **$failures**. Memory-skipped rows: **$skipped**.\n")
    println(io, "## Interpretation\n")
    if isempty(valid_dims)
        println(io, "No completed timing cases are available. Resolve the statuses in the raw CSV before using this report.\n")
    else
        println(io, "At the smallest completed dimension ($(first(valid_dims))), the fastest method was **$(first(winners))**; at the largest completed dimension ($(last(valid_dims))), it was **$(last(winners))**.")
        if isempty(transitions)
            println(io, "No fastest-strategy crossover was observed over the completed range. This is itself the measured result and should not be replaced by an expected crossover.\n")
        else
            println(io, "Observed fastest-strategy transitions occurred between: " * join(["dimensions $(a) and $(b) ($(x) → $(y))" for (a,b,x,y) in transitions], "; ") * ".\n")
        end
    end
    println(io, "## Draft response to R1-2\n")
    println(io, "We thank the reviewer for this suggestion. We added a complementary GPU experiment that fixes the number of second-order cones at $(only(cone_counts)) and varies only the dimension of each cone. For every dimension, we compare the grid-wise, block-wise, warp-wise, and thread-wise projection implementations using $EXPECTED_TRIALS independently generated Gaussian inputs and report mean runtime with one-standard-deviation bands. The new experiment separates the effect of cone dimension from the cone-count effect in Figure 3. $finding\n")
    println(io, "## Draft manuscript text\n")
    println(io, "To isolate the influence of individual cone dimension, we additionally fix the number of SOC blocks at $(only(cone_counts)) and vary their common full dimension. We project independent standard-Gaussian vectors using grid-wise, block-wise, warp-wise, and thread-wise GPU strategies. Each point reports the mean of $EXPECTED_TRIALS independent trials, and the shaded region represents one sample standard deviation. $finding\n")
    println(io, "**Suggested caption.** Runtime of GPU projections onto $(only(cone_counts)) equal-dimensional second-order cones as the individual cone dimension varies. Curves show means over $EXPECTED_TRIALS independent trials; shaded regions indicate one sample standard deviation. Cases exceeding 15 seconds are reported as timeouts.")
end

colors = Dict("Grid-wise" => "blue", "Block-wise" => "red",
              "Warp-wise" => "orange", "Thread-wise" => "teal")
mkpath(dirname(FIGURE_TEX))
open(FIGURE_TEX, "w") do io
    println(io, raw"\documentclass[tikz,border=2pt]{standalone}")
    println(io, raw"\usepackage{pgfplots}")
    println(io, raw"\usepgfplotslibrary{fillbetween}")
    println(io, raw"\pgfplotsset{compat=1.17}")
    println(io, raw"\begin{document}")
    println(io, raw"\begin{tikzpicture}")
    println(io, raw"\begin{axis}[width=10cm,height=6.2cm,xmode=log,ymode=log,xlabel={Individual SOC dimension},ylabel={Projection time (s)},legend pos=north west,grid=major]")
    for strategy in STRATEGIES
        data = filter(s -> s.strategy == strategy && isfinite(s.mean), summaries)
        isempty(data) && continue
        key = replace(lowercase(strategy), "-wise" => "")
        upper = join(["($(s.dimension),$((s.mean + s.std) / 1000))" for s in data], " ")
        lower = join(["($(s.dimension),$(max((s.mean - s.std) / 1000, eps())))" for s in data], " ")
        center = join(["($(s.dimension),$(s.mean / 1000))" for s in data], " ")
        println(io, "\\addplot[name path=$(key)upper,draw=none] coordinates {$upper};")
        println(io, "\\addplot[name path=$(key)lower,draw=none] coordinates {$lower};")
        println(io, "\\addplot[$(colors[strategy])!18] fill between[of=$(key)upper and $(key)lower];")
        println(io, "\\addplot[$(colors[strategy]),thick,mark=*] coordinates {$center};")
        println(io, "\\addlegendentry{$strategy}")
        timeout_dimensions = sort(unique(parse(Int, row["cone_dimension"]) for row in rows if row["strategy"] == strategy && row["status"] == "TIMEOUT"))
        if !isempty(timeout_dimensions)
            timeout_points = join(["($dimension,15)" for dimension in timeout_dimensions], " ")
            println(io, "\\addplot[$(colors[strategy]),only marks,mark=triangle*,mark size=3pt] coordinates {$timeout_points};")
        end
    end
    println(io, raw"\end{axis}")
    println(io, raw"\end{tikzpicture}")
    println(io, raw"\end{document}")
end

@info "SOC dimension summary generated" summary=SUMMARY report=REPORT figure_tex=FIGURE_TEX
