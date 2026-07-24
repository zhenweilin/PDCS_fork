#!/usr/bin/env julia

using Printf

function option(name, default)
    prefix = "--$(name)="
    for (i, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix) + 1:end]
        value == "--$(name)" && i < length(ARGS) && return ARGS[i + 1]
    end
    return default
end

const ROOT = abspath(option("root", "benchmark/results/soc_dimension"))
const OUTPUT = abspath(option("output", "rebuttal_plan/cone_projectioin_results.md"))
const FIGURE = abspath(option("figure-tex", "rebuttal_plan/figures/soc_count_fixed_total.tex"))
const COUNTS = [10, 100, 1_000, 10_000, 100_000, 1_000_000,
                10_000_000, 100_000_000, 120_000_000]
const STRATEGIES = ("Grid-wise", "Block-wise", "Warp-wise", "Thread-wise")
const COLORS = Dict("Grid-wise" => "blue", "Block-wise" => "red", "Warp-wise" => "orange", "Thread-wise" => "teal")

function fields(line)
    result, buffer = String[], IOBuffer()
    quoted = false
    for c in line
        if c == '"'; quoted = !quoted
        elseif c == ',' && !quoted; push!(result, String(take!(buffer)))
        else; write(buffer, c)
        end
    end
    push!(result, String(take!(buffer)))
    result
end

function read_csv(path)
    lines = readlines(path)
    header = fields(first(lines))
    [Dict(zip(header, fields(line))) for line in Iterators.drop(lines, 1) if !isempty(strip(line))]
end

summaries = Dict{Int,Vector{Dict{String,String}}}()
raws = Dict{Int,Vector{Dict{String,String}}}()
for count in COUNTS
    directory = joinpath(ROOT, "gpu7_paper_total1p2e9_count$count")
    summaries[count] = read_csv(joinpath(directory, "summary.csv"))
    raws[count] = read_csv(joinpath(directory, "raw.csv"))
end

all(row["status"] in ("PASS", "TIMEOUT", "SKIPPED_TIMEOUT_RISK") for rows in values(raws) for row in rows) || error("unexpected raw status")
meta = raws[100][1]
by_key = Dict((count, row["strategy"]) => row for count in COUNTS for row in summaries[count])

cell(row) = row["status"] == "TIMEOUT" ? ">15 (timeout)" :
            row["status"] == "SKIPPED_TIMEOUT_RISK" ? "not run (launch-count risk)" :
            @sprintf("%.4g ± %.3g", parse(Float64, row["mean_ms"])/1000, parse(Float64, row["std_ms"])/1000)
winners = [by_key[(count, "Grid-wise")]["fastest_strategy"] for count in COUNTS]
transitions = [(COUNTS[i-1], COUNTS[i], winners[i-1], winners[i]) for i in 2:length(COUNTS) if winners[i] != winners[i-1]]

mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    println(io, "# Figure 3 reproduction with four GPU hierarchy levels\n")
    println(io, "## Environment\n")
    println(io, "- GPU: `$(meta["gpu_name"])` (`$(meta["gpu_uuid"])`)")
    println(io, "- NVIDIA driver: `$(meta["driver_version"])`")
    println(io, "- CUDA build toolkit: `$(meta["cuda_toolkit"])`; CUDA.jl runtime: `$(meta["cuda_runtime"])`")
    println(io, "- Julia: `$(meta["julia_version"])`; Git commit: `$(meta["git_commit"])`\n")
    println(io, "## Method\n")
    println(io, "Following Figure 3, total SOC dimension is fixed at **1.2 × 10^9**. For each cone count `m`, every cone has full dimension `ceil(1.2 × 10^9 / m)`. We compare grid-wise (`few`), block-wise (`moderate`), warp-wise (`sufficient`), and thread-wise (`massive`) projection. Values are mean seconds ± one sample standard deviation over 10 independently seeded inputs. A projection exceeding 15 seconds is recorded as a timeout. The sweep reaches 120 million cones and dimension 10. Grid-wise is not launched beyond 10^6 cones because it would require 10–120 million sequential cuBLAS calls before the current in-process timeout could observe completion.\n")
    println(io, "## Results\n")
    println(io, "| Cones | Dimension per cone | Grid-wise (s) | Block-wise (s) | Warp-wise (s) | Thread-wise (s) | Fastest |")
    println(io, "|---:|---:|---:|---:|---:|---:|:---|")
    for count in COUNTS
        dimension = parse(Int, by_key[(count,"Grid-wise")]["cone_dimension"])
        values = [cell(by_key[(count,s)]) for s in STRATEGIES]
        println(io, "| $count | $dimension | $(join(values, " | ")) | $(by_key[(count,"Grid-wise")]["fastest_strategy"]) |")
    end
    println(io, "\nAll completed trials passed the closed-form SOC check. Thread-wise timed out for 10 extremely high-dimensional cones, grid-wise timed out for 10^6 cones, and grid-wise was explicitly not launched above 10^6 cones because of sequential-launch timeout risk.\n")
    println(io, "## Interpretation\n")
    println(io, "The fastest method changes from **$(first(winners))** at $(first(COUNTS)) cones to **$(last(winners))** at $(last(COUNTS)) cones. Observed transitions: " * join(["between $a and $b cones ($x → $y)" for (a,b,x,y) in transitions], "; ") * ". Grid-wise dominates a few extremely high-dimensional cones, warp-wise dominates from 10^3 through 10^7 cones, and thread-wise dominates at 10^8 and 1.2×10^8 cones (dimensions 12 and 10).\n")
    println(io, "## Draft manuscript text\n")
    println(io, "We repeat the fixed-total-dimension experiment with the warp-wise implementation included. The total SOC dimension is 1.2 × 10^9, while the cone count ranges from 10 to 1.2 × 10^8 and each cone dimension is adjusted inversely from 1.2 × 10^8 to 10. Grid-wise projection is fastest for 10 and 100 very high-dimensional cones. Warp-wise projection is fastest from 10^3 through 10^7 cones, while thread-wise projection is fastest at 10^8 and 1.2 × 10^8 cones. Thread-wise exceeds 15 seconds for 10 cones; grid-wise exceeds 15 seconds at 10^6 cones and is not launched at higher counts because of sequential-launch timeout risk.")
end

mkpath(dirname(FIGURE))
open(FIGURE, "w") do io
    println(io, raw"\documentclass[tikz,border=2pt]{standalone}")
    println(io, raw"\usepackage{pgfplots}")
    println(io, raw"\usepgfplotslibrary{fillbetween}")
    println(io, raw"\pgfplotsset{compat=1.17}")
    println(io, raw"\begin{document}\begin{tikzpicture}")
    println(io, raw"\begin{axis}[width=10cm,height=6.3cm,xmode=log,ymode=log,xlabel={Number of cones},ylabel={Projection time (s)},grid=major,legend pos=north west,ymax=25]")
    for strategy in STRATEGIES
        complete = [(count, by_key[(count,strategy)]) for count in COUNTS if by_key[(count,strategy)]["status"] == "COMPLETE"]
        center = join(["($count,$(parse(Float64,row["mean_ms"])/1000))" for (count,row) in complete], " ")
        upper = join(["($count,$((parse(Float64,row["mean_ms"])+parse(Float64,row["std_ms"]))/1000))" for (count,row) in complete], " ")
        lower = join(["($count,$(max((parse(Float64,row["mean_ms"])-parse(Float64,row["std_ms"]))/1000,eps())))" for (count,row) in complete], " ")
        key = replace(lowercase(strategy), "-wise"=>"")
        println(io, "\\addplot[name path=$(key)u,draw=none] coordinates {$upper};")
        println(io, "\\addplot[name path=$(key)l,draw=none] coordinates {$lower};")
        println(io, "\\addplot[$(COLORS[strategy])!15] fill between[of=$(key)u and $(key)l];")
        println(io, "\\addplot[$(COLORS[strategy]),thick,mark=*] coordinates {$center};")
        println(io, "\\addlegendentry{$strategy}")
        timeout = [count for count in COUNTS if by_key[(count,strategy)]["status"] == "TIMEOUT"]
        isempty(timeout) || println(io, "\\addplot[$(COLORS[strategy]),only marks,mark=triangle*] coordinates {$(join(["($c,15)" for c in timeout]," "))};")
    end
    println(io, raw"\end{axis}\end{tikzpicture}\end{document}")
end

@info "Paper count-sweep report generated" output=OUTPUT figure=FIGURE
