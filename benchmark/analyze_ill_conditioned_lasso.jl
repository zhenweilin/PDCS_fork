#!/usr/bin/env julia

using Dates
using Statistics

function option(name,default)
    for (i,a) in pairs(ARGS)
        a=="--$name" && i<length(ARGS) && return ARGS[i+1]
        startswith(a,"--$name=") && return split(a,'=';limit=2)[2]
    end
    default
end
const ROOT=abspath(option("root","benchmark/results/rebuttal/ill_conditioned_lasso"))
const OUTPUT=abspath(option("output",joinpath(ROOT,"ill_conditioned_lasso_report.md")))
function readrows(path)
    isfile(path) || error("missing $path")
    lines=readlines(path); names=Symbol.(split(first(lines),','))
    records=NamedTuple[]
    skipped=0
    for line in lines[2:end]
        isempty(line) && continue
        fields=split(line,',')
        # Result messages are sanitized by the current runner.  Preserve old
        # development logs without aborting analysis if a historical exception
        # contained a newline and split one CSV record across two physical lines.
        if length(fields) != length(names)
            skipped += 1
            continue
        end
        push!(records,NamedTuple{Tuple(names)}(Tuple(fields)))
    end
    records,skipped
end
records,skipped_rows=readrows(joinpath(ROOT,"seed_level_results.csv"))
# Keep only completed solver calls in the numeric table.  Earlier development
# attempts (e.g. an unsupported optimizer attribute) remain in the raw CSV,
# but are not numerical outcomes and must not be reported as such.
completed_statuses=Set(["solved_verified","solver_claimed_solved_but_inaccurate","timeout"])
completed_records=filter(row -> row.status in completed_statuses,records)
development_records=length(records)-length(completed_records)
groups=Dict{Tuple{String,String,String},Vector{Any}}()
for row in completed_records
    push!(get!(groups,(row.solver,row.K,row.tolerance),Any[]),row)
end
open(OUTPUT,"w") do io
    println(io,"# Ill-Conditioned Lasso Experiment Report\n")
    println(io,"Generated: ",Dates.format(now(Dates.UTC),dateformat"yyyy-mm-ddTHH:MM:SSZ"),". The statistical unit is the workload seed; this report does not pool different seeds or solver attempts as independent iterations.\n")
    if skipped_rows > 0 || development_records > 0
        println(io,"Historical log note: $skipped_rows malformed physical CSV lines and $development_records non-completed development/infrastructure records were excluded from the numerical-outcome table; the raw file is retained unchanged.\n")
    end
    println(io,"## Verified outcomes\n")
    println(io,"| Solver | K | tolerance | attempts | verified | status counts | median wall seconds | median normalized KKT |")
    println(io,"|:--|--:|--:|--:|--:|:--|--:|--:|")
    for ((solver,K,tol),rows) in sort(collect(groups);by=first)
        status=Dict{String,Int}(); for r in rows status[r.status]=get(status,r.status,0)+1 end
        wall=[tryparse(Float64,r.solve_seconds) for r in rows]; wall=filter(!isnothing,wall)
        kkt=[tryparse(Float64,r.normalized_kkt) for r in rows]; kkt=filter(x->!isnothing(x)&&isfinite(x),kkt)
        verified=get(status,"solved_verified",0)
        fmt(x)=isempty(x) ? "NA" : string(round(median(x);sigdigits=5))
        count_text=join(("$k=$v" for (k,v) in sort(collect(status))),"; ")
        println(io,"| $solver | $K | $tol | $(length(rows)) | $verified | $count_text | $(fmt(wall)) | $(fmt(kkt)) |")
    end
    println(io,"\n## Interpretation\n")
    println(io,"Only rows labelled `solved_verified` satisfy the independent original-Lasso KKT check. Rows labelled `input_or_conversion_failure` indicate unavailable external packages or model conversion; `not_attempted_hardware_limit` indicates CUDA was unavailable. Neither is an algorithmic failure. MOSEK is intentionally limited by default to the smoke and medium-size reference profiles as a correctness reference; it is excluded from the pilot and production-medium profiles. This run must not be used to claim a solver ranking unless all planned solver/seed/K cells are present and verified.\n")
    println(io,"## Required artifacts\n")
    println(io,"- `instance_manifest.csv` records the exact active condition number, KKT construction, and SHA-256 hashes.\n- `seed_level_results.csv` contains raw solver outcomes and verifier fields.\n")
end
println("ILL_CONDITIONED_LASSO_REPORT_COMPLETE output=$OUTPUT")
