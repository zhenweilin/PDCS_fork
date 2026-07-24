#!/usr/bin/env julia

function option(name, default)
    prefix = "--$(name)="
    for (i, value) in pairs(ARGS)
        startswith(value, prefix) && return value[length(prefix) + 1:end]
        value == "--$(name)" && i < length(ARGS) && return ARGS[i + 1]
    end
    return default
end

dimension_report = abspath(option("dimension-report", "rebuttal_plan/cone_dimension_results.md"))
count_report = abspath(option("count-report", "rebuttal_plan/cone_count_results.md"))
output = abspath(option("output", "rebuttal_plan/cone_projectioin_results.md"))

isfile(dimension_report) || error("missing dimension report: $dimension_report")
isfile(count_report) || error("missing count report: $count_report")

mkpath(dirname(output))
open(output, "w") do io
    println(io, "# SOC projection experiments for reviewer R1-2\n")
    println(io, "This master report preserves both complementary experiments:\n")
    println(io, "1. Fixed cone counts (3 and 100), varying individual cone dimension.")
    println(io, "2. Fixed total dimension (1.2 × 10^9), varying the number of cones.\n")
    println(io, "The component reports are generated directly from their raw CSV files.\n")
    println(io, "---\n")
    write(io, read(dimension_report, String))
    println(io, "\n\n---\n")
    write(io, read(count_report, String))
    println(io)
end

@info "Master SOC rebuttal report assembled" output
