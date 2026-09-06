#!/usr/bin/env julia

using Statistics
using TOML

const SCALES = [
    (m = 10_000, n = 100_000),
    (m = 70_000, n = 700_000),
    (m = 400_000, n = 7_000_000),
    (m = 700_000, n = 7_000_000),
    (m = 750_000, n = 7_500_000),
]
const SOLVERS = ("cupdcs", "cuscs", "cuclarabel")
const REPLICATES = 5
const REQUIRED_TOLERANCE = 1.0e-6

function parse_cli(args)
    values = Dict{String,String}()
    index = 1
    while index <= length(args)
        index == length(args) && error("missing value for $(args[index])")
        startswith(args[index], "--") || error("unexpected argument: $(args[index])")
        values[args[index]] = args[index + 1]
        index += 2
    end
    for solver in SOLVERS
        haskey(values, "--$solver") || error("missing --$solver RESULT_DIRECTORY")
    end
    output = get(values, "--output", joinpath(@__DIR__, "results", "verified"))
    require_complete = lowercase(get(values, "--require-complete", "false")) in
        ("1", "true", "yes")
    return (;
        directories = Dict(solver => abspath(values["--$solver"]) for solver in SOLVERS),
        output = abspath(output),
        require_complete,
    )
end

instance_id(scale, replicate) =
    "table5-m$(scale.m)-n$(scale.n)-r$(lpad(replicate, 2, '0'))"

function load_case(directory, solver, scale, replicate)
    id = instance_id(scale, replicate)
    path = joinpath(directory, "$id.toml")
    isfile(path) || return nothing
    result = TOML.parsefile(path)
    get(result, "instance_id", "") == id || error("instance ID mismatch in $path")
    get(result, "solver", "") == solver || error("solver mismatch in $path")
    get(result, "m", 0) == scale.m || error("m mismatch in $path")
    get(result, "n", 0) == scale.n || error("n mismatch in $path")
    get(result, "replicate", 0) == replicate || error("replicate mismatch in $path")
    occursin("H100", get(result, "gpu_name", "")) ||
        error("case was not recorded on an H100: $path")
    get(result, "run_status", "") in ("passed", "failed") ||
        error("invalid run_status in $path")
    get(result, "tolerance", NaN) == REQUIRED_TOLERANCE ||
        error("solver tolerance is not exactly 1e-6 in $path")
    if result["run_status"] == "passed"
        # A passed result must have been judged with the exact independent
        # threshold.  A failed legacy result may record a looser checker
        # threshold: failure under that threshold is also failure under 1e-6,
        # while its solver tolerance is still required to be exactly 1e-6.
        get(result, "validation_tolerance", NaN) == REQUIRED_TOLERANCE ||
            error("validation tolerance is not exactly 1e-6 in passed case $path")
        violation = get(result, "normalized_violation", NaN)
        isfinite(violation) && violation <= REQUIRED_TOLERANCE ||
            error("passed case exceeds the 1e-6 validation tolerance in $path")
        get(result, "validation_accepted", false) === true ||
            error("passed case was not independently validated in $path")
        if solver == "cupdcs"
            kkt = get(result, "solver_relative_kkt_max", NaN)
            isfinite(kkt) && kkt <= REQUIRED_TOLERANCE ||
                error("passed cuPDCS case exceeds its 1e-6 KKT tolerance in $path")
            get(result, "solver_tolerance_accepted", false) === true ||
                error("passed cuPDCS case was not accepted by its own tolerance check in $path")
        end
    end
    return result
end

function finite_mean(values)
    usable = filter(isfinite, Float64.(values))
    return isempty(usable) ? NaN : mean(usable)
end

function solver_summary(solver, directory)
    rows = Vector{Dict{String,Any}}()
    by_scale = Vector{Vector{Union{Nothing,Dict{String,Any}}}}()
    for scale in SCALES
        cases = Union{Nothing,Dict{String,Any}}[]
        for replicate in 1:REPLICATES
            result = load_case(directory, solver, scale, replicate)
            push!(cases, result)
            result === nothing && continue
            push!(rows, Dict{String,Any}(
                "solver" => solver,
                "instance_id" => result["instance_id"],
                "m" => scale.m,
                "n" => scale.n,
                "replicate" => replicate,
                "run_status" => result["run_status"],
                "termination_status" => get(result, "termination_status", ""),
                "gpu_name" => result["gpu_name"],
                "solve_seconds" => get(result, "solve_seconds", NaN),
                "objective_value" => get(result, "objective_value", NaN),
                "normalized_violation" => get(result, "normalized_violation", NaN),
                "tolerance" => get(result, "tolerance", NaN),
                "validation_tolerance" => get(result, "validation_tolerance", NaN),
                "solver_relative_kkt_max" => get(
                    result, "solver_relative_kkt_max", NaN
                ),
                "error" => get(result, "error", ""),
            ))
        end
        push!(by_scale, cases)
    end

    failure_index = findfirst(cases -> any(
        result !== nothing && result["run_status"] == "failed" for result in cases
    ), by_scale)
    expected_last = solver == "cupdcs" || failure_index === nothing ?
        length(SCALES) : failure_index
    violations = String[]
    scale_rows = Vector{Dict{String,Any}}()
    for (index, scale) in enumerate(SCALES)
        cases = by_scale[index]
        present = filter(!isnothing, cases)
        passed = count(result -> result["run_status"] == "passed", present)
        failed = count(result -> result["run_status"] == "failed", present)
        missing = REPLICATES - length(present)
        if index <= expected_last
            missing == 0 || push!(violations,
                "m$(scale.m)-n$(scale.n) has $missing missing replicate(s)")
        elseif !isempty(present)
            push!(violations, "results exist after the required early-stop scale")
        end
        if failure_index !== nothing && index < failure_index && failed != 0
            push!(violations, "failure appears before computed failure scale")
        end
        push!(scale_rows, Dict{String,Any}(
            "m" => scale.m,
            "n" => scale.n,
            "passed" => passed,
            "failed" => failed,
            "missing" => missing,
            "mean_solve_seconds" => finite_mean(
                get(result, "solve_seconds", NaN) for result in present
            ),
        ))
    end

    summary_path = joinpath(directory, "campaign_summary.txt")
    campaign_finished = isfile(summary_path)
    campaign_finished || push!(violations, "campaign_summary.txt is missing")
    if campaign_finished
        text = read(summary_path, String)
        occursin("gpu=NVIDIA H100", text) ||
            push!(violations, "campaign summary does not identify an H100")
        occursin("replicates=5", text) ||
            push!(violations, "campaign summary does not record five replicates")
        if solver != "cupdcs" && failure_index !== nothing
            failed_scale = "m$(SCALES[failure_index].m)-n$(SCALES[failure_index].n)"
            occursin("stopped_after_scale=$failed_scale", text) ||
                push!(violations, "campaign summary has the wrong early-stop scale")
        end
    end

    return (
        summary = Dict{String,Any}(
            "solver" => solver,
            "directory" => directory,
            "campaign_finished" => campaign_finished,
            "first_failed_scale" => failure_index === nothing ? "" :
                "m$(SCALES[failure_index].m)-n$(SCALES[failure_index].n)",
            "expected_scales" => expected_last,
            "case_count" => length(rows),
            "complete" => isempty(violations),
            "violations" => violations,
            "scales" => scale_rows,
        ),
        rows,
    )
end

function write_tsv(path, rows)
    columns = (
        "solver", "instance_id", "m", "n", "replicate", "run_status",
        "termination_status", "gpu_name", "solve_seconds", "objective_value",
        "normalized_violation", "tolerance", "validation_tolerance",
        "solver_relative_kkt_max", "error",
    )
    open(path, "w") do stream
        println(stream, join(columns, '\t'))
        for row in rows
            values = map(columns) do column
                replace(string(get(row, column, "")), '\t' => ' ', '\n' => ' ')
            end
            println(stream, join(values, '\t'))
        end
    end
end

function main()
    options = parse_cli(ARGS)
    mkpath(options.output)
    summaries = Vector{Dict{String,Any}}()
    rows = Vector{Dict{String,Any}}()
    for solver in SOLVERS
        collected = solver_summary(solver, options.directories[solver])
        push!(summaries, collected.summary)
        append!(rows, collected.rows)
    end
    complete = all(summary["complete"] for summary in summaries)
    open(joinpath(options.output, "verification.toml"), "w") do stream
        TOML.print(stream, Dict(
            "schema_version" => 1,
            "complete" => complete,
            "solvers" => summaries,
        ); sorted = true)
    end
    write_tsv(joinpath(options.output, "cases.tsv"), rows)
    for summary in summaries
        println(
            summary["solver"],
            ": complete=", summary["complete"],
            " cases=", summary["case_count"],
            " first_failed_scale=", repr(summary["first_failed_scale"]),
        )
        for violation in summary["violations"]
            println("  - ", violation)
        end
    end
    println("CAMPAIGN_VERIFICATION_COMPLETE=$complete")
    options.require_complete && !complete && exit(1)
end

main()
