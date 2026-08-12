#!/usr/bin/env julia

using Printf
using TOML

const SOLVERS = ("cupdcs", "scs_gpu", "cuclarabel")

function parse_cli(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        index == length(arguments) &&
            error("missing value for $(arguments[index])")
        options[arguments[index]] = arguments[index + 1]
        index += 2
    end
    for option in ("--manifest", "--result-root", "--output")
        haskey(options, option) || error("missing $option")
    end
    return (
        manifest = abspath(options["--manifest"]),
        result_root = abspath(options["--result-root"]),
        output = abspath(options["--output"]),
    )
end

function load_result(result_root, mode, solver, instance_id)
    case_dir = joinpath(result_root, mode, solver, instance_id)
    done_path = joinpath(case_dir, "DONE")
    isfile(done_path) || return nothing, "PENDING"
    attempt_name = strip(read(done_path, String))
    result_path = joinpath(case_dir, attempt_name, "result.toml")
    isfile(result_path) || return nothing, "BROKEN_DONE"
    return TOML.parsefile(result_path), result_path
end

function result_rows(manifest, result_root, mode, group)
    rows = NamedTuple[]
    for entry in manifest[group], solver in SOLVERS
        result, path = load_result(
            result_root,
            mode,
            solver,
            entry["id"],
        )
        push!(rows, (
            entry = entry,
            solver = solver,
            result = result,
            path = path,
        ))
    end
    return rows
end

function format_number(value; digits = 6)
    value isa Number || return ""
    isfinite(value) || return string(value)
    return @sprintf("%.*g", digits, value)
end

function same_scale(first, second)
    return first["m"] == second["m"] &&
           first["n"] == second["n"] &&
           first["density"] == second["density"]
end

function skipped_after_scale_failure(row, rows)
    row.result === nothing || return false
    row.solver in ("scs_gpu", "cuclarabel") || return false
    return any(rows) do previous
        previous.solver == row.solver || return false
        same_scale(previous.entry, row.entry) || return false
        previous.entry["replicate"] < row.entry["replicate"] || return false
        previous.result === nothing && return false
        return get(previous.result, "termination_status", "") ∉
               ("OPTIMAL", "ALMOST_OPTIMAL")
    end
end

function missing_status(row, rows)
    return skipped_after_scale_failure(row, rows) ?
           "SKIPPED_AFTER_SCALE_FAILURE" : row.path
end

function digest_errors(rows)
    by_instance = Dict{String,Set{String}}()
    for row in rows
        row.result === nothing && continue
        digests = get!(
            by_instance,
            row.entry["id"],
            Set{String}(),
        )
        push!(
            digests,
            string(get(row.result, "numerical_digest", "MISSING")),
        )
    end
    return [
        "$instance_id: $(sort!(collect(digests)))"
        for (instance_id, digests) in by_instance
        if length(digests) != 1 || "MISSING" in digests
    ]
end

function append_coverage!(lines, label, rows; show_skipped = false)
    recorded = count(row -> row.result !== nothing, rows)
    optimal = count(
        row ->
            row.result !== nothing &&
            get(row.result, "termination_status", "") in
            ("OPTIMAL", "ALMOST_OPTIMAL"),
        rows,
    )
    skipped = show_skipped ?
              count(row -> skipped_after_scale_failure(row, rows), rows) : 0
    push!(
        lines,
        "- $label: $recorded/$(length(rows)) recorded, " *
        "$optimal/$(length(rows)) optimal or almost optimal" *
        (show_skipped ? ", $skipped skipped by policy." : "."),
    )
end

function write_report(manifest, smoke_rows, formal_rows, output)
    lines = String[
        "# Large-scale Fisher market report",
        "",
        "Instances are regenerated deterministically in memory and passed " *
        "directly to each solver source API. No JuMP model, CBF, JLD2, NPZ, " *
        "or matrix-data file is used.",
        "",
        "## Coverage",
        "",
    ]
    append_coverage!(lines, "Smoke", smoke_rows)
    append_coverage!(lines, "Formal", formal_rows; show_skipped = true)
    errors = digest_errors(vcat(smoke_rows, formal_rows))
    push!(
        lines,
        "- Cross-solver numerical digest check: " *
        (isempty(errors) ? "PASS." : "FAIL."),
        "",
        "## Small correctness case",
        "",
        "| Solver | Status | Iterations | Objective | Supply rel. residual " *
        "| Utility rel. residual | Exp-cone log violation | Wall time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    )
    for row in smoke_rows
        result = row.result
        if result === nothing
            push!(
                lines,
                "| $(row.solver) | $(row.path) |  |  |  |  |  |  |",
            )
            continue
        end
        push!(
            lines,
            "| $(row.solver) " *
            "| $(get(result, "termination_status", "")) " *
            "| $(get(result, "iterations", "")) " *
            "| $(format_number(get(result, "objective_value", ""))) " *
            "| $(format_number(get(result, "supply_rel_residual", ""))) " *
            "| $(format_number(get(result, "utility_rel_residual", ""))) " *
            "| $(format_number(get(result, "exponential_log_violation", ""))) " *
            "| $(format_number(get(result, "solve_wall_seconds", ""); digits=5)) |",
        )
    end

    complete_smoke = [
        row.result for row in smoke_rows if row.result !== nothing
    ]
    if length(complete_smoke) == length(SOLVERS)
        objectives = Float64[
            result["objective_value"] for result in complete_smoke
        ]
        spread = maximum(objectives) - minimum(objectives)
        relative_spread =
            spread / max(1.0, maximum(abs, objectives))
        push!(
            lines,
            "",
            "- Objective range: `$(format_number(spread))`.",
            "- Relative objective range: " *
            "`$(format_number(relative_spread))`.",
        )
    end

    push!(
        lines,
        "",
        "## Formal cases",
        "",
        "| Instance | Solver | Status | Iterations | Objective | " *
        "Generation (s) | Setup (s) | Solve wall (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    )
    for row in formal_rows
        result = row.result
        if result === nothing
            push!(
                lines,
                "| $(row.entry["id"]) | $(row.solver) " *
                "| $(missing_status(row, formal_rows)) " *
                "|  |  |  |  |  |",
            )
            continue
        end
        push!(
            lines,
            "| $(row.entry["id"]) " *
            "| $(row.solver) " *
            "| $(get(result, "termination_status", "")) " *
            "| $(get(result, "iterations", "")) " *
            "| $(format_number(get(result, "objective_value", ""))) " *
            "| $(format_number(get(result, "generation_seconds", ""); digits=5)) " *
            "| $(format_number(get(result, "setup_seconds", ""); digits=5)) " *
            "| $(format_number(get(result, "solve_wall_seconds", ""); digits=5)) |",
        )
    end

    if !isempty(errors)
        push!(lines, "", "## Digest errors", "")
        append!(lines, ["- $error" for error in errors])
    end
    push!(
        lines,
        "",
        "## Manifest",
        "",
        "- Julia: $(manifest["julia_version"])",
        "- Seeds: $(join(manifest["seeds"], ", "))",
        "- Replicates per formal size: $(manifest["replicates"])",
        "- Storage policy: $(manifest["storage"])",
        "- Modeling interface: $(manifest["model_interface"])",
        "",
    )
    mkpath(dirname(output))
    open(output, "w") do stream
        write(stream, join(lines, "\n"))
    end
end

function main()
    options = parse_cli(ARGS)
    manifest = TOML.parsefile(options.manifest)
    smoke_rows = result_rows(
        manifest,
        options.result_root,
        "smoke",
        "smoke_instances",
    )
    formal_rows = result_rows(
        manifest,
        options.result_root,
        "formal",
        "instances",
    )
    write_report(manifest, smoke_rows, formal_rows, options.output)
    println(
        "FISHER_REPORT_WRITTEN path=$(options.output) " *
        "smoke_rows=$(length(smoke_rows)) formal_rows=$(length(formal_rows))",
    )
end

main()
