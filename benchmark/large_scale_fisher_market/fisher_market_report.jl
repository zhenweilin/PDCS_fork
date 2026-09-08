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

function comparison_value(row, name)
    row.result === nothing && return ""
    haskey(row.result, name) && return row.result[name]
    # Legacy cuPDCS result files already stored these three values with the
    # same original-scale definitions under solver_* names.  Never apply this
    # fallback to SCS or Clarabel because their native scalings differ.
    row.solver == "cupdcs" || return ""
    legacy_name = get(
        Dict(
            "comparison_primal_infeasibility_rel" =>
                "solver_primal_residual",
            "comparison_dual_infeasibility_rel" =>
                "solver_dual_residual",
            "comparison_primal_dual_gap_rel" =>
                "solver_relative_gap",
        ),
        name,
        "",
    )
    isempty(legacy_name) && return ""
    return get(row.result, legacy_name, "")
end

function same_scale(first, second)
    return first["m"] == second["m"] &&
           first["n"] == second["n"] &&
           first["density"] == second["density"]
end

function result_is_memory_failure(result)
    result === nothing && return false
    text = lowercase(
        string(
            get(result, "termination_status", ""),
            " ",
            get(result, "error", ""),
        ),
    )
    return occursin("out_of_memory", text) ||
           occursin("outofmemoryerror", text) ||
           occursin("out of memory", text) ||
           occursin("cudaerrormemoryallocation", text) ||
           occursin("cumemalloc", text) ||
           occursin("failed to allocate", text)
end

function skipped_after_memory_failure(row, rows)
    row.result === nothing || return false
    row.solver in ("scs_gpu", "cuclarabel") || return false
    return any(rows) do previous
        previous.solver == row.solver || return false
        same_scale(previous.entry, row.entry) || return false
        previous.entry["replicate"] < row.entry["replicate"] || return false
        previous.result === nothing && return false
        return result_is_memory_failure(previous.result)
    end
end

function missing_status(row, rows)
    return skipped_after_memory_failure(row, rows) ?
           "SKIPPED_AFTER_MEMORY_FAILURE" : row.path
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
              count(row -> skipped_after_memory_failure(row, rows), rows) : 0
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
        "The `comparison_*` errors use the same original-scale infinity-norm " *
        "normalization as cuPDCS: each absolute residual is divided by " *
        "`1 + max` of its corresponding unscaled problem quantities. The " *
        "three solvers' native residual fields are retained in result.toml " *
        "but are not used for cross-solver comparison.",
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
        "| Solver | Status | Iterations | Objective | Primal infeas. rel. " *
        "| Dual infeas. rel. | P-D gap rel. | EXP log abs. " *
        "| EXP rel. upper bound | Wall time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    )
    for row in smoke_rows
        result = row.result
        if result === nothing
            push!(
                lines,
                "| $(row.solver) | $(row.path) |  |  |  |  |  |  |  |  |",
            )
            continue
        end
        push!(
            lines,
            "| $(row.solver) " *
            "| $(get(result, "termination_status", "")) " *
            "| $(get(result, "iterations", "")) " *
            "| $(format_number(get(result, "objective_value", ""))) " *
            "| $(format_number(comparison_value(row, "comparison_primal_infeasibility_rel"))) " *
            "| $(format_number(comparison_value(row, "comparison_dual_infeasibility_rel"))) " *
            "| $(format_number(comparison_value(row, "comparison_primal_dual_gap_rel"))) " *
            "| $(format_number(get(result, "exponential_log_violation", ""))) " *
            "| $(format_number(get(result, "exponential_cone_relative_violation_upper_bound", ""))) " *
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
        "Primal infeas. rel. | Dual infeas. rel. | P-D gap rel. | " *
        "Generation (s) | Setup (s) | Solve wall (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    )
    for row in formal_rows
        result = row.result
        if result === nothing
            push!(
                lines,
                "| $(row.entry["id"]) | $(row.solver) " *
                "| $(missing_status(row, formal_rows)) " *
                "|  |  |  |  |  |  |  |  |",
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
            "| $(format_number(comparison_value(row, "comparison_primal_infeasibility_rel"))) " *
            "| $(format_number(comparison_value(row, "comparison_dual_infeasibility_rel"))) " *
            "| $(format_number(comparison_value(row, "comparison_primal_dual_gap_rel"))) " *
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
