"""
Helpers for R3.5 experiments 1 and 3.

The profiled solve and the timed solve must be separate runs. Profile kernels
perform extra synchronization, reset, and device-to-host copies.
"""

using CUDA
using PDCS

const PDCS_GPU = PDCS.PDCS_GPU

"""Run `solve!()` once with diagnostic projection kernels enabled."""
function run_projection_work_profile(solve!::Function)
    PDCS_GPU.enable_projection_work_profile!(scope = :pdhg_iterations)
    result = nothing
    try
        result = solve!()
    finally
        work = PDCS_GPU.disable_projection_work_profile!()
    end
    return (; result, work)
end

"""Rows matching the experiment-1 baseline/diagonal comparison table."""
function projection_work_comparison_rows(baseline, diagonal)
    baseline_rows = Dict(
        (row.cone_type, row.cone_dimension) => row for row in baseline.by_cone
        if !row.diagonal
    )
    diagonal_rows = Dict(
        (row.cone_type, row.cone_dimension) => row for row in diagonal.by_cone
        if row.diagonal
    )
    keys_union = sort!(
        collect(union(keys(baseline_rows), keys(diagonal_rows)));
        by = key -> (string(key[1]), key[2]),
    )
    return [
        begin
            base = get(baseline_rows, key, nothing)
            diag = get(diagonal_rows, key, nothing)
            base_average = isnothing(base) ? 0.0 :
                base.average_vector_vector_reductions
            diag_average = isnothing(diag) ? 0.0 :
                diag.average_vector_vector_reductions
            (
                cone_type = key[1],
                cone_dimension = key[2],
                average_baseline = base_average,
                average_diagonal = diag_average,
                average_extra = diag_average - base_average,
            )
        end for key in keys_union
    ]
end

"""Summary fields used by the adaptive-vs-fixed experiment-3 table."""
function projection_work_run_row(application, rule, iterations, profile_summary)
    return (
        application = application,
        rule = rule,
        pdhg_iterations = iterations,
        average_vector_vector_reductions =
            profile_summary.average_vector_vector_reductions,
        vector_vector_reductions = profile_summary.vector_vector_reductions,
        projection_events = profile_summary.projection_events,
    )
end
