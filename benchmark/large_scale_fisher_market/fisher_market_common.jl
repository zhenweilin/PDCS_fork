module FisherMarketCommon

using LinearAlgebra
using Random
using SHA
using SparseArrays

export build_pdcs_formulation
export build_compact_standard_formulation
export build_standard_formulation
export empty_quadratic
export generate_instance
export independent_primal_metrics
export standard_form_relative_errors

const INDEX_TYPE = Int32

function _update_tag!(context, tag::AbstractString)
    SHA.update!(context, Vector{UInt8}(codeunits(tag)))
    SHA.update!(context, UInt8[0x00])
    return context
end

function _update_array!(context, tag::AbstractString, values::AbstractVector)
    _update_tag!(context, tag)
    SHA.update!(context, reinterpret(UInt8, [Int64(length(values))]))
    isempty(values) || SHA.update!(context, reinterpret(UInt8, values))
    return context
end

function _numerical_digest(m, n, density, seed, w, utility)
    context = SHA.SHA256_CTX()
    _update_array!(context, "dimensions", Int64[m, n])
    _update_array!(context, "density", Float64[density])
    _update_array!(context, "seed", Int64[seed])
    _update_array!(context, "weights", w)
    _update_array!(context, "utility.indices", Int64.(utility.nzind))
    _update_array!(context, "utility.values", utility.nzval)
    _update_array!(context, "supply", Float64[0.25 * m])
    return bytes2hex(SHA.digest!(context))
end

"""
Generate one Fisher-market instance entirely in memory.

The allocation variables use buyer-major ordering:
`allocation_index = (buyer - 1) * n + good`.
"""
function generate_instance(
    m::Integer,
    n::Integer,
    density::Real,
    seed::Integer,
)
    m > 0 || throw(ArgumentError("m must be positive"))
    n > 0 || throw(ArgumentError("n must be positive"))
    0.0 < density <= 1.0 ||
        throw(ArgumentError("density must be in (0, 1]"))
    allocation_count = Base.checked_mul(Int64(m), Int64(n))

    rng = MersenneTwister(seed)
    weights = rand(rng, Float64, m)
    utility = sprand(rng, allocation_count, Float64(density))

    # The exponential-cone model needs every buyer to have positive aggregate
    # utility. The formal densities make an empty row overwhelmingly unlikely;
    # fail loudly instead of silently changing a random instance.
    buyers_with_utility = falses(m)
    goods_with_utility = falses(n)
    for cursor in eachindex(utility.nzind)
        utility.nzval[cursor] > 0.0 || continue
        allocation_index = utility.nzind[cursor]
        buyer = fld(allocation_index - 1, n) + 1
        good = mod(allocation_index - 1, n) + 1
        buyers_with_utility[buyer] = true
        goods_with_utility[good] = true
    end
    all(buyers_with_utility) ||
        error("seed $seed generated a buyer with no positive utility")
    all(goods_with_utility) ||
        error("seed $seed generated a good with no positive valuation")

    supply = 0.25 * m
    digest = _numerical_digest(
        m,
        n,
        Float64(density),
        seed,
        weights,
        utility,
    )
    summary = (
        m = Int64(m),
        n = Int64(n),
        density = Float64(density),
        seed = Int64(seed),
        allocation_count = allocation_count,
        utility_nnz = Int64(nnz(utility)),
        supply_per_good = supply,
        numerical_digest = digest,
    )
    return (
        weights = weights,
        utility = utility,
        supply = supply,
        summary = summary,
    )
end

function _checked_index(value::Integer, label::AbstractString)
    value <= typemax(INDEX_TYPE) ||
        error("$label=$value exceeds $(INDEX_TYPE)")
    value >= 0 || error("$label must be nonnegative")
    return INDEX_TYPE(value)
end

"""
Construct the canonical positive-sign matrix `G` for cuPDCS:

    G * x - h ∈ Zero(n+m) × ExpCone^m,

with nonnegative allocations represented by variable lower bounds.
"""
function build_pdcs_formulation(instance)
    return _build_formulation(instance; explicit_nonnegative_rows = false)
end

"""
Construct the standard conic form used by SCS and cuClarabel:

    A * x + s = b,  s ∈ Zero(n+m) × Nonnegative(m*n) × ExpCone^m.

The returned `A` and `b` already have the signs required by those APIs, so no
large negated matrix copy is created.
"""
function build_standard_formulation(instance)
    positive = _build_formulation(
        instance;
        explicit_nonnegative_rows = true,
    )
    positive.A.nzval .*= -1.0
    positive.b .*= -1.0
    return positive
end

"""
Construct the standard conic form after removing allocation variables whose
valuation is zero.  The returned allocation indices retain the original
buyer-major positions, so primal feasibility and objectives can be checked
against the unmodified Fisher instance without expanding a dense `m*n`
vector.

With equality supply constraints this reduction is equivalent only when every
good has at least one strictly positive valuation.  Enforce that condition
here instead of silently changing the original problem.
"""
function build_compact_standard_formulation(instance)
    m = Int(instance.summary.m)
    n = Int(instance.summary.n)
    original_allocation_count = Int(instance.summary.allocation_count)
    utility = instance.utility
    all(value -> isfinite(value) && value >= 0.0, utility.nzval) ||
        error("compact Fisher formulation requires finite nonnegative valuations")

    allocation_indices, allocation_values = if all(>(0.0), utility.nzval)
        (utility.nzind, utility.nzval)
    else
        positive = findall(>(0.0), utility.nzval)
        (utility.nzind[positive], utility.nzval[positive])
    end
    allocation_count = length(allocation_indices)
    allocation_count > 0 ||
        error("compact Fisher formulation has no positive valuations")

    buyers_with_utility = falses(m)
    goods_with_utility = falses(n)
    for allocation_index in allocation_indices
        1 <= allocation_index <= original_allocation_count ||
            error("utility index $allocation_index is outside the allocation vector")
        buyer = fld(allocation_index - 1, n) + 1
        good = mod(allocation_index - 1, n) + 1
        buyers_with_utility[buyer] = true
        goods_with_utility[good] = true
    end
    all(buyers_with_utility) ||
        error("compact Fisher formulation found a buyer with no positive valuation")
    all(goods_with_utility) ||
        error(
            "compact Fisher formulation is not equivalent: " *
            "a good has no positive valuation",
        )

    variable_count = allocation_count + 2m
    nonnegative_rows = allocation_count
    row_count = n + m + nonnegative_rows + 3m
    matrix_nnz = 3allocation_count + 3m

    _checked_index(variable_count, "compact variable_count")
    _checked_index(row_count, "compact row_count")
    _checked_index(matrix_nnz + 1, "compact matrix_nnz+1")

    colptr = Vector{INDEX_TYPE}(undef, variable_count + 1)
    rowval = Vector{INDEX_TYPE}(undef, matrix_nnz)
    nzval = Vector{Float64}(undef, matrix_nnz)
    position = 1

    for compact_index in eachindex(allocation_indices)
        allocation_index = allocation_indices[compact_index]
        buyer = fld(allocation_index - 1, n) + 1
        good = mod(allocation_index - 1, n) + 1
        colptr[compact_index] = INDEX_TYPE(position)

        rowval[position] = INDEX_TYPE(good)
        nzval[position] = 1.0
        position += 1
        rowval[position] = INDEX_TYPE(n + buyer)
        nzval[position] = allocation_values[compact_index]
        position += 1
        rowval[position] = INDEX_TYPE(n + m + compact_index)
        nzval[position] = 1.0
        position += 1
    end

    exponential_offset = n + m + nonnegative_rows
    for buyer in 1:m
        t_column = allocation_count + 2buyer - 1
        z_column = t_column + 1
        cone_row = exponential_offset + 3(buyer - 1)

        colptr[t_column] = INDEX_TYPE(position)
        rowval[position] = INDEX_TYPE(cone_row + 1)
        nzval[position] = 1.0
        position += 1

        colptr[z_column] = INDEX_TYPE(position)
        rowval[position] = INDEX_TYPE(n + buyer)
        nzval[position] = -1.0
        position += 1
        rowval[position] = INDEX_TYPE(cone_row + 3)
        nzval[position] = 1.0
        position += 1
    end
    colptr[variable_count + 1] = INDEX_TYPE(position)
    position == matrix_nnz + 1 ||
        error(
            "compact matrix nnz mismatch: filled $(position - 1), " *
            "expected $matrix_nnz",
        )

    matrix = SparseMatrixCSC{Float64,INDEX_TYPE}(
        row_count,
        variable_count,
        colptr,
        rowval,
        nzval,
    )
    rhs = zeros(Float64, row_count)
    rhs[1:n] .= instance.supply
    for buyer in 1:m
        rhs[exponential_offset + 3(buyer - 1) + 2] = -1.0
    end
    objective = zeros(Float64, variable_count)
    for buyer in 1:m
        objective[allocation_count + 2buyer - 1] = -instance.weights[buyer]
    end

    # Convert `G*x - h in K` to the standard solver convention
    # `A*x + s = b, s in K` with `A = -G` and `b = -h`.
    matrix.nzval .*= -1.0
    rhs .*= -1.0
    return (
        A = matrix,
        b = rhs,
        c = objective,
        row_count = row_count,
        variable_count = variable_count,
        zero_count = n + m,
        nonnegative_count = allocation_count,
        exponential_count = m,
        allocation_indices = allocation_indices,
        allocation_values = allocation_values,
        original_allocation_count = original_allocation_count,
        modeled_allocation_count = allocation_count,
        removed_zero_valuation_count =
            original_allocation_count - allocation_count,
        formulation_variant = "positive_valuation_reduced_v1",
    )
end

function _build_formulation(instance; explicit_nonnegative_rows::Bool)
    m = Int(instance.summary.m)
    n = Int(instance.summary.n)
    allocation_count = Int(instance.summary.allocation_count)
    variable_count = allocation_count + 2m
    nonnegative_rows = explicit_nonnegative_rows ? allocation_count : 0
    row_count = n + m + nonnegative_rows + 3m
    matrix_nnz =
        allocation_count +
        nnz(instance.utility) +
        3m +
        nonnegative_rows

    _checked_index(variable_count, "variable_count")
    _checked_index(row_count, "row_count")
    _checked_index(matrix_nnz + 1, "matrix_nnz+1")

    colptr = Vector{INDEX_TYPE}(undef, variable_count + 1)
    rowval = Vector{INDEX_TYPE}(undef, matrix_nnz)
    nzval = Vector{Float64}(undef, matrix_nnz)

    utility_indices = instance.utility.nzind
    utility_values = instance.utility.nzval
    utility_cursor = firstindex(utility_indices)
    utility_stop = lastindex(utility_indices)
    position = 1

    for allocation_index in 1:allocation_count
        colptr[allocation_index] = INDEX_TYPE(position)
        buyer = fld(allocation_index - 1, n) + 1
        good = mod(allocation_index - 1, n) + 1

        rowval[position] = INDEX_TYPE(good)
        nzval[position] = 1.0
        position += 1

        if utility_cursor <= utility_stop &&
           utility_indices[utility_cursor] == allocation_index
            rowval[position] = INDEX_TYPE(n + buyer)
            nzval[position] = utility_values[utility_cursor]
            position += 1
            utility_cursor += 1
        end

        if explicit_nonnegative_rows
            rowval[position] = INDEX_TYPE(n + m + allocation_index)
            nzval[position] = 1.0
            position += 1
        end
    end
    utility_cursor == utility_stop + 1 ||
        error("not all utility entries were consumed")

    exponential_offset = n + m + nonnegative_rows
    for buyer in 1:m
        t_column = allocation_count + 2buyer - 1
        z_column = t_column + 1
        cone_row = exponential_offset + 3(buyer - 1)

        colptr[t_column] = INDEX_TYPE(position)
        rowval[position] = INDEX_TYPE(cone_row + 1)
        nzval[position] = 1.0
        position += 1

        colptr[z_column] = INDEX_TYPE(position)
        rowval[position] = INDEX_TYPE(n + buyer)
        nzval[position] = -1.0
        position += 1
        rowval[position] = INDEX_TYPE(cone_row + 3)
        nzval[position] = 1.0
        position += 1
    end
    colptr[variable_count + 1] = INDEX_TYPE(position)
    position == matrix_nnz + 1 ||
        error("matrix nnz mismatch: filled $(position - 1), expected $matrix_nnz")

    matrix = SparseMatrixCSC{Float64,INDEX_TYPE}(
        row_count,
        variable_count,
        colptr,
        rowval,
        nzval,
    )
    rhs = zeros(Float64, row_count)
    rhs[1:n] .= instance.supply
    for buyer in 1:m
        rhs[exponential_offset + 3(buyer - 1) + 2] = -1.0
    end
    objective = zeros(Float64, variable_count)
    for buyer in 1:m
        objective[allocation_count + 2buyer - 1] =
            -instance.weights[buyer]
    end

    if explicit_nonnegative_rows
        return (
            A = matrix,
            b = rhs,
            c = objective,
            row_count = row_count,
            variable_count = variable_count,
            zero_count = n + m,
            nonnegative_count = allocation_count,
            exponential_count = m,
        )
    end

    lower_bounds = fill(-Inf, variable_count)
    lower_bounds[1:allocation_count] .= 0.0
    upper_bounds = fill(Inf, variable_count)
    return (
        A = matrix,
        b = rhs,
        c = objective,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds,
        row_count = row_count,
        variable_count = variable_count,
        zero_count = n + m,
        nonnegative_count = 0,
        exponential_count = m,
    )
end

function empty_quadratic(variable_count::Integer)
    _checked_index(variable_count, "quadratic variable_count")
    return SparseMatrixCSC{Float64,INDEX_TYPE}(
        variable_count,
        variable_count,
        fill(INDEX_TYPE(1), variable_count + 1),
        INDEX_TYPE[],
        Float64[],
    )
end

"""
Compute solver-independent primal checks without storing the primal vector.

For `(t, 1, z) ∈ ExpCone`, the cone condition is checked in log space as
`t <= log(z)`, avoiding overflow in `exp(t)`.  The raw log violation is kept
as a diagnostic.  Acceptance uses a relative primal-feasibility upper bound
with the same global `1 + max(||h||∞, ||Gx||∞)` scale as the solver's
relative conic residual.  Decreasing `t` by the raw log violation produces a
feasible exponential-cone point, so that quantity is a conservative upper
bound on the cone distance rather than an unrelated absolute acceptance test.
"""
function independent_primal_metrics(
    primal::AbstractVector,
    instance;
    allocation_indices = nothing,
    allocation_values = nothing,
)
    m = Int(instance.summary.m)
    n = Int(instance.summary.n)
    original_allocation_count = Int(instance.summary.allocation_count)
    compact = allocation_indices !== nothing || allocation_values !== nothing
    compact &&
        (allocation_indices === nothing || allocation_values === nothing) &&
        error("compact primal metrics require both allocation indices and values")
    allocation_count = compact ?
        length(allocation_indices) : original_allocation_count
    expected_length = allocation_count + 2m
    length(primal) == expected_length ||
        error("primal length $(length(primal)) != $expected_length")

    allocation = @view primal[1:allocation_count]
    supply_sums = zeros(Float64, n)
    utility_sums = zeros(Float64, m)
    if compact
        length(allocation_values) == allocation_count ||
            error("compact allocation index/value lengths differ")
        for cursor in eachindex(allocation_indices)
            allocation_index = allocation_indices[cursor]
            1 <= allocation_index <= original_allocation_count ||
                error("compact allocation index $allocation_index is invalid")
            buyer = fld(allocation_index - 1, n) + 1
            good = mod(allocation_index - 1, n) + 1
            allocation_value = allocation[cursor]
            supply_sums[good] += allocation_value
            utility_sums[buyer] +=
                allocation_values[cursor] * allocation_value
        end
    else
        allocation_matrix = reshape(allocation, n, m)
        supply_sums .= vec(sum(allocation_matrix; dims = 2))
        for cursor in eachindex(instance.utility.nzind)
            allocation_index = instance.utility.nzind[cursor]
            buyer = fld(allocation_index - 1, n) + 1
            utility_sums[buyer] +=
                instance.utility.nzval[cursor] * primal[allocation_index]
        end
    end
    supply_abs_residual = maximum(abs.(supply_sums .- instance.supply))
    supply_rel_residual =
        supply_abs_residual / max(1.0, abs(instance.supply))

    utility_abs_residual = 0.0
    exponential_log_violation = 0.0
    max_abs_t = 0.0
    max_abs_z = 0.0
    max_abs_utility_row = 0.0
    for buyer in 1:m
        t_value = primal[allocation_count + 2buyer - 1]
        z_value = primal[allocation_count + 2buyer]
        utility_row_value = utility_sums[buyer] - z_value
        utility_abs_residual = max(
            utility_abs_residual,
            abs(utility_row_value),
        )
        max_abs_utility_row = max(
            max_abs_utility_row,
            abs(utility_row_value),
        )
        max_abs_t = max(max_abs_t, abs(t_value))
        max_abs_z = max(max_abs_z, abs(z_value))
        cone_violation = if z_value > 0.0
            max(0.0, t_value - log(z_value))
        else
            Inf
        end
        exponential_log_violation =
            max(exponential_log_violation, cone_violation)
    end
    utility_rel_residual =
        utility_abs_residual / max(1.0, max_abs_z)

    minimum_allocation = minimum(allocation)
    nonnegative_violation = max(0.0, -minimum_allocation)
    max_abs_allocation = maximum(abs, allocation)
    nonnegative_relative_violation =
        nonnegative_violation / (1.0 + max_abs_allocation)

    # In the PDCS formulation, Gx contains the supply sums, the utility
    # equality rows, and the `(t, 0, z)` exponential-cone rows.  The right-hand
    # side contains the per-good supply and `-1` in the middle coordinate of
    # every exponential cone.  Use those original-scale quantities rather
    # than a per-component absolute threshold.
    gx_inf = max(
        maximum(abs, supply_sums),
        max_abs_utility_row,
        max_abs_t,
        max_abs_z,
        max_abs_allocation,
    )
    h_inf = max(abs(instance.supply), 1.0)
    conic_relative_scale = 1.0 + max(h_inf, gx_inf)
    conic_absolute_violation_upper_bound = max(
        supply_abs_residual,
        utility_abs_residual,
        exponential_log_violation,
    )
    conic_relative_violation_upper_bound =
        conic_absolute_violation_upper_bound / conic_relative_scale
    exponential_cone_relative_violation_upper_bound =
        exponential_log_violation / conic_relative_scale
    standard_form_absolute_violation_upper_bound = max(
        conic_absolute_violation_upper_bound,
        nonnegative_violation,
    )
    # Explicit nonnegative rows are part of the same standard-form residual
    # as the zero and exponential-cone rows.  Normalize their maximum with the
    # same global scale; a separate per-allocation scale would not match the
    # cuPDCS criterion and can reject an otherwise sub-tolerance KKT point.
    relative_primal_violation_upper_bound =
        standard_form_absolute_violation_upper_bound /
        conic_relative_scale
    objective_value = -dot(
        instance.weights,
        @view(
            primal[
                allocation_count + 1:2:allocation_count + 2m - 1
            ],
        ),
    )

    return (
        objective_value = objective_value,
        supply_abs_residual = supply_abs_residual,
        supply_rel_residual = supply_rel_residual,
        utility_abs_residual = utility_abs_residual,
        utility_rel_residual = utility_rel_residual,
        nonnegative_violation = nonnegative_violation,
        nonnegative_relative_violation = nonnegative_relative_violation,
        exponential_log_violation = exponential_log_violation,
        exponential_cone_relative_violation_upper_bound =
            exponential_cone_relative_violation_upper_bound,
        conic_absolute_violation_upper_bound =
            conic_absolute_violation_upper_bound,
        conic_relative_scale = conic_relative_scale,
        conic_relative_violation_upper_bound =
            conic_relative_violation_upper_bound,
        standard_form_absolute_violation_upper_bound =
            standard_form_absolute_violation_upper_bound,
        relative_primal_violation_upper_bound =
            relative_primal_violation_upper_bound,
        minimum_allocation = minimum_allocation,
        max_abs_allocation = max_abs_allocation,
    )
end

"""
Compute original-scale KKT errors for the standard conic form

    minimize c'x
    subject to A*x + s = b,  s in K.

The infinity-norm relative denominators follow the convention printed by
cuPDCS: `1 + max` of the original-scale quantities.  The caller is
responsible for supplying the solver's primal, dual, and slack vectors in the
same standard-form sign convention.
"""
function standard_form_relative_errors(
    formulation,
    primal::AbstractVector,
    dual::AbstractVector,
    slack::AbstractVector,
)
    row_count, variable_count = size(formulation.A)
    length(primal) == variable_count || error(
        "primal length $(length(primal)) != $variable_count",
    )
    length(dual) == row_count || error(
        "dual length $(length(dual)) != $row_count",
    )
    length(slack) == row_count || error(
        "slack length $(length(slack)) != $row_count",
    )

    primal_infeasibility_abs, primal_infeasibility_rel = let
        ax = Vector{Float64}(undef, row_count)
        mul!(ax, formulation.A, primal)
        ax_inf = maximum(abs, ax)
        slack_inf = maximum(abs, slack)
        b_inf = maximum(abs, formulation.b)
        ax .+= slack
        ax .-= formulation.b
        residual = maximum(abs, ax)
        scale = 1.0 + max(ax_inf, slack_inf, b_inf)
        (residual, residual / scale)
    end

    # Release the row-sized work vector before allocating the column-sized
    # stationarity vector on the largest Fisher instances.
    GC.gc(false)

    dual_infeasibility_abs, dual_infeasibility_rel = let
        aty = Vector{Float64}(undef, variable_count)
        mul!(aty, transpose(formulation.A), dual)
        aty_inf = maximum(abs, aty)
        c_inf = maximum(abs, formulation.c)
        aty .+= formulation.c
        residual = maximum(abs, aty)
        scale = 1.0 + max(aty_inf, c_inf)
        (residual, residual / scale)
    end

    primal_objective = dot(formulation.c, primal)
    dual_objective = -dot(formulation.b, dual)
    primal_dual_gap_abs = abs(primal_objective - dual_objective)
    primal_dual_gap_rel = primal_dual_gap_abs /
        (1.0 + max(abs(primal_objective), abs(dual_objective)))

    return (
        comparison_primal_objective = primal_objective,
        comparison_dual_objective = dual_objective,
        comparison_primal_infeasibility_abs = primal_infeasibility_abs,
        comparison_primal_infeasibility_rel = primal_infeasibility_rel,
        comparison_dual_infeasibility_abs = dual_infeasibility_abs,
        comparison_dual_infeasibility_rel = dual_infeasibility_rel,
        comparison_primal_dual_gap_abs = primal_dual_gap_abs,
        comparison_primal_dual_gap_rel = primal_dual_gap_rel,
    )
end

end # module
