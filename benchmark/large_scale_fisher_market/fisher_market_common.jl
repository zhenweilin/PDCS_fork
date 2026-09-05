module FisherMarketCommon

using LinearAlgebra
using Random
using SHA
using SparseArrays

export build_pdcs_formulation
export build_standard_formulation
export empty_quadratic
export generate_instance
export independent_primal_metrics

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
    for allocation_index in utility.nzind
        buyer = fld(allocation_index - 1, n) + 1
        buyers_with_utility[buyer] = true
    end
    all(buyers_with_utility) ||
        error("seed $seed generated a buyer with no positive utility")

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
`t <= log(z)`, avoiding overflow in `exp(t)`.
"""
function independent_primal_metrics(primal::AbstractVector, instance)
    m = Int(instance.summary.m)
    n = Int(instance.summary.n)
    allocation_count = Int(instance.summary.allocation_count)
    expected_length = allocation_count + 2m
    length(primal) == expected_length ||
        error("primal length $(length(primal)) != $expected_length")

    allocation = @view primal[1:allocation_count]
    allocation_matrix = reshape(allocation, n, m)
    supply_sums = vec(sum(allocation_matrix; dims = 2))
    supply_abs_residual = maximum(abs.(supply_sums .- instance.supply))
    supply_rel_residual =
        supply_abs_residual / max(1.0, abs(instance.supply))

    utility_sums = zeros(Float64, m)
    for cursor in eachindex(instance.utility.nzind)
        allocation_index = instance.utility.nzind[cursor]
        buyer = fld(allocation_index - 1, n) + 1
        utility_sums[buyer] +=
            instance.utility.nzval[cursor] * primal[allocation_index]
    end

    utility_abs_residual = 0.0
    exponential_log_violation = 0.0
    max_abs_z = 0.0
    for buyer in 1:m
        t_value = primal[allocation_count + 2buyer - 1]
        z_value = primal[allocation_count + 2buyer]
        utility_abs_residual = max(
            utility_abs_residual,
            abs(utility_sums[buyer] - z_value),
        )
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
        exponential_log_violation = exponential_log_violation,
        minimum_allocation = minimum_allocation,
    )
end

end # module
