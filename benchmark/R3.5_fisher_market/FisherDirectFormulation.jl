module FisherDirectFormulation

using LinearAlgebra
using SparseArrays

export build_direct_pdcs_formulation
export independent_direct_metrics

const INDEX_TYPE = Int32

function _checked_index(value::Integer, label::AbstractString)
    value <= typemax(INDEX_TYPE) ||
        error("$label=$value exceeds $(INDEX_TYPE)")
    value >= 0 || error("$label must be nonnegative")
    return INDEX_TYPE(value)
end

"""
Construct the direct Fisher-market formulation for cuPDCS:

    min  -sum_i w_i p_i
    s.t. sum_i X_ij = supply,                         j = 1,...,n,
         (p_i, 1, sum_j U_ij X_ij) in K_exp,          i = 1,...,m,
         X >= 0.

The returned positive-sign convention is `G*x - h in Zero(n) x ExpCone^m`.
Allocation nonnegativity is represented by variable lower bounds.  Variables
use buyer-major allocation order followed by `p[1:m]`.
"""
function build_direct_pdcs_formulation(instance)
    m = Int(instance.summary.m)
    n = Int(instance.summary.n)
    allocation_count = Int(instance.summary.allocation_count)
    variable_count = allocation_count + m
    row_count = n + 3m
    matrix_nnz = allocation_count + nnz(instance.utility) + m

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

        # Supply equality for this good.
        rowval[position] = INDEX_TYPE(good)
        nzval[position] = 1.0
        position += 1

        # The utility coefficient appears directly in the third coordinate
        # of buyer i's exponential cone.
        if utility_cursor <= utility_stop &&
           utility_indices[utility_cursor] == allocation_index
            cone_third_row = n + 3(buyer - 1) + 3
            rowval[position] = INDEX_TYPE(cone_third_row)
            nzval[position] = utility_values[utility_cursor]
            position += 1
            utility_cursor += 1
        end
    end
    utility_cursor == utility_stop + 1 ||
        error("not all utility entries were consumed")

    for buyer in 1:m
        p_column = allocation_count + buyer
        cone_first_row = n + 3(buyer - 1) + 1
        colptr[p_column] = INDEX_TYPE(position)
        rowval[position] = INDEX_TYPE(cone_first_row)
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
        # With G*x-h in K_exp, h=-1 creates the constant middle coordinate 1.
        rhs[n + 3(buyer - 1) + 2] = -1.0
    end
    objective = zeros(Float64, variable_count)
    objective[(allocation_count + 1):variable_count] .= -instance.weights

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
        zero_count = n,
        nonnegative_count = 0,
        exponential_count = m,
        formulation = "direct_utility",
        matrix_nnz = matrix_nnz,
    )
end

"""
Compute solver-independent primal checks for the direct formulation.

For `(p_i, 1, u_i) in K_exp`, feasibility is checked in log space as
`p_i <= log(u_i)`, avoiding overflow in `exp(p_i)`.
"""
function independent_direct_metrics(primal::AbstractVector, instance)
    m = Int(instance.summary.m)
    n = Int(instance.summary.n)
    allocation_count = Int(instance.summary.allocation_count)
    expected_length = allocation_count + m
    length(primal) == expected_length ||
        error("primal length $(length(primal)) != $expected_length")

    allocation = @view primal[1:allocation_count]
    p = @view primal[(allocation_count + 1):expected_length]
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

    exponential_log_violation = 0.0
    for buyer in 1:m
        utility_value = utility_sums[buyer]
        cone_violation = utility_value > 0.0 ?
            max(0.0, p[buyer] - log(utility_value)) : Inf
        exponential_log_violation =
            max(exponential_log_violation, cone_violation)
    end

    minimum_allocation = minimum(allocation)
    return (
        objective_value = -dot(instance.weights, p),
        supply_abs_residual = supply_abs_residual,
        supply_rel_residual = supply_rel_residual,
        nonnegative_violation = max(0.0, -minimum_allocation),
        exponential_log_violation = exponential_log_violation,
        minimum_allocation = minimum_allocation,
        minimum_utility = minimum(utility_sums),
    )
end

end # module
