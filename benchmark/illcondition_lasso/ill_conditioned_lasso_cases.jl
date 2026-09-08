module IllConditionedLassoCases

using LinearAlgebra
using Random
using SHA

export GENERATOR_VERSION,
       LassoInstance,
       arithmetic_eigenvalues,
       generate_family_base,
       generate_instance,
       instance_hashes,
       lasso_metrics,
       verify_instance

const GENERATOR_VERSION = "dense-spd-arithmetic-v1"

struct LassoInstance
    A::Matrix{Float64}
    b::Vector{Float64}
    xstar::Vector{Float64}
    support::Vector{Int}
    lambda::Float64
    lambda_ratio::Float64
    condition_number::Float64
    eigenvalues::Vector{Float64}
    seed::Int
    noise_std::Float64
end

"""Return `n` descending, arithmetically spaced eigenvalues in `[1/K, 1]`."""
function arithmetic_eigenvalues(n::Integer, condition_number::Real)
    n >= 2 || throw(ArgumentError("dimension must be at least two"))
    condition_number > 1 || throw(ArgumentError("condition number must exceed one"))
    return collect(range(1.0; stop = inv(Float64(condition_number)), length = n))
end

"""
Generate the shared orthogonal basis, sparse signal, and noise direction.

Every condition-number case uses this same base, so the spectrum is the only
matrix-family parameter that changes between cases.
"""
function generate_family_base(
    dimension::Integer,
    support_size::Integer,
    seed::Integer,
)
    dimension >= 2 || throw(ArgumentError("dimension must be at least two"))
    1 <= support_size <= dimension || throw(ArgumentError(
        "support_size must lie in 1:dimension",
    ))
    rng = MersenneTwister(seed)
    factorization = qr(randn(rng, dimension, dimension))
    basis = Matrix(factorization.Q)
    triangular = factorization.R
    # Fix QR's otherwise arbitrary column signs for a stable serialized family.
    for column in axes(basis, 2)
        if signbit(triangular[column, column])
            @views basis[:, column] .*= -1.0
        end
    end
    support = sort!(randperm(rng, dimension)[1:support_size])
    xstar = zeros(Float64, dimension)
    xstar[support] .= randn(rng, support_size)
    xstar ./= norm(xstar)
    noise_direction = randn(rng, dimension)
    noise_direction ./= norm(noise_direction)
    return (; basis, xstar, support, noise_direction, seed = Int(seed))
end

"""
Construct a dense symmetric positive-definite Lasso matrix with exactly the
requested arithmetic spectrum. The objective is
`||A*x-b||_2^2 + lambda*||x||_1`.
"""
function generate_instance(
    base,
    condition_number::Real;
    lambda_ratio::Real,
    noise_std::Real,
)
    lambda_ratio > 0 || throw(ArgumentError("lambda_ratio must be positive"))
    noise_std >= 0 || throw(ArgumentError("noise_std must be nonnegative"))
    n = size(base.basis, 1)
    size(base.basis, 2) == n || throw(DimensionMismatch("basis must be square"))
    eigenvalues = arithmetic_eigenvalues(n, condition_number)
    scaled_basis = base.basis .* reshape(eigenvalues, 1, :)
    A = scaled_basis * transpose(base.basis)
    # Remove the last-bit GEMM asymmetry while preserving a dense Matrix.
    A .= 0.5 .* (A .+ transpose(A))
    noise = Float64(noise_std) .* base.noise_direction
    b = A * base.xstar + noise
    lambda_reference = norm(transpose(A) * b, Inf)
    lambda = Float64(lambda_ratio) * lambda_reference
    return LassoInstance(
        A,
        b,
        copy(base.xstar),
        copy(base.support),
        lambda,
        Float64(lambda_ratio),
        Float64(condition_number),
        eigenvalues,
        base.seed,
        Float64(noise_std),
    )
end

function lasso_metrics(inst::LassoInstance, x::AbstractVector{<:Real})
    length(x) == size(inst.A, 2) || throw(DimensionMismatch("invalid x length"))
    residual = inst.A * x - inst.b
    gradient = 2.0 .* (transpose(inst.A) * residual)
    zero_threshold = 1e-10 * max(1.0, norm(x, Inf))
    stationarity = 0.0
    for index in eachindex(x)
        violation = if abs(x[index]) > zero_threshold
            abs(gradient[index] + inst.lambda * sign(x[index]))
        else
            max(abs(gradient[index]) - inst.lambda, 0.0)
        end
        stationarity = max(stationarity, violation)
    end
    normalized_stationarity = stationarity /
        (1.0 + norm(gradient, Inf) + inst.lambda)
    objective = dot(residual, residual) + inst.lambda * norm(x, 1)
    return (;
        stationarity,
        normalized_stationarity,
        objective,
        residual_norm = norm(residual),
        xstar_relative_error = norm(x - inst.xstar) / (1.0 + norm(inst.xstar)),
    )
end

function verify_instance(inst::LassoInstance; compute_spectrum::Bool = true)
    n = size(inst.A, 1)
    size(inst.A, 2) == n || throw(DimensionMismatch("A must be square"))
    length(inst.b) == n || throw(DimensionMismatch("b has invalid length"))
    length(inst.xstar) == n || throw(DimensionMismatch("xstar has invalid length"))
    expected = arithmetic_eigenvalues(n, inst.condition_number)
    inst.eigenvalues == expected || error("stored eigenvalue sequence is not arithmetic")
    symmetry_error = norm(inst.A - transpose(inst.A), Inf) /
        max(1.0, norm(inst.A, Inf))
    dense_nonzeros = count(!iszero, inst.A)
    dense_nonzeros == n * n || error(
        "A is not numerically dense: $dense_nonzeros of $(n * n) entries are nonzero",
    )
    measured_condition_number = NaN
    condition_relative_error = NaN
    eigenvalue_spacing_error = NaN
    if compute_spectrum
        measured = reverse!(eigvals(Symmetric(inst.A)))
        measured_condition_number = first(measured) / last(measured)
        condition_relative_error = abs(
            measured_condition_number - inst.condition_number,
        ) / inst.condition_number
        eigenvalue_spacing_error = norm(measured - expected, Inf) /
            max(1.0, norm(expected, Inf))
    end
    return (;
        dimension = n,
        dense_nonzeros,
        symmetry_error,
        measured_condition_number,
        condition_relative_error,
        eigenvalue_spacing_error,
        minimum_eigenvalue = last(expected),
        maximum_eigenvalue = first(expected),
    )
end

function instance_hashes(inst::LassoInstance)
    digest(array) = bytes2hex(sha256(reinterpret(UInt8, vec(array))))
    return (;
        matrix = digest(inst.A),
        b = digest(inst.b),
        xstar = digest(inst.xstar),
        eigenvalues = digest(inst.eigenvalues),
    )
end

end # module
