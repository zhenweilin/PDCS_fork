module AE11Common

using Dates
using LinearAlgebra
using Printf
using Random
using SHA
using SparseArrays
using TOML

export StructuredDesign, StructuredPattern, assemble_matrix, build_conic_data,
       config_path, file_sha256, gaussian_qr_components,
       gaussian_qr_matrix, instance_metadata, load_config, make_design,
       matrix_pattern_hash, matrix_value_hash, penalty_spec, recover_lasso_x,
       sha256_array, singular_values, structured_vt_mul, verify_lasso,
       write_toml_atomic

const SCHEMA_VERSION = 1

config_path() = joinpath(@__DIR__, "experiment.toml")

function load_config(path::AbstractString = config_path())
    config = TOML.parsefile(path)
    get(config, "schema_version", nothing) == SCHEMA_VERSION ||
        error("unsupported experiment schema")
    get(config, "objective", "") ==
        "norm(A*x-b,2)^2 + lambda*norm(x,1)" ||
        error("objective normalization mismatch")
    get(config, "value_type", "") == "Float64" ||
        error("AE-11 requires Float64")
    get(config, "save_instance_data", true) == false ||
        error("AE-11 must generate instances in memory")
    return config
end

file_sha256(path::AbstractString) = open(path, "r") do stream
    bytes2hex(SHA.sha256(stream))
end

function sha256_array(values::AbstractArray)
    context = SHA.SHA256_CTX()
    SHA.update!(context, reinterpret(UInt8, vec(values)))
    return bytes2hex(SHA.digest!(context))
end

function update_array!(context, tag::AbstractString, values::AbstractVector)
    SHA.update!(context, Vector{UInt8}(codeunits(tag)))
    SHA.update!(context, UInt8[0x00])
    SHA.update!(context, reinterpret(UInt8, [Int64(length(values))]))
    isempty(values) || SHA.update!(context, reinterpret(UInt8, values))
    return context
end

function write_toml_atomic(path::AbstractString, value::AbstractDict)
    mkpath(dirname(abspath(path)))
    temporary, stream = mktemp(dirname(abspath(path)))
    try
        TOML.print(stream, value; sorted = true)
        close(stream)
        mv(temporary, path; force = true)
    catch
        isopen(stream) && close(stream)
        isfile(temporary) && rm(temporary)
        rethrow()
    end
    return path
end

"""Fixed singular-vector data for all condition numbers of one seed."""
struct StructuredDesign
    m::Int
    n::Int
    q::Int
    hu::Int
    hv::Int
    seed::Int
    # U_local[:, j] is the nonzero part of column j of block-diagonal U.
    U_local::Matrix{Float64}
    # Each output column has hv nonzeros in its W row. The latent index and
    # value arrays represent all q blocks of V without materializing V.
    latent_index::Matrix{Int32}
    latent_value::Matrix{Float64}
    spectral_permutation::Vector{Int32}
    c::Vector{Float64}
    b::Vector{Float64}
end

"""K-independent CSC structure for A_K."""
struct StructuredPattern
    m::Int
    n::Int
    colptr::Vector{Int32}
    rowval::Vector{Int32}
    hash::String
end

function haar_block!(destination::AbstractMatrix{Float64}, rng)
    size(destination, 1) == size(destination, 2) ||
        error("Haar block must be square")
    gaussian = randn(rng, size(destination)...)
    factor = qr(gaussian)
    orthogonal = Matrix(factor.Q)
    triangular = factor.R
    for column in axes(orthogonal, 2)
        sign_value = sign(triangular[column, column])
        sign_value == 0.0 && (sign_value = 1.0)
        @views orthogonal[:, column] .*= sign_value
    end
    copyto!(destination, orthogonal)
    return destination
end

function haar_matrix(rng, rows::Int, columns::Int)
    rows >= columns || error("thin Haar matrix requires rows >= columns")
    gaussian = randn(rng, rows, columns)
    factor = qr(gaussian)
    orthogonal = Matrix(factor.Q[:, 1:columns])
    triangular = factor.R
    for column in 1:columns
        sign_value = sign(triangular[column, column])
        sign_value == 0.0 && (sign_value = 1.0)
        @views orthogonal[:, column] .*= sign_value
    end
    return orthogonal
end

"""Dense Gaussian-QR construction used only by the configured pilot."""
function gaussian_qr_components(m::Integer, q::Integer, seed::Integer)
    m > 1 || error("m must exceed one")
    q > 0 || error("q must be positive")
    n = Int(q) * Int(m)
    rng = MersenneTwister(Int(seed))
    U = haar_matrix(rng, Int(m), Int(m))
    V = haar_matrix(rng, n, Int(m))
    permutation = randperm(rng, Int(m))
    normalization = inv(sqrt(Float64(m)))
    c = [rand(rng, Bool) ? normalization : -normalization for _ in 1:Int(m)]
    b = U * c
    return (; U, V, permutation, c, b)
end

function gaussian_qr_matrix(components, kappa::Real)
    m = size(components.U, 2)
    sigma = Vector{Float64}(undef, m)
    log_kappa = log(Float64(kappa))
    for rank_index in 1:m
        sigma[components.permutation[rank_index]] =
            exp(-log_kappa * (rank_index - 1) / Float64(m - 1))
    end
    scaled_U = components.U .* reshape(sigma, 1, :)
    return scaled_U * transpose(components.V), sigma
end

function make_design(m::Integer, q::Integer, hu::Integer, hv::Integer,
                     seed::Integer)
    m > 1 || error("m must exceed one")
    q > 0 || error("q must be positive")
    m % hu == 0 || error("m must be divisible by u_block_size")
    m % hv == 0 || error("m must be divisible by v_block_size")
    m <= typemax(Int32) || error("m exceeds Int32 sparse-index range")
    n = q * m
    n <= typemax(Int32) || error("n exceeds Int32 sparse-index range")
    rng = MersenneTwister(Int(seed))

    U_local = Matrix{Float64}(undef, hu, m)
    block = Matrix{Float64}(undef, hu, hu)
    for block_start in 1:hu:m
        haar_block!(block, rng)
        @views U_local[:, block_start:(block_start + hu - 1)] .= block
    end

    latent_index = Matrix{Int32}(undef, hv, n)
    latent_value = Matrix{Float64}(undef, hv, n)
    wblock = Matrix{Float64}(undef, hv, hv)
    for stack in 1:q
        row_permutation = randperm(rng, m)
        column_permutation = randperm(rng, m)
        stack_offset = (stack - 1) * m
        for block_start in 1:hv:m
            haar_block!(wblock, rng)
            for local_row in 1:hv
                output_row = row_permutation[block_start + local_row - 1]
                output_column = stack_offset + output_row
                for local_column in 1:hv
                    latent = column_permutation[
                        block_start + local_column - 1
                    ]
                    latent_index[local_column, output_column] = Int32(latent)
                    latent_value[local_column, output_column] =
                        wblock[local_row, local_column]
                end
            end
        end
    end

    spectral_permutation = Int32.(randperm(rng, m))
    c = Vector{Float64}(undef, m)
    normalization = inv(sqrt(Float64(m)))
    for index in eachindex(c)
        c[index] = rand(rng, Bool) ? normalization : -normalization
    end
    b = zeros(Float64, m)
    for block_start in 1:hu:m
        for local_row in 1:hu
            value = 0.0
            global_row = block_start + local_row - 1
            for local_column in 1:hu
                latent = block_start + local_column - 1
                value += U_local[local_row, latent] * c[latent]
            end
            b[global_row] = value
        end
    end
    isapprox(norm(b), 1.0; rtol = 2e-13, atol = 2e-13) ||
        error("constructed response is not unit norm")
    return StructuredDesign(
        Int(m), Int(n), Int(q), Int(hu), Int(hv), Int(seed), U_local,
        latent_index, latent_value, spectral_permutation, c, b,
    )
end

function singular_values(design::StructuredDesign, kappa::Real)
    kappa >= 1 || error("target kappa must be at least one")
    sigma = Vector{Float64}(undef, design.m)
    log_kappa = log(Float64(kappa))
    denominator = Float64(design.m - 1)
    for rank_index in 1:design.m
        latent = Int(design.spectral_permutation[rank_index])
        sigma[latent] = exp(-log_kappa * (rank_index - 1) / denominator)
    end
    return sigma
end

@inline function sort_latents_by_ublock!(
    latents::Vector{Int32}, values::Vector{Float64}, hu::Int,
)
    for index in 2:length(latents)
        latent = latents[index]
        value = values[index]
        block_id = (Int(latent) - 1) ÷ hu
        previous = index - 1
        while previous >= 1 &&
              ((Int(latents[previous]) - 1) ÷ hu) > block_id
            latents[previous + 1] = latents[previous]
            values[previous + 1] = values[previous]
            previous -= 1
        end
        latents[previous + 1] = latent
        values[previous + 1] = value
    end
    return nothing
end

function sorted_column_latents!(
    latents::Vector{Int32}, values::Vector{Float64},
    design::StructuredDesign, column::Int,
)
    @inbounds for entry in 1:design.hv
        latents[entry] = design.latent_index[entry, column]
        values[entry] = design.latent_value[entry, column]
    end
    sort_latents_by_ublock!(latents, values, design.hu)
    return nothing
end

function unique_ublocks(latents::Vector{Int32}, hu::Int)
    count = 0
    previous = -1
    @inbounds for latent in latents
        block_id = (Int(latent) - 1) ÷ hu
        if block_id != previous
            count += 1
            previous = block_id
        end
    end
    return count
end

function build_pattern(design::StructuredDesign)
    colptr = Vector{Int32}(undef, design.n + 1)
    colptr[1] = 1
    latents = Vector{Int32}(undef, design.hv)
    values = Vector{Float64}(undef, design.hv)
    next_position = Int64(1)
    for column in 1:design.n
        sorted_column_latents!(latents, values, design, column)
        next_position += design.hu * unique_ublocks(latents, design.hu)
        next_position <= typemax(Int32) ||
            error("CSC entry count exceeds Int32 range")
        colptr[column + 1] = Int32(next_position)
    end
    rowval = Vector{Int32}(undef, Int(next_position - 1))
    for column in 1:design.n
        sorted_column_latents!(latents, values, design, column)
        destination = Int(colptr[column])
        entry = 1
        while entry <= design.hv
            block_id = (Int(latents[entry]) - 1) ÷ design.hu
            block_start = block_id * design.hu + 1
            next_entry = entry + 1
            while next_entry <= design.hv &&
                  (Int(latents[next_entry]) - 1) ÷ design.hu == block_id
                next_entry += 1
            end
            for local_row in 1:design.hu
                rowval[destination] = Int32(block_start + local_row - 1)
                destination += 1
            end
            entry = next_entry
        end
        destination == Int(colptr[column + 1]) ||
            error("internal CSC pattern assembly error")
    end
    context = SHA.SHA256_CTX()
    update_array!(context, "colptr", colptr)
    update_array!(context, "rowval", rowval)
    hash = bytes2hex(SHA.digest!(context))
    return StructuredPattern(design.m, design.n, colptr, rowval, hash)
end

matrix_pattern_hash(pattern::StructuredPattern) = pattern.hash

function assemble_matrix(
    design::StructuredDesign,
    pattern::StructuredPattern,
    kappa::Real,
)
    design.m == pattern.m && design.n == pattern.n ||
        error("pattern dimensions do not match design")
    sigma = singular_values(design, kappa)
    nzval = Vector{Float64}(undef, length(pattern.rowval))
    latents = Vector{Int32}(undef, design.hv)
    weights = Vector{Float64}(undef, design.hv)
    stack_normalization = inv(sqrt(Float64(design.q)))
    for column in 1:design.n
        sorted_column_latents!(latents, weights, design, column)
        destination = Int(pattern.colptr[column])
        entry = 1
        while entry <= design.hv
            block_id = (Int(latents[entry]) - 1) ÷ design.hu
            next_entry = entry + 1
            while next_entry <= design.hv &&
                  (Int(latents[next_entry]) - 1) ÷ design.hu == block_id
                next_entry += 1
            end
            for local_row in 1:design.hu
                value = 0.0
                @inbounds for item in entry:(next_entry - 1)
                    latent = Int(latents[item])
                    value += design.U_local[local_row, latent] *
                        sigma[latent] * weights[item]
                end
                nzval[destination] = value * stack_normalization
                destination += 1
            end
            entry = next_entry
        end
        destination == Int(pattern.colptr[column + 1]) ||
            error("internal CSC value assembly error")
    end
    A = SparseMatrixCSC{Float64,Int32}(
        design.m, design.n, copy(pattern.colptr), copy(pattern.rowval), nzval,
    )
    return A, sigma
end

function matrix_pattern_hash(A::SparseMatrixCSC)
    context = SHA.SHA256_CTX()
    update_array!(context, "colptr", A.colptr)
    update_array!(context, "rowval", A.rowval)
    return bytes2hex(SHA.digest!(context))
end

function matrix_value_hash(A::SparseMatrixCSC)
    context = SHA.SHA256_CTX()
    update_array!(context, "nzval", A.nzval)
    return bytes2hex(SHA.digest!(context))
end

function penalty_spec(A, b, panel::AbstractString, parameter::Real)
    lambda_reference = norm(transpose(A) * b, Inf)
    if uppercase(panel) == "A"
        beta = Float64(parameter)
        kappa = error("panel A requires penalty_spec(A,b,panel,beta,kappa)")
    elseif uppercase(panel) == "B"
        alpha = Float64(parameter)
        return (
            panel = "B", beta = NaN, alpha = alpha,
            effective_alpha = alpha,
            lambda_reference = lambda_reference,
            lambda = alpha * lambda_reference,
        )
    end
    error("unknown panel: $panel")
end

function penalty_spec(
    A, b, panel::AbstractString, parameter::Real, kappa::Real,
)
    uppercase(panel) == "A" || return penalty_spec(A, b, panel, parameter)
    beta = Float64(parameter)
    effective_alpha = 2beta / Float64(kappa)
    lambda_reference = norm(transpose(A) * b, Inf)
    return (
        panel = "A", beta = beta, alpha = NaN,
        effective_alpha = effective_alpha,
        lambda_reference = lambda_reference,
        lambda = effective_alpha * lambda_reference,
    )
end

"""
Build a direct conic form with variables
`[x_positive; x_negative; residual_epigraph; residual]`.

The equality rows impose `residual + A*(x_positive-x_negative) = b`. The SOC
rows are `[(1+s)/sqrt(2); (1-s)/sqrt(2); residual]`, so cone membership gives
`2s >= norm(residual)^2`; minimizing `2s` represents the squared residual
exactly. This constraint-side SOC uses the production path shared by all
solvers and avoids any solver-specific primal-cone conversion.
"""
function build_conic_data(A::SparseMatrixCSC{Float64}, b, lambda::Float64)
    m, n = size(A)
    num_variables = 2n + m + 1
    num_constraints = 2m + 2
    matrix_nnz = 2nnz(A) + 2m + 2
    matrix_nnz <= typemax(Int32) ||
        error("conic CSC entry count exceeds Int32 range")

    colptr = Vector{Int32}(undef, num_variables + 1)
    rowval = Vector{Int32}(undef, matrix_nnz)
    nzval = Vector{Float64}(undef, matrix_nnz)
    position = 1
    colptr[1] = 1
    for column in 1:n
        for source in Int(A.colptr[column]):(Int(A.colptr[column + 1]) - 1)
            rowval[position] = A.rowval[source]
            nzval[position] = A.nzval[source]
            position += 1
        end
        colptr[column + 1] = Int32(position)
    end
    for column in 1:n
        for source in Int(A.colptr[column]):(Int(A.colptr[column + 1]) - 1)
            rowval[position] = A.rowval[source]
            nzval[position] = -A.nzval[source]
            position += 1
        end
        colptr[n + column + 1] = Int32(position)
    end
    # residual_epigraph contributes to the first two SOC coordinates.
    rowval[position] = Int32(m + 1)
    nzval[position] = inv(sqrt(2.0))
    position += 1
    colptr[2n + 2] = Int32(position)
    rowval[position] = Int32(m + 2)
    nzval[position] = -inv(sqrt(2.0))
    position += 1
    colptr[2n + 2] = Int32(position)
    # Each residual variable appears in one equality and one SOC coordinate.
    for row in 1:m
        rowval[position] = Int32(row)
        nzval[position] = 1.0
        position += 1
        rowval[position] = Int32(m + 2 + row)
        nzval[position] = 1.0
        position += 1
        colptr[2n + 2 + row] = Int32(position)
    end
    position == matrix_nnz + 1 || error("conic CSC assembly error")

    G = SparseMatrixCSC{Float64,Int32}(
        num_constraints, num_variables, colptr, rowval, nzval,
    )
    c = zeros(Float64, num_variables)
    c[1:(2n)] .= lambda
    c[2n + 1] = 2.0
    h = Vector{Float64}(undef, num_constraints)
    h[1:m] .= b
    h[m + 1] = -inv(sqrt(2.0))
    h[m + 2] = -inv(sqrt(2.0))
    h[(m + 3):end] .= 0.0
    bl = fill(-Inf, num_variables)
    bl[1:(2n)] .= 0.0
    return (
        n_lasso = n,
        m_lasso = m,
        n_conic = num_variables,
        m_conic = num_constraints,
        nb = num_variables,
        c = c,
        G = G,
        h = h,
        mGzero = m,
        mGnonnegative = 0,
        socG = Integer[m + 2],
        rsocG = Integer[],
        expG = 0,
        dual_expG = 0,
        bl = bl,
        bu = fill(Inf, num_variables),
        soc_x = Integer[],
        rsoc_x = Integer[],
        exp_x = 0,
        dual_exp_x = 0,
    )
end

function recover_lasso_x(primal::AbstractVector, n::Integer)
    length(primal) >= 2n || error("conic primal is too short")
    return Vector{Float64}(@view(primal[1:n])) .-
        Vector{Float64}(@view(primal[(n + 1):(2n)]))
end

function structured_vt_mul(design::StructuredDesign, x::AbstractVector)
    length(x) == design.n || error("x length mismatch")
    z = zeros(Float64, design.m)
    normalization = inv(sqrt(Float64(design.q)))
    for column in 1:design.n
        x_value = x[column] * normalization
        @inbounds for entry in 1:design.hv
            latent = Int(design.latent_index[entry, column])
            z[latent] += design.latent_value[entry, column] * x_value
        end
    end
    return z
end

function verify_lasso(
    A::SparseMatrixCSC{Float64}, b::Vector{Float64}, lambda::Float64,
    x::AbstractVector; tolerance::Float64, reference_objective = nothing,
    design::Union{Nothing,StructuredDesign} = nothing,
    sigma::Union{Nothing,Vector{Float64}} = nothing,
    tail_fraction::Float64 = 0.10,
)
    n = size(A, 2)
    length(x) == n || error("solution length mismatch")
    x_values = Vector{Float64}(x)
    nonfinite_count = count(!isfinite, x_values)
    if nonfinite_count > 0
        return Dict{String,Any}(
            "verified_solved" => false,
            "nonfinite_count" => nonfinite_count,
            "independent_kkt" => Inf,
        )
    end
    residual = A * x_values - b
    gradient = 2.0 .* (transpose(A) * residual)
    active_threshold = 1e-10 * max(1.0, norm(x_values, Inf))
    distance_inf = 0.0
    for index in eachindex(x_values)
        distance = if x_values[index] > active_threshold
            abs(gradient[index] + lambda)
        elseif x_values[index] < -active_threshold
            abs(gradient[index] - lambda)
        else
            max(abs(gradient[index]) - lambda, 0.0)
        end
        distance_inf = max(distance_inf, distance)
    end
    independent_kkt = distance_inf /
        (1.0 + norm(gradient, Inf) + lambda)
    objective = dot(residual, residual) + lambda * norm(x_values, 1)

    # y=2(Ax-b) satisfies the Lasso stationarity constraint at optimality.
    # Scaling it toward zero gives an always-feasible dual point.
    y = 2.0 .* residual
    dual_gradient_inf = norm(transpose(A) * y, Inf)
    dual_scale = dual_gradient_inf == 0.0 ? 1.0 :
        min(1.0, lambda / dual_gradient_inf)
    y .*= dual_scale
    dual_feasibility = max(norm(transpose(A) * y, Inf) - lambda, 0.0) /
        (1.0 + lambda)
    dual_objective = -dot(b, y) - 0.25 * dot(y, y)
    relative_gap = abs(objective - dual_objective) /
        (1.0 + abs(objective) + abs(dual_objective))
    relative_objective_error = reference_objective === nothing ? NaN :
        abs(objective - reference_objective) /
        (1.0 + abs(reference_objective))

    tail_activation_ratio = NaN
    if design !== nothing && sigma !== nothing
        z = structured_vt_mul(design, x_values)
        tail_count = max(1, ceil(Int, tail_fraction * length(sigma)))
        tail_indices = partialsortperm(sigma, 1:tail_count)
        tail_activation_ratio = norm(@view(z[tail_indices])) /
            (1.0 + norm(z))
    end
    return Dict{String,Any}(
        "verified_solved" => independent_kkt <= tolerance &&
            dual_feasibility <= max(tolerance, 10eps(Float64)) &&
            relative_gap <= max(10tolerance, 1e-12),
        "nonfinite_count" => nonfinite_count,
        "active_coordinate_threshold" => active_threshold,
        "primal_feasibility" => 0.0,
        "dual_feasibility" => dual_feasibility,
        "relative_gap" => relative_gap,
        "independent_kkt" => independent_kkt,
        "objective_value" => objective,
        "dual_objective_value" => dual_objective,
        "relative_objective_error" => relative_objective_error,
        "residual_norm_2" => norm(residual),
        "x_norm_1" => norm(x_values, 1),
        "tail_activation_ratio" => tail_activation_ratio,
    )
end

function instance_metadata(
    design::StructuredDesign, pattern::StructuredPattern,
    A::SparseMatrixCSC{Float64}, sigma::Vector{Float64}, kappa::Real,
    penalty, panel::AbstractString,
)
    return Dict{String,Any}(
        "instance_id" => @sprintf(
            "m%d_n%d_seed%d_K%.0e_panel%s", design.m, design.n,
            design.seed, Float64(kappa), uppercase(panel),
        ),
        "seed" => design.seed,
        "m" => design.m,
        "n" => design.n,
        "q" => design.q,
        "u_block_size" => design.hu,
        "v_block_size" => design.hv,
        "target_kappa" => Float64(kappa),
        "sigma_max" => maximum(sigma),
        "sigma_min" => minimum(sigma),
        "measured_kappa" => maximum(sigma) / minimum(sigma),
        "panel" => uppercase(panel),
        "beta" => penalty.beta,
        "alpha" => penalty.alpha,
        "effective_alpha" => penalty.effective_alpha,
        "lambda" => penalty.lambda,
        "lambda_reference" => penalty.lambda_reference,
        "nnz" => nnz(A),
        "A_norm_fro" => norm(A.nzval),
        "b_norm_2" => norm(design.b),
        "b_norm_inf" => norm(design.b, Inf),
        "value_type" => string(eltype(A)),
        "matrix_pattern_hash" => matrix_pattern_hash(pattern),
        "matrix_value_hash" => matrix_value_hash(A),
        "b_hash" => sha256_array(design.b),
    )
end

end # module
