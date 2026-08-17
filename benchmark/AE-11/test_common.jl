#!/usr/bin/env julia

include(joinpath(@__DIR__, "AE11Common.jl"))

using .AE11Common
using LinearAlgebra
using SparseArrays
using Test

@testset "AE-11 structured generator" begin
    design = make_design(32, 5, 16, 4, 2026)
    pattern = AE11Common.build_pattern(design)
    @test size(design.latent_index) == (4, 160)
    @test norm(design.b) ≈ 1.0 atol = 2e-13
    hashes = String[]
    for kappa in (1.0, 1e2, 1e4, 1e6, 1e8)
        A, sigma = assemble_matrix(design, pattern, kappa)
        values = svdvals(Matrix(A))
        @test maximum(values) / minimum(values) ≈ kappa rtol = 5e-8
        @test sort(values) ≈ sort(sigma) rtol = 5e-8 atol = 2e-15
        @test eltype(A) == Float64
        push!(hashes, matrix_pattern_hash(A))
    end
    @test length(unique(hashes)) == 1
    @test only(unique(hashes)) == matrix_pattern_hash(pattern)
end

@testset "AE-11 penalty and conic normalization" begin
    design = make_design(32, 5, 16, 4, 2027)
    pattern = AE11Common.build_pattern(design)
    A, _ = assemble_matrix(design, pattern, 1e4)
    panel_a = penalty_spec(A, design.b, "A", 0.5, 1e4)
    panel_b = penalty_spec(A, design.b, "B", 0.1, 1e4)
    @test panel_a.effective_alpha == 1e-4
    @test panel_a.lambda ≈ panel_a.lambda_reference / 1e4
    @test panel_b.lambda ≈ 0.1panel_b.lambda_reference

    conic = build_conic_data(A, design.b, panel_a.lambda)
    @test size(conic.G) == (66, 353)
    @test conic.mGzero == 32
    @test conic.socG == Integer[34]
    @test conic.mGzero + sum(conic.socG) == conic.m_conic
    @test all(conic.bl[1:320] .== 0.0)
    @test all(isinf, conic.bu)

    x = randn(160)
    # The conic residual variable is b-Ax; the verifier uses Ax-b. Their
    # squared norms are identical.
    residual = design.b - A * x
    x_positive = max.(x, 0.0)
    x_negative = max.(-x, 0.0)
    epigraph = dot(residual, residual) / 2
    primal = [x_positive; x_negative; epigraph; residual]
    conic_residual = conic.G * primal - conic.h
    @test norm(conic_residual[1:32], Inf) ≤ 5e-14
    soc = conic_residual[33:end]
    @test soc[1] ≥ norm(soc[2:end]) - 5e-13
    @test dot(conic.c, primal) ≈
        dot(residual, residual) + panel_a.lambda * norm(x, 1)
    @test recover_lasso_x(primal, 160) ≈ x
end

@testset "AE-11 independent verifier" begin
    design = make_design(32, 5, 16, 4, 2026)
    pattern = AE11Common.build_pattern(design)
    A, sigma = assemble_matrix(design, pattern, 1.0)
    # lambda >= 2*lambda_reference makes x=0 optimal for this normalization.
    lambda_reference = norm(transpose(A) * design.b, Inf)
    result = verify_lasso(
        A, design.b, 2lambda_reference, zeros(design.n);
        tolerance = 1e-12, design = design, sigma = sigma,
    )
    @test result["verified_solved"]
    @test result["independent_kkt"] ≤ 1e-14
    @test result["relative_gap"] ≤ 1e-14
end
