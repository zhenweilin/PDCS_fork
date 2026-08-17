#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays
using Test

const SIBLING_COMMON = normpath(joinpath(
    @__DIR__, "..", "..", "..", "PDCS_fork", "benchmark",
    "large_scale_fisher_market", "fisher_market_common.jl",
))
isfile(SIBLING_COMMON) || error("missing Fisher generator: $SIBLING_COMMON")
include(SIBLING_COMMON)
using .FisherMarketCommon

include(joinpath(@__DIR__, "FisherDirectFormulation.jl"))
using .FisherDirectFormulation

@testset "direct Fisher formulation structure" begin
    m, n = 5, 7
    instance = generate_instance(m, n, 0.65, 2026)
    direct = build_direct_pdcs_formulation(instance)
    allocation_count = m * n

    @test direct.variable_count == allocation_count + m
    @test direct.row_count == n + 3m
    @test direct.zero_count == n
    @test direct.exponential_count == m
    @test nnz(direct.A) == allocation_count + nnz(instance.utility) + m
    @test all(direct.b[1:n] .== instance.supply)
    @test all(
        direct.b[n + 3(i - 1) + 2] == -1.0 for i in 1:m
    )

    # Every nonzero U_ij must appear in the third row of buyer i's cone.
    for cursor in eachindex(instance.utility.nzind)
        allocation_index = instance.utility.nzind[cursor]
        buyer = fld(allocation_index - 1, n) + 1
        cone_third_row = n + 3(buyer - 1) + 3
        @test direct.A[cone_third_row, allocation_index] ==
            instance.utility.nzval[cursor]
    end
    exponential_rows = (n + 1):(n + 3m)
    @test any(
        value != 1.0 for value in nonzeros(direct.A[exponential_rows, :])
    )
end

@testset "direct and lifted formulations are analytically equivalent" begin
    m, n = 6, 9
    instance = generate_instance(m, n, 0.55, 2030)
    allocation_count = m * n
    allocation = fill(instance.supply / m, allocation_count)

    utility_sums = zeros(Float64, m)
    for cursor in eachindex(instance.utility.nzind)
        allocation_index = instance.utility.nzind[cursor]
        buyer = fld(allocation_index - 1, n) + 1
        utility_sums[buyer] +=
            instance.utility.nzval[cursor] * allocation[allocation_index]
    end
    @test all(utility_sums .> 0.0)
    p = log.(utility_sums)

    direct_primal = vcat(allocation, p)
    direct_formulation = build_direct_pdcs_formulation(instance)
    direct_conic_point =
        direct_formulation.A * direct_primal - direct_formulation.b
    for buyer in 1:m
        offset = n + 3(buyer - 1)
        @test direct_conic_point[(offset + 1):(offset + 3)] ≈
            [p[buyer], 1.0, utility_sums[buyer]] atol = 2e-15
    end

    lifted_primal = Vector{Float64}(undef, allocation_count + 2m)
    lifted_primal[1:allocation_count] .= allocation
    for buyer in 1:m
        lifted_primal[allocation_count + 2buyer - 1] = p[buyer]
        lifted_primal[allocation_count + 2buyer] = utility_sums[buyer]
    end

    direct_metrics = independent_direct_metrics(direct_primal, instance)
    lifted_metrics = independent_primal_metrics(lifted_primal, instance)
    @test direct_metrics.objective_value ≈ lifted_metrics.objective_value
    @test direct_metrics.supply_abs_residual <= 1e-14
    @test lifted_metrics.supply_abs_residual <= 1e-14
    @test lifted_metrics.utility_abs_residual <= 1e-14
    @test direct_metrics.nonnegative_violation == 0.0
    @test lifted_metrics.nonnegative_violation == 0.0
    @test direct_metrics.exponential_log_violation <= 1e-14
    @test lifted_metrics.exponential_log_violation <= 1e-14

    # The compact model eliminates exactly m variables, m zero-cone rows,
    # and 2m matrix nonzeros relative to the lifted model.
    lifted_formulation = build_pdcs_formulation(instance)
    @test lifted_formulation.variable_count - direct_formulation.variable_count == m
    @test lifted_formulation.row_count - direct_formulation.row_count == m
    @test nnz(lifted_formulation.A) - nnz(direct_formulation.A) == 2m
end
