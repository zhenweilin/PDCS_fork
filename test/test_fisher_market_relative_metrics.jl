using LinearAlgebra
using Random
using SparseArrays
using Test

include(joinpath(
    @__DIR__,
    "..",
    "benchmark",
    "large_scale_fisher_market",
    "fisher_market_common.jl",
))
using .FisherMarketCommon

@testset "Fisher original-scale comparison errors" begin
    formulation = (
        A = sparse([2.0 0.0; 0.0 -3.0]),
        b = [3.0, 4.0],
        c = [-19.0, -3.0],
    )
    primal = [1.0, 2.0]
    dual = [10.0, -1.0]
    slack = [1.5, 10.0]

    metrics = standard_form_relative_errors(
        formulation,
        primal,
        dual,
        slack,
    )

    @test metrics.comparison_primal_infeasibility_abs == 0.5
    @test metrics.comparison_primal_infeasibility_rel == 0.5 / 11.0
    @test metrics.comparison_dual_infeasibility_abs == 1.0
    @test metrics.comparison_dual_infeasibility_rel == 1.0 / 21.0
    @test metrics.comparison_primal_objective == -25.0
    @test metrics.comparison_dual_objective == -26.0
    @test metrics.comparison_primal_dual_gap_abs == 1.0
    @test metrics.comparison_primal_dual_gap_rel == 1.0 / 27.0
end

@testset "Reduced Fisher matrix is an exact submodel" begin
    for seed in 1:40
        rng = MersenneTwister(seed)
        m = rand(rng, 2:8)
        n = rand(rng, 2:9)
        dense_utility = rand(rng, m, n)
        dense_utility[rand(rng, m, n) .< 0.55] .= 0.0
        for buyer in 1:m
            dense_utility[buyer, rand(rng, 1:n)] = rand(rng) + 0.1
        end
        for good in 1:n
            dense_utility[rand(rng, 1:m), good] = rand(rng) + 0.1
        end
        buyer_major_utility = vec(permutedims(dense_utility))
        utility = sparse(buyer_major_utility)
        instance = (
            weights = rand(rng, m) .+ 0.1,
            utility = utility,
            supply = rand(rng) + 0.5,
            summary = (
                m = Int64(m),
                n = Int64(n),
                allocation_count = Int64(m * n),
            ),
        )

        full = build_standard_formulation(instance)
        compact = build_compact_standard_formulation(instance)
        full_pdcs = build_pdcs_formulation(instance)
        compact_pdcs = build_compact_pdcs_formulation(instance)
        allocation_count = length(compact.allocation_indices)
        column_map = vcat(
            compact.allocation_indices,
            collect((m * n + 1):(m * n + 2m)),
        )
        row_map = vcat(
            collect(1:(n + m)),
            n + m .+ compact.allocation_indices,
            collect(
                (n + m + m * n + 1):(n + m + m * n + 3m),
            ),
        )

        @test compact.A == full.A[row_map, column_map]
        @test compact.b == full.b[row_map]
        @test compact.c == full.c[column_map]
        @test compact.variable_count == allocation_count + 2m
        @test compact.row_count == n + m + allocation_count + 3m
        @test compact_pdcs.A == full_pdcs.A[:, column_map]
        @test compact_pdcs.b == full_pdcs.b
        @test compact_pdcs.c == full_pdcs.c[column_map]
        @test compact_pdcs.lower_bounds ==
              full_pdcs.lower_bounds[column_map]
        @test compact_pdcs.upper_bounds ==
              full_pdcs.upper_bounds[column_map]
        @test compact_pdcs.nonnegative_count == 0
        @test compact_pdcs.row_count == n + 4m

        compact_primal = randn(rng, compact.variable_count)
        full_primal = zeros(full.variable_count)
        full_primal[column_map] .= compact_primal
        compact_metrics = independent_primal_metrics(
            compact_primal,
            instance;
            allocation_indices = compact.allocation_indices,
            allocation_values = compact.allocation_values,
        )
        full_metrics = independent_primal_metrics(full_primal, instance)
        for name in setdiff(
            propertynames(full_metrics),
            (:minimum_allocation,),
        )
            @test getproperty(compact_metrics, name) ≈
                  getproperty(full_metrics, name)
        end
    end
end

@testset "Reduced Fisher formulation preserves the original problem" begin
    utility = sparsevec([1, 3, 5], [2.0, 3.0, 4.0], 6)
    instance = (
        weights = [1.0, 2.0],
        utility = utility,
        supply = 2.0,
        summary = (
            m = Int64(2),
            n = Int64(3),
            allocation_count = Int64(6),
        ),
    )
    formulation = build_compact_standard_formulation(instance)

    @test formulation.formulation_variant ==
          "positive_valuation_reduced_v1"
    @test formulation.original_allocation_count == 6
    @test formulation.modeled_allocation_count == 3
    @test formulation.removed_zero_valuation_count == 3
    @test formulation.allocation_indices == [1, 3, 5]
    @test formulation.allocation_values == [2.0, 3.0, 4.0]
    @test formulation.variable_count == 7
    @test formulation.nonnegative_count == 3

    compact_primal = [
        2.0,
        2.0,
        2.0,
        log(10.0),
        10.0,
        log(8.0),
        8.0,
    ]
    compact_slack = formulation.b - formulation.A * compact_primal
    @test compact_slack[1:formulation.zero_count] ≈ zeros(5)
    @test compact_slack[6:8] ≈ [2.0, 2.0, 2.0]
    @test compact_slack[9:end] ≈ [
        log(10.0),
        1.0,
        10.0,
        log(8.0),
        1.0,
        8.0,
    ]
    full_primal = [
        2.0,
        0.0,
        2.0,
        0.0,
        2.0,
        0.0,
        log(10.0),
        10.0,
        log(8.0),
        8.0,
    ]
    compact_metrics = independent_primal_metrics(
        compact_primal,
        instance;
        allocation_indices = formulation.allocation_indices,
        allocation_values = formulation.allocation_values,
    )
    full_metrics = independent_primal_metrics(full_primal, instance)
    for name in setdiff(
        propertynames(full_metrics),
        (:minimum_allocation,),
    )
        @test getproperty(compact_metrics, name) ≈
              getproperty(full_metrics, name)
    end
    @test compact_metrics.minimum_allocation == 2.0
    @test full_metrics.minimum_allocation == 0.0

    missing_good_instance = (
        weights = [1.0],
        utility = sparsevec([1], [1.0], 2),
        supply = 1.0,
        summary = (
            m = Int64(1),
            n = Int64(2),
            allocation_count = Int64(2),
        ),
    )
    @test_throws ErrorException build_compact_standard_formulation(
        missing_good_instance,
    )
end

@testset "Fisher EXP validation is relative" begin
    instance = (
        weights = [1.0],
        utility = sparsevec([1], [1.0], 1),
        supply = 1.0e6,
        summary = (
            m = Int64(1),
            n = Int64(1),
            allocation_count = Int64(1),
        ),
    )
    primal = [1.0e6, log(1.0e6) + 0.1, 1.0e6]
    metrics = independent_primal_metrics(primal, instance)

    @test metrics.supply_abs_residual == 0.0
    @test metrics.utility_abs_residual == 0.0
    @test metrics.exponential_log_violation ≈ 0.1
    @test metrics.exponential_log_violation > 1.0e-6
    @test metrics.exponential_cone_relative_violation_upper_bound < 1.0e-6
    @test metrics.relative_primal_violation_upper_bound < 1.0e-6
end

@testset "Nonnegative residual uses the global primal scale" begin
    epsilon = 1.4e-5
    second_allocation = 5.0 + epsilon
    first_utility = epsilon
    second_utility = 100.0 * second_allocation
    instance = (
        weights = [1.0, 1.0],
        utility = sparsevec([1, 2], [-1.0, 100.0], 2),
        supply = 5.0,
        summary = (
            m = Int64(2),
            n = Int64(1),
            allocation_count = Int64(2),
        ),
    )
    primal = [
        -epsilon,
        second_allocation,
        log(first_utility),
        first_utility,
        log(second_utility),
        second_utility,
    ]
    metrics = independent_primal_metrics(primal, instance)

    @test metrics.nonnegative_violation == epsilon
    @test metrics.nonnegative_relative_violation > 1.0e-6
    @test metrics.standard_form_absolute_violation_upper_bound == epsilon
    @test metrics.relative_primal_violation_upper_bound < 1.0e-6
    @test metrics.relative_primal_violation_upper_bound ≈
          epsilon / (1.0 + second_utility)
end
