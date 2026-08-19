using Test
using JuMP
using LinearAlgebra
using SparseArrays

import MathOptInterface as MOI

include(joinpath(@__DIR__, "large_scale_lasso.jl"))

@testset "synthetic compact Lasso formulation" begin
    entry = Dict{String,Any}(
        "id" => "tiny",
        "m" => 8,
        "n" => 12,
        "density" => 0.4,
        "replicate" => 1,
        "seed" => "20260819",
    )
    first_data = generate_instance(entry)
    second_data = generate_instance(entry)

    @test first_data.A == second_data.A
    @test first_data.x_feat == second_data.x_feat
    @test first_data.b == second_data.b
    @test first_data.lambda == norm(transpose(first_data.A) * first_data.b, Inf)
    @test numerical_digest(first_data) == numerical_digest(second_data)

    m, n = size(first_data.A)
    arrays = model_arrays(first_data)
    @test length(arrays.c) == 2 * n + 1
    @test arrays.c[1:n] == zeros(n)
    @test arrays.c[(n + 1):(2 * n)] == fill(first_data.lambda, n)
    @test arrays.c[end] == 2.0

    model = build_model(first_data, arrays)
    variables = all_variables(model)
    objective = objective_function(model)
    @test num_variables(model) == 2 * n + 1
    @test all(get(objective.terms, variables[j], 0.0) == 0.0 for j in 1:n)
    @test all(
        objective.terms[variables[n + j]] == first_data.lambda for j in 1:n
    )
    @test objective.terms[variables[end]] == 2.0
    @test num_constraints(
        model,
        Vector{AffExpr},
        MOI.SecondOrderCone,
    ) == 1

    summary = summary_for(entry, first_data, arrays, "tiny.cbf.gz", 0.0)
    @test summary["formulation"] == "compact_epigraph_direct_soc"
    @test summary["variable_count_before_bridge"] == 2 * n + 1
    @test summary["affine_conic_row_count"] == m + 2 * n + 2
    @test summary["canonical_matrix_nnz"] == nnz(first_data.A) + 4 * n + 2
    @test summary["soc_dimension"] == m + 2
end
