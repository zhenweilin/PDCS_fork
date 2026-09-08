using LinearAlgebra
using TOML
using Test

include(joinpath(@__DIR__, "ill_conditioned_lasso_cases.jl"))
using .IllConditionedLassoCases

@testset "dense arithmetic-spectrum Lasso family" begin
    base = generate_family_base(24, 6, 20260814)
    low = generate_instance(
        base,
        10.0;
        lambda_ratio = 1e-3,
        noise_std = 1e-3,
    )
    high = generate_instance(
        base,
        1e4;
        lambda_ratio = 1e-3,
        noise_std = 1e-3,
    )
    low_check = verify_instance(low)
    high_check = verify_instance(high)

    @test low.A isa Matrix{Float64}
    @test count(!iszero, low.A) == 24^2
    @test low_check.symmetry_error <= 1e-14
    @test low_check.condition_relative_error <= 1e-12
    @test high_check.condition_relative_error <= 1e-9
    @test high_check.eigenvalue_spacing_error <= 1e-12
    @test maximum(abs.(diff(low.eigenvalues) .- first(diff(low.eigenvalues)))) <= 1e-14
    @test first(low.eigenvalues) == 1.0
    @test last(low.eigenvalues) == 0.1
    @test low.xstar == high.xstar
    @test low.support == high.support
    @test instance_hashes(low).matrix != instance_hashes(high).matrix
    @test isfinite(lasso_metrics(low, low.xstar).normalized_stationarity)
end

@testset "formal campaign configuration" begin
    config = TOML.parsefile(joinpath(@__DIR__, "illcondition_lasso.toml"))
    @test config["generator_version"] == GENERATOR_VERSION
    @test config["dimension"] == 1000
    @test config["matrix_storage"] == "dense_float64"
    @test config["eigenvalue_spacing"] == "arithmetic"
    @test config["condition_numbers"] == [10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    @test config["julia_version"] == "1.10.4"
    @test config["tolerance"] == 1e-6
    @test config["solver_policy"]["solvers"] == [
        "cuclarabel",
        "cuscs",
        "cupdcs",
    ]
    @test !config["solver_policy"]["save_solution_vectors"]
end
