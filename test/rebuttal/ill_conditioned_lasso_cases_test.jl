using Test
using LinearAlgebra
using SparseArrays

include("../../benchmark/rebuttal/ill_conditioned_lasso_cases.jl")
using .IllConditionedLassoCases

@testset "exact sparse ill-conditioned Lasso generator" begin
    for K in (1.0,1e2,1e4,1e6)
        inst=generate_instance(1000,5000,40,10,K,2026)
        v=verify_instance(inst)
        @test isapprox(v.kappa_measured,K;rtol=1e-8,atol=1e-8)
        @test v.kkt_active <= 1e-9
        @test v.kkt_inactive <= 1e-11
        @test v.normalized_stationarity <= 1e-10
        @test v.column_norm_error <= 1e-12
        @test nnz(inst.A) <= 40*15 + (5000-40)*11
    end
    a=generate_instance(1000,5000,40,10,1e4,2026)
    b=generate_instance(1000,5000,40,10,1e4,2026)
    c=generate_instance(1000,5000,40,10,1e4,2027)
    @test instance_hashes(a)==instance_hashes(b)
    @test instance_hashes(a)!=instance_hashes(c)
    fixed=generate_instance(1000,5000,40,10,1e4,2026;panel=:fixed_residual,residual_norm=1)
    @test isapprox(norm(fixed.rstar),1.0;rtol=1e-12,atol=1e-12)
end
