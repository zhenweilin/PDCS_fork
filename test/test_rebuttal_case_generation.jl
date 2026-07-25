using Test
include("../benchmark/rebuttal/generate_soc_cases.jl")
include("../benchmark/rebuttal/generate_exp_cases.jl")
using .GenerateSOCCases
using .GenerateExpCases

@testset "rebuttal case generation" begin
    a=ordinary_soc(8,10;sigma=2.0,seed=2026)
    b=ordinary_soc(8,10;sigma=2.0,seed=2026)
    c=ordinary_soc(8,10;sigma=1.0,seed=2026)
    @test a == b
    @test a == 2c
    @test length(a)==80

    soc=rescaled_soc(32,10;sigma_x=1,sigma_d=2,seed=2026)
    @test length(soc.x)==320
    @test all(isfinite,soc.x)
    @test all(1e-3 .<= soc.diagonal .<= 1e3)
    @test all(soc.diagonal[1:10:end] .== 1)

    exp1=random_exp(32;sigma_x=2,sigma_d=.5,seed=2026)
    exp2=random_exp(32;sigma_x=2,sigma_d=.5,seed=2026)
    @test exp1 == exp2
    @test length(exp1.x)==96
    @test all(1e-3 .<= exp1.diagonal .<= 1e3)
end
