using Test
include("../benchmark/rebuttal/soc_divergence_cases.jl")
using .SOCDivergenceCases

@testset "SOC divergence generators are deterministic" begin
    a=generate_candidates(256,10,2026;family=:positive)
    b=generate_candidates(256,10,2026;family=:positive)
    c=generate_candidates(256,10,2027;family=:positive)
    @test isequal(a.x,b.x)
    @test isequal(a.diagonal,b.diagonal)
    @test permutation_hash(a)==permutation_hash(b)
    @test permutation_hash(a)!=permutation_hash(c)
    @test all(abs.(prod(a.diagonal[2:end,:];dims=1).-1).<1e-12)
    @test all((1e-3 .<= a.diagonal) .& (a.diagonal .<= 1e3))
end

@testset "exact grouped/interleaved layouts" begin
    source=generate_branch_case(256,10,2026)
    groups=[findall(==(Int8(c)),source.construction_class) for c in 1:4]
    grouped,interleaved,g,i,ginv,iinv=paired_layouts(source,groups)
    @test sort(grouped.ids)==sort(interleaved.ids)
    @test grouped.ids[ginv]==source.ids
    @test interleaved.ids[iinv]==source.ids
    @test all(length(unique(grouped.construction_class[j:j+31]))==1 for j in 1:32:256)
    @test all(length(unique(interleaved.construction_class[j:j+31]))==4 for j in 1:32:256)
    @test validate_pair(grouped,interleaved)
end

@testset "parametric-similar generator" begin
    for delta in (0.0,1e-4,1e-3,1e-2)
        case=generate_parametric_case(128,10,2026,delta)
        @test all(isfinite,case.x)
        @test all(abs.(prod(case.diagonal[2:end,:];dims=1).-1).<1e-12)
    end
end
