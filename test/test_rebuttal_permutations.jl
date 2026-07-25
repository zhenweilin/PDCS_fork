using Test
include("../benchmark/rebuttal/common.jl")
using .RebuttalCommon

@testset "paired cone permutations" begin
    classes=repeat(1:4,32)
    for layout in (:grouped,:random,:interleaved)
        p=permutation(classes,layout;seed=2026)
        @test verify_permutation(p,length(classes))
        inv=inverse_permutation(p)
        @test inv[p] == collect(eachindex(p))
        @test sort(classes[p]) == sort(classes)
    end
    @test permutation(classes,:random;seed=2026) ==
          permutation(classes,:random;seed=2026)
end
