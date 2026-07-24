using Test

include(joinpath(@__DIR__, "..", "src", "pdcs_gpu", "projection_strategy.jl"))

@testset "GPU projection strategy heuristic" begin
    @test select_projection_strategy([8], [20]) == :gridWise
    @test select_projection_strategy([1, 1, 8], [2, 3, 20]) == :gridWise
    @test select_projection_strategy([1, 1, 8], [2, 3, 23]) == :blockWise
    @test select_projection_strategy(fill(32, 1_001), fill(20, 1_001)) == :warpWise
    @test select_projection_strategy(vcat([1, 1, 2_000], fill(32, 1_000)), fill(20, 1_003)) == :blockWise
    @test select_projection_strategy(fill(32, 60_001), fill(20, 60_001)) == :threadWise
    @test select_projection_strategy(vcat([1, 1, 150], fill(32, 60_000)), fill(20, 60_003)) == :warpWise
    @test_throws DimensionMismatch select_projection_strategy([2], [20, 20])
end
