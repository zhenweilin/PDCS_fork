using Test
using CUDA
include("../benchmark/rebuttal/timing.jl")
using .RebuttalTiming

@testset "CUDA event timing" begin
    if CUDA.functional()
        x=CUDA.zeros(Float64,1_000_000)
        elapsed=cuda_event_time(() -> fill!(x,1.0))
        @test elapsed > 0
        @test bandwidth_lower_bound(length(x),elapsed) > 0
    else
        @test_skip CUDA.functional()
    end
end
