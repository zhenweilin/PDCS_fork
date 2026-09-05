using Test
using CUDA
using LinearAlgebra

Base.include(Main, joinpath(@__DIR__, "..", "src", "pdcs_gpu", "PDCS_GPU.jl"))

CUDA.functional() || error("A functional CUDA device is required")

configuration = PDCS_GPU.gridWise_cublas_configuration()
println("GRID_SOC_CUBLAS_CONFIGURATION=$configuration")
@test configuration.atomics_mode == 0
@test configuration.math_mode & 2 == 2
@test configuration.math_mode & 16 == 16

function gridwise_soc_arguments(input; alias_workspace::Bool)
    dimension = length(input)
    solution = CuArray(input)
    dummy = CUDA.zeros(Float64, dimension)
    workspace = alias_workspace ? solution : CUDA.zeros(Float64, dimension)
    warm_start = CUDA.zeros(Float64, 1)
    starts = Int64[0]
    sizes = Int64[dimension]
    projection_types = Int64[20]
    arguments = (
        solution,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        workspace,
        warm_start,
        starts,
        CuArray(sizes),
        sizes,
        Int64(1),
        projection_types,
    )
    return solution, arguments
end

@testset "grid-wise SOC projection is correct and idempotent" begin
    dimension = 10_002
    tail_norm = 0.8
    tail = fill(tail_norm / sqrt(dimension - 1), dimension - 1)
    cases = (
        (name = "interior", input = [0.9; tail], expected = [0.9; tail]),
        (
            name = "exterior",
            input = [0.2; tail],
            expected = [0.5; (0.5 / tail_norm) .* tail],
        ),
        (
            name = "polar",
            input = [-0.9; tail],
            expected = zeros(Float64, dimension),
        ),
    )

    for case in cases, alias_workspace in (false, true)
        solution, arguments = gridwise_soc_arguments(
            case.input; alias_workspace = alias_workspace
        )
        for repetition in 1:3
            PDCS_GPU.gridWise_block_proj(arguments...)
            projected = Array(solution)
            error = maximum(abs, projected .- case.expected)
            println(
                "GRID_SOC_IDEMPOTENCE case=$(case.name) " *
                "alias_workspace=$alias_workspace repetition=$repetition " *
                "head=$(projected[1]) max_error=$error",
            )
            @test error <= 64eps(Float64)
        end
    end
end
