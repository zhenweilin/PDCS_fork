using Test
using CUDA
using Base.Threads

Base.include(Main, joinpath(@__DIR__, "..", "src", "pdcs_gpu", "PDCS_GPU.jl"))

CUDA.functional() || error("A functional CUDA device is required")
expected_gpu = get(ENV, "PDCS_EXPECTED_GPU", "H100")
occursin(expected_gpu, CUDA.name(CUDA.device())) || error(
    "This robustness test must run on $expected_gpu; found " *
    "$(CUDA.name(CUDA.device()))",
)

function ordinary_soc_arguments(input; alias_workspace::Bool = false)
    dimension = length(input)
    solution = CuArray(Float64.(input))
    dummy = CUDA.zeros(Float64, dimension)
    workspace = alias_workspace ? reshape(solution, :) : CUDA.zeros(Float64, dimension)
    warm_start = CUDA.zeros(Float64, 1)
    starts = Int64[0]
    sizes = Int64[dimension]
    types = Int64[20]
    arguments = (
        solution, dummy, dummy, dummy, dummy, dummy, workspace, warm_start,
        starts, CuArray(sizes), sizes, Int64(1), types, 1e-12, 1e-12,
    )
    return solution, arguments
end

function project_and_check!(solution, arguments, expected; repetitions = 25)
    for repetition in 1:repetitions
        PDCS_GPU.gridWise_block_proj(arguments...)
        projected = Array(solution)
        error = maximum(abs, projected .- expected)
        @test all(isfinite, projected)
        @test error <= 64eps(Float64)
    end
end

function exp_projection(
    strategy::Symbol, input, projection_type;
    diagonal = ones(Float64, 3), alias_workspace = false,
)
    solution = CuArray(Float64.(input))
    bl = CUDA.fill(-Inf, 3)
    bu = CUDA.fill(Inf, 3)
    d = CuArray(Float64.(diagonal))
    d2 = d .* d
    d_times_x = CUDA.zeros(Float64, 3)
    workspace = alias_workspace ? reshape(solution, :) : CUDA.zeros(Float64, 3)
    warm = CUDA.ones(Float64, 1)
    starts = Int64[0]
    sizes = Int64[3]
    types = Int64[projection_type]
    if strategy === :gridWise
        PDCS_GPU.gridWise_block_proj(
            solution, bl, bu, d, d2, d_times_x, workspace, warm,
            starts, CuArray(sizes), sizes, Int64(1), types, 1e-12, 1e-12,
        )
    elseif strategy === :threadWise
        PDCS_GPU.threadWise_block_proj(
            solution, bl, bu, d, d2, d_times_x, workspace, warm,
            CuArray(starts), CuArray(sizes), Int64(1), CuArray(types),
            1e-12, 1e-12,
        )
    else
        error("unknown projection strategy $strategy")
    end
    return Array(solution), (
        solution, bl, bu, d, d2, d_times_x, workspace, warm, starts,
        CuArray(sizes), sizes, Int64(1), types, 1e-12, 1e-12,
    )
end

function exp_cpu_reference(input, dual::Bool)
    if dual
        negative_projection = -Float64.(input)
        PDCS_GPU.exponent_proj_cpu!(negative_projection, 1e-12)
        return Float64.(input) .+ negative_projection
    end
    projection = Float64.(input)
    PDCS_GPU.exponent_proj_cpu!(projection, 1e-12)
    return projection
end

function generic_projection(
    strategy::Symbol, input, projection_type;
    lower = fill(-Inf, length(input)), upper = fill(Inf, length(input)),
    diagonal = ones(Float64, length(input)), alias_workspace = false,
)
    dimension = length(input)
    solution = CuArray(Float64.(input))
    bl = CuArray(Float64.(lower))
    bu = CuArray(Float64.(upper))
    d = CuArray(Float64.(diagonal))
    d2 = d .* d
    d_times_x = CUDA.zeros(Float64, dimension)
    workspace = alias_workspace ? reshape(solution, :) : CUDA.zeros(Float64, dimension)
    warm = CUDA.ones(Float64, 1)
    starts = Int64[0]
    sizes = Int64[dimension]
    types = Int64[projection_type]
    if strategy === :gridWise
        arguments = (
            solution, bl, bu, d, d2, d_times_x, workspace, warm, starts,
            CuArray(sizes), sizes, Int64(1), types, 1e-12, 1e-12,
        )
        PDCS_GPU.gridWise_block_proj(arguments...)
        return Array(solution), arguments
    end
    PDCS_GPU.threadWise_block_proj(
        solution, bl, bu, d, d2, d_times_x, workspace, warm,
        CuArray(starts), CuArray(sizes), Int64(1), CuArray(types),
        1e-12, 1e-12,
    )
    return Array(solution), nothing
end

@testset "strict native grid-wise runtime on $expected_gpu" begin
    runtime = PDCS_GPU.check_gridWise_runtime!()
    @test runtime.mode == "native"
    @test runtime.native_enabled
    @test runtime.state == :passed
    @test runtime.abi_version == 2
    @test runtime.required_abi_version == 2
    @test runtime.configuration !== nothing
    @test runtime.configuration.atomics_mode == 0
    @test runtime.configuration.math_mode & 2 == 2
    @test runtime.configuration.math_mode & 16 == 16
end

@testset "repeated and alias-safe ordinary SOC projection" begin
    for dimension in (3, 10_002)
        tail_norm = 0.8
        tail = fill(tail_norm / sqrt(dimension - 1), dimension - 1)
        cases = (
            (input = [0.9; tail], expected = [0.9; tail]),
            (input = [0.2; tail], expected = [0.5; (0.5 / tail_norm) .* tail]),
            (input = [-0.9; tail], expected = zeros(Float64, dimension)),
        )
        for case in cases, alias_workspace in (false, true)
            solution, arguments = ordinary_soc_arguments(
                case.input; alias_workspace = alias_workspace,
            )
            project_and_check!(solution, arguments, case.expected)
        end
    end
end

@testset "every simple grid-wise and thread-wise projection code" begin
    input = [-2.0, 0.5, 3.0]
    lower = [-1.0, 0.0, 1.0]
    upper = [1.0, 1.0, 2.0]
    expectations = Dict(
        0 => input, 1 => input, 2 => zeros(3),
        3 => [0.0, 0.5, 3.0], 4 => [0.0, 0.5, 3.0],
        17 => [-1.0, 0.5, 2.0], 18 => [-1.0, 0.5, 2.0],
        19 => [-1.0, 0.5, 2.0],
    )
    for projection_type in sort!(collect(keys(expectations)))
        expected = expectations[projection_type]
        for strategy in (:gridWise, :threadWise)
            result, _ = generic_projection(
                strategy, input, projection_type;
                lower = lower, upper = upper,
                alias_workspace = strategy === :gridWise,
            )
            @test result == expected
        end
    end
end

@testset "grid-wise ignores short inactive auxiliary placeholders" begin
    input_parts = (
        [1.0, -2.0, 3.0], [-1.0, 2.0, -3.0], [4.0, -5.0, 6.0],
        [-2.0, 0.5, 3.0], [-7.0, 8.0, -9.0],
    )
    input = reduce(vcat, input_parts)
    expected = reduce(vcat, (
        input_parts[1], [0.0, 2.0, 0.0], zeros(3),
        [-1.0, 0.5, 2.0], [0.0, 8.0, 0.0],
    ))
    solution = CuArray(input)
    lower = CuArray(repeat([-1.0, 0.0, 1.0], length(input_parts)))
    upper = CuArray(repeat([1.0, 1.0, 2.0], length(input_parts)))
    placeholder = CUDA.ones(Float64, 1)
    starts = Int64.(0:3:length(input)-3)
    sizes = fill(Int64(3), length(input_parts))
    types = Int64[0, 3, 2, 17, 4]
    PDCS_GPU.gridWise_block_proj(
        solution, lower, upper, placeholder, placeholder, placeholder,
        placeholder, CUDA.zeros(Float64, length(types)), starts,
        CuArray(sizes), sizes, Int64(length(types)), types, 1e-12, 1e-12,
    )
    @test Array(solution) == expected
end

@testset "simple projection codes work beyond the first two thread-wise blocks" begin
    input_parts = (
        [1.0, -2.0, 3.0], [-1.0, 2.0, -3.0], [4.0, -5.0, 6.0],
        [-2.0, 0.5, 3.0], [-7.0, 8.0, -9.0],
    )
    expected_parts = (
        input_parts[1], [0.0, 2.0, 0.0], zeros(3),
        [-1.0, 0.5, 2.0], [0.0, 8.0, 0.0],
    )
    input = reduce(vcat, input_parts)
    expected = reduce(vcat, expected_parts)
    lower = repeat([-1.0, 0.0, 1.0], length(input_parts))
    upper = repeat([1.0, 1.0, 2.0], length(input_parts))
    starts = Int64.(0:3:length(input)-3)
    sizes = fill(Int64(3), length(input_parts))
    types = Int64[0, 3, 2, 17, 4]
    for strategy in (:gridWise, :threadWise)
        solution = CuArray(input)
        bl = CuArray(lower)
        bu = CuArray(upper)
        dummy = CUDA.ones(Float64, length(input))
        temp = CUDA.zeros(Float64, length(input))
        warm = CUDA.zeros(Float64, length(input_parts))
        if strategy === :gridWise
            PDCS_GPU.gridWise_block_proj(
                solution, bl, bu, dummy, dummy, dummy, temp, warm, starts,
                CuArray(sizes), sizes, Int64(length(types)), types,
                1e-12, 1e-12,
            )
        else
            PDCS_GPU.threadWise_block_proj(
                solution, bl, bu, dummy, dummy, dummy, temp, warm,
                CuArray(starts), CuArray(sizes), Int64(length(types)),
                CuArray(types), 1e-12, 1e-12,
            )
        end
        @test Array(solution) == expected
    end
end

@testset "every ordinary SOC grid-wise and thread-wise projection code" begin
    dimension = 10_002
    tail_norm = 0.8
    tail = fill(tail_norm / sqrt(dimension - 1), dimension - 1)
    cases = (
        ([0.9; tail], [0.9; tail]),
        ([0.2; tail], [0.5; (0.5 / tail_norm) .* tail]),
        ([-0.9; tail], zeros(Float64, dimension)),
    )
    for projection_type in (5, 6, 7, 20, 21, 22), (input, expected) in cases
        scale = max(1.0, maximum(abs, expected))
        for strategy in (:gridWise, :threadWise)
            result, _ = generic_projection(
                strategy, input, projection_type;
                alias_workspace = strategy === :gridWise,
            )
            @test all(isfinite, result)
            @test maximum(abs, result .- expected) / scale <= 1e-12
        end
    end
end

@testset "multiple cones in one native call" begin
    dimension = 4098
    tail_norm = 0.8
    tail = fill(tail_norm / sqrt(dimension - 1), dimension - 1)
    inputs = ([0.9; tail], [0.2; tail], [-0.9; tail])
    expected_parts = (
        inputs[1],
        [0.5; (0.5 / tail_norm) .* tail],
        zeros(Float64, dimension),
    )
    input = reduce(vcat, inputs)
    expected = reduce(vcat, expected_parts)
    solution = CuArray(input)
    dummy = CUDA.zeros(Float64, length(input))
    warm = CUDA.zeros(Float64, length(inputs))
    starts = Int64.(0:dimension:length(input)-dimension)
    sizes = fill(Int64(dimension), length(inputs))
    types = fill(Int64(20), length(inputs))
    arguments = (
        solution, dummy, dummy, dummy, dummy, dummy, reshape(solution, :), warm,
        starts, CuArray(sizes), sizes, Int64(length(inputs)), types, 1e-12, 1e-12,
    )
    project_and_check!(solution, arguments, expected)
end

@testset "grid-wise and thread-wise exponential cone projections" begin
    inputs = (
        [0.0, 1.0, 2.0],       # interior
        [1.0, 0.0, -1.0],      # polar branch
        [0.0, 0.0, 1.0],       # boundary
        [1.0, 1.0, 0.05],      # positive root search
        [1.0, -0.2, -0.5],     # negative root search
        [1e-12, 1e-14, 1e-16], # near-degenerate
    )
    primal_codes = (13, 14, 15, 26, 27)
    dual_codes = (11, 12, 16, 28, 29)
    for (dual, codes) in ((false, primal_codes), (true, dual_codes))
        for projection_type in codes, input in inputs
            expected = exp_cpu_reference(input, dual)
            thread_result, _ = exp_projection(
                :threadWise, input, projection_type,
            )
            scale = max(1.0, maximum(abs, expected))
            @test all(isfinite, thread_result)
            @test maximum(abs, thread_result .- expected) / scale <= 5e-7
            for alias_workspace in (false, true)
                grid_result, arguments = exp_projection(
                    :gridWise, input, projection_type;
                    alias_workspace = alias_workspace,
                )
                @test all(isfinite, grid_result)
                @test maximum(abs, grid_result .- expected) / scale <= 5e-7
                PDCS_GPU.gridWise_block_proj(arguments...)
                repeated = Array(arguments[1])
                @test maximum(abs, repeated .- grid_result) / scale <= 5e-7
            end
        end
    end

    # Non-identity diagonal scaling exercises the grid-wise inversion kernel;
    # compare it to the independently launched thread-wise implementation.
    diagonal = [0.5, 2.0, 3.0]
    for projection_type in (27, 29), input in inputs
        thread_result, _ = exp_projection(
            :threadWise, input, projection_type; diagonal = diagonal,
        )
        grid_result, arguments = exp_projection(
            :gridWise, input, projection_type; diagonal = diagonal,
            alias_workspace = true,
        )
        scale = max(1.0, maximum(abs, thread_result))
        @test all(isfinite, thread_result)
        @test all(isfinite, grid_result)
        @test maximum(abs, grid_result .- thread_result) / scale <= 5e-7
        PDCS_GPU.gridWise_block_proj(arguments...)
        repeated = Array(arguments[1])
        @test maximum(abs, repeated .- grid_result) / scale <= 5e-7
    end
end

@testset "shared native handle is serialized across Julia tasks" begin
    device = CUDA.device()
    tasks = map(1:max(4, min(8, Threads.nthreads()))) do task_index
        Threads.@spawn begin
            CUDA.device!(device)
            dimension = 2050 + 2task_index
            tail_norm = 0.8
            tail = fill(tail_norm / sqrt(dimension - 1), dimension - 1)
            expected = [0.5; (0.5 / tail_norm) .* tail]
            solution, arguments = ordinary_soc_arguments(
                [0.2; tail]; alias_workspace = isodd(task_index),
            )
            project_and_check!(solution, arguments, expected; repetitions = 10)
            true
        end
    end
    @test all(fetch, tasks)
end

@testset "handle teardown and recreation" begin
    prior_handle = PDCS_GPU._gridWise_cublas_handle[]
    @test prior_handle !== nothing
    for _ in 1:3
        PDCS_GPU.release_gridWise_cublas_handle!()
        @test PDCS_GPU._gridWise_cublas_handle[] === nothing
        solution, arguments = ordinary_soc_arguments([0.0, 2.0, 0.0])
        project_and_check!(solution, arguments, [1.0, 1.0, 0.0]; repetitions = 2)
        @test PDCS_GPU._gridWise_cublas_handle[] !== nothing
        @test PDCS_GPU._gridWise_cublas_handle[].handle != C_NULL
    end
end

@testset "bad layouts fail before output mutation" begin
    original = [0.0, 2.0, 0.0]
    solution, arguments = ordinary_soc_arguments(original)

    bad_types = Base.setindex(arguments, Int64[99], 13)
    @test_throws ArgumentError PDCS_GPU.gridWise_block_proj(bad_types...)
    @test Array(solution) == original

    overlapping_starts = Base.setindex(arguments, Int64[1], 9)
    @test_throws DimensionMismatch PDCS_GPU.gridWise_block_proj(overlapping_starts...)
    @test Array(solution) == original

    zero_tolerance = Base.setindex(arguments, 0.0, 14)
    @test_throws ArgumentError PDCS_GPU.gridWise_block_proj(zero_tolerance...)
    @test Array(solution) == original

    dummy = CUDA.zeros(Float64, length(original))
    @test_throws ArgumentError PDCS_GPU.threadWise_block_proj(
        solution, dummy, dummy, dummy, dummy, dummy, dummy,
        CUDA.zeros(Float64, 1), CuArray(Int64[0]), CuArray(Int64[3]),
        Int64(1), CuArray(Int64[99]), 1e-12, 1e-12,
    )
    @test Array(solution) == original

    empty_float = CUDA.zeros(Float64, 0)
    empty_int = CUDA.zeros(Int64, 0)
    @test isnothing(PDCS_GPU.threadWise_block_proj(
        empty_float, empty_float, empty_float, empty_float, empty_float,
        empty_float, empty_float, empty_float, empty_int, empty_int,
        Int64(0), empty_int, 1e-12, 1e-12,
    ))
end

runtime = PDCS_GPU.gridWise_runtime_status()
println("GRIDWISE_ROBUSTNESS_STATUS=$runtime")
println("GRIDWISE_ROBUSTNESS_PASS gpu=$(CUDA.name(CUDA.device())) threads=$(Threads.nthreads())")
