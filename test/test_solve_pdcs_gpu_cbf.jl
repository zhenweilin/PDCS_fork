using Test

include(joinpath(@__DIR__, "solve_pdcs_gpu_cbf.jl"))

const GPU_TEST_FIXTURE = normpath(joinpath(
    @__DIR__,
    "..",
    "..",
    "..",
    "test_data",
    "small_scale",
    "gptest.cbf.gz",
))

@testset "PDCS GPU single-CBF CLI" begin
    options = parse_gpu_arguments([
        "--input-file",
        GPU_TEST_FIXTURE,
    ])
    @test options.input_file == GPU_TEST_FIXTURE
    @test options.tolerance == 1e-6
    @test options.time_limit == 3600.0
    @test options.verbose == 2
    @test options.workers == Threads.nthreads()

    overridden = parse_gpu_arguments([
        "--input_file=$GPU_TEST_FIXTURE",
        "--tolerance",
        "1e-5",
        "--time-limit",
        "15",
        "--verbose",
        "0",
        "--workers",
        "3",
    ])
    @test overridden.tolerance == 1e-5
    @test overridden.time_limit == 15.0
    @test overridden.verbose == 0
    @test overridden.workers == 3
    @test_throws ErrorException parse_gpu_arguments(String[])
    @test_throws ErrorException parse_gpu_arguments([
        "--input-file",
        GPU_TEST_FIXTURE,
        "--workers",
        "0",
    ])
end

@testset "PDCS GPU delegates solving to JumpRW" begin
    options = parse_gpu_arguments([
        "--input-file",
        GPU_TEST_FIXTURE,
        "--workers",
        "2",
    ])
    captured = Ref{Any}()
    synchronizations = Ref(0)
    fake_solve_cbf = function (path, optimizer; kwargs...)
        captured[] = (; path, optimizer, kwargs=(; kwargs...))
        return :gpu_model
    end
    fake_factory = _ -> :pdcs_gpu_factory
    fake_synchronize = () -> (synchronizations[] += 1)

    model, elapsed = solve_gpu_model(
        options;
        solve_cbf=fake_solve_cbf,
        make_optimizer=fake_factory,
        synchronize=fake_synchronize,
    )

    @test model == :gpu_model
    @test elapsed >= 0
    @test captured[].path == GPU_TEST_FIXTURE
    @test captured[].optimizer == :pdcs_gpu_factory
    @test captured[].kwargs.fast_path == :auto
    @test captured[].kwargs.workers == 2
    @test captured[].kwargs.add_bridges === true
    @test synchronizations[] == 2
end
