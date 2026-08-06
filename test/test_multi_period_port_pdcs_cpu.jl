using Test

include(joinpath(
    @__DIR__,
    "..",
    "benchmark",
    "multi_period_port_pdcs_cpu.jl",
))

@testset "PDCS CPU CBF runner CLI" begin
    defaults = parse_arguments(String[])
    @test defaults.input_folder == normpath(joinpath(
        @__DIR__,
        "..",
        "..",
        "..",
        "test_data",
        "small_scale",
    ))
    @test defaults.tolerance == 1e-6
    @test defaults.time_limit === nothing
    @test defaults.verbose == 2
    @test defaults.workers == Threads.nthreads()

    overridden = parse_arguments([
        "--input-folder",
        defaults.input_folder,
        "--output-folder=/tmp/pdcs-cpu-logs",
        "--solver",
        "pdcs_cpu",
        "--tolerance",
        "1e-5",
        "--time-limit",
        "3",
        "--verbose",
        "0",
        "--workers",
        "2",
    ])
    @test overridden.output_folder == "/tmp/pdcs-cpu-logs"
    @test overridden.tolerance == 1e-5
    @test overridden.time_limit == 3.0
    @test overridden.verbose == 0
    @test overridden.workers == 2
    @test_throws ErrorException parse_arguments(["--workers", "0"])
    @test_throws ErrorException parse_arguments(["--solver", "cuPDCS"])
end

@testset "PDCS CPU CBF discovery" begin
    mktempdir() do directory
        for name in ("b.cbf.gz", "A.cbf", "c.cbf.bz2", "ignored.mps")
            write(joinpath(directory, name), "fixture")
        end
        @test basename.(cbf_files(directory)) == [
            "A.cbf",
            "b.cbf.gz",
            "c.cbf.bz2",
        ]
        @test log_path(directory, joinpath(directory, "b.cbf.gz")) ==
              joinpath(directory, "b.raw.log")
        @test log_path(directory, joinpath(directory, "c.cbf.bz2")) ==
              joinpath(directory, "c.raw.log")
    end
end

@testset "PDCS CPU delegates CBF solving to JumpRW" begin
    options = parse_arguments(["--time-limit", "1", "--workers", "2"])
    captured = Ref{Any}()
    fake_solve_cbf = function (path, optimizer; kwargs...)
        captured[] = (; path, optimizer, kwargs=(; kwargs...))
        return :solved_model
    end
    fake_factory = _ -> :pdcs_optimizer_factory

    model, elapsed = solve_model(
        "fixture.cbf.gz",
        options;
        solve_cbf=fake_solve_cbf,
        make_optimizer=fake_factory,
    )

    @test model == :solved_model
    @test elapsed >= 0
    @test captured[].path == "fixture.cbf.gz"
    @test captured[].optimizer == :pdcs_optimizer_factory
    @test captured[].kwargs.fast_path == :auto
    @test captured[].kwargs.workers == 2
    @test captured[].kwargs.add_bridges === true
end
