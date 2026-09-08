using Test
using SparseArrays

include(joinpath(@__DIR__, "run_penalty_sweep.jl"))

@testset "penalty sweep modeling path" begin
    @test DEFAULT_PDCS_ROOT == normpath(joinpath(@__DIR__, "..", ".."))
    options = parse_options([
        "--dataset", "news20",
        "--mode", "build",
        "--modeling", "bulk",
        "--workers", "2",
    ])
    @test options["modeling"] == "bulk"
    @test options["workers"] == "2"
    @test DEFAULT_DATASET_IDS == ("news20", "E2006-log1p", "rcv1-train")
    @test ALPHAS == [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    @test resolve_compact_zero_columns("auto")
    @test resolve_compact_zero_columns("true")
    @test !resolve_compact_zero_columns("false")
    @test safe_run_tag("slurm/123 bad") == "slurm_123_bad"
    @test safe_run_tag("   ") == "local"

    A = SparseMatrixCSC{Float32, Int32}(sparse(
        [1, 2, 2],
        [1, 1, 2],
        Float32[1, -2, 3],
        2,
        2,
    ))
    data = LassoData("tiny", A, Float32[1, -1], 3.0, 2, nothing)
    bulk = build_experiment_representation(
        data;
        modeling = "bulk",
        penalty_ratio = 0.1,
        workers = 2,
    )
    jump = build_experiment_representation(
        data;
        modeling = "jump",
        penalty_ratio = 0.1,
        workers = 2,
    )
    @test bulk isa LassoConicData
    @test jump isa LassoSOCPModel
    @test bulk.penalty ≈ jump.lambda ≈ 0.3
    @test_throws ArgumentError build_experiment_representation(
        data;
        modeling = "invalid",
        penalty_ratio = 0.1,
        workers = 1,
    )
end

@testset "penalty sweep resume metadata" begin
    mktempdir() do directory
        path = joinpath(directory, "resume.toml")
        expected = Dict{String,Any}(
            "dataset" => "tiny",
            "mode" => "pdcs-gpu",
            "modeling" => "bulk",
            "index_type" => "int32",
            "time_limit_seconds" => 3600.0,
            "relative_tolerance" => 1e-6,
            "absolute_tolerance" => 1e-6,
            "alphas" => [0.1, 1.0],
            "runs" => Any[
                Dict("alpha" => 0.1, "termination_status" => "OPTIMAL"),
                Dict("alpha" => 1.0, "termination_status" => "SCRIPT_ERROR"),
            ],
        )
        atomic_toml_write(path, expected)
        completed = completed_resume_runs(path, expected, [0.1, 1.0])
        @test collect(keys(completed)) == [0.1]
        @test completed[0.1]["termination_status"] == "OPTIMAL"

        mismatched = copy(expected)
        mismatched["relative_tolerance"] = 1e-5
        @test_throws ErrorException completed_resume_runs(
            path,
            mismatched,
            [0.1, 1.0],
        )
    end
end
