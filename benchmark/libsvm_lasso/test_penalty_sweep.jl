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
