using Test

const SCRIPT = joinpath(@__DIR__, "large_scale_lasso.jl")
const SOURCE = read(SCRIPT, String)

@testset "safe inclusion" begin
    @test occursin("abspath(PROGRAM_FILE) == @__FILE__", SOURCE)
end

if occursin("abspath(PROGRAM_FILE) == @__FILE__", SOURCE)
    include(SCRIPT)

    @testset "manifest presets and seeds" begin
        pilot = make_manifest("pilot", UInt64(20260728))
        @test pilot["schema_version"] == 1
        @test pilot["preset"] == "pilot"
        @test length(pilot["instances"]) == 5
        @test all(
            entry["m"] == 100 &&
            entry["n"] == 1000 &&
            entry["density"] == 1e-2
            for entry in pilot["instances"]
        )
        @test length(unique(entry["seed"] for entry in pilot["instances"])) == 5
        @test pilot == make_manifest("pilot", UInt64(20260728))

        table5 = make_manifest("table5", UInt64(20260728))
        @test length(table5["instances"]) == 25
        @test Set(
            (entry["m"], entry["n"], entry["density"])
            for entry in table5["instances"]
        ) == Set([
            (10_000, 100_000, 1e-4),
            (70_000, 700_000, 1e-4),
            (400_000, 7_000_000, 1e-4),
            (700_000, 7_000_000, 1e-4),
            (750_000, 7_500_000, 1e-4),
        ])
        @test length(unique(entry["seed"] for entry in table5["instances"])) == 25

        mktempdir() do dir
            path = joinpath(dir, "manifest.toml")
            write_manifest(path, pilot)
            @test read_manifest(path) == pilot
        end
    end

    @testset "deterministic instance generation" begin
        manifest = make_manifest("pilot", UInt64(20260728))
        first_entry = manifest["instances"][1]
        second_entry = manifest["instances"][2]
        data1 = generate_instance(first_entry)
        data2 = generate_instance(first_entry)
        data3 = generate_instance(second_entry)

        @test data1.A == data2.A
        @test data1.x_feat == data2.x_feat
        @test data1.b == data2.b
        @test reinterpret(UInt64, data1.lambda) == reinterpret(UInt64, data2.lambda)
        @test numerical_digest(data1) == numerical_digest(data2)
        @test numerical_digest(data1) != numerical_digest(data3)

        arrays1 = model_arrays(data1)
        arrays2 = model_arrays(data2)
        @test model_digest(arrays1) == model_digest(arrays2)
        @test output_filename(first_entry) ==
              "lasso_pilot-r01_seed$(first_entry["seed"]).cbf.gz"
    end

    @testset "pilot build and verification" begin
        mktempdir() do dir
            manifest = make_manifest("pilot", UInt64(20260728))
            config_path = joinpath(dir, "pilot.toml")
            results_path = joinpath(dir, "source_results.toml")
            source_dir = joinpath(dir, "source")
            write_manifest(config_path, manifest)

            results = build_instances(
                manifest;
                output_dir = source_dir,
                results_path,
            )
            @test length(results["results"]) == 5
            @test length(filter(endswith(".cbf.gz"), readdir(source_dir))) == 5
            @test all(length(result["numerical_digest"]) == 64 for result in results["results"])
            @test all(length(result["model_digest"]) == 64 for result in results["results"])

            skipped = build_instances(
                manifest;
                output_dir = source_dir,
                results_path,
            )
            @test [
                comparable_result(result) for result in skipped["results"]
            ] == [
                comparable_result(result) for result in results["results"]
            ]

            @test verify_instances(
                manifest,
                results;
                output_dir = joinpath(dir, "verified"),
            )

            bad_reference = deepcopy(results)
            bad_reference["results"][1]["numerical_digest"] = repeat("0", 64)
            @test_throws ErrorException verify_instances(
                manifest,
                bad_reference;
                output_dir = joinpath(dir, "mismatch"),
            )
        end
    end

    @testset "CLI parsing" begin
        @test parse_cli(["--help"])[1] == "--help"
        command, options = parse_cli([
            "generate-config",
            "--preset", "pilot",
            "--master-seed", "20260728",
            "--config", "pilot.toml",
        ])
        @test command == "generate-config"
        @test options["preset"] == "pilot"
        @test options["master-seed"] == "20260728"
        @test_throws ErrorException parse_cli(["unknown"])
    end
end
