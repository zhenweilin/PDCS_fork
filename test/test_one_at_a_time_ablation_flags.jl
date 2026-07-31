#!/usr/bin/env julia

using Test

include(
    joinpath(
        @__DIR__,
        "..",
        "benchmark",
        "ablation",
        "run_ablation_case.jl",
    ),
)

const EXPECTED_ONE_AT_A_TIME_FLAGS = Dict(
    "full" => (true, true, true, true, true, false),
    "no_scaling" => (false, true, true, true, true, false),
    "no_adaptive_step" => (true, false, true, true, true, false),
    "no_adaptive_primal_weight" =>
        (true, true, false, true, true, false),
    "no_restart" => (true, true, true, false, true, false),
    "no_reflection" => (true, true, true, true, false, false),
)

@testset "core one-at-a-time ablation flags" begin
    for (configuration, expected) in EXPECTED_ONE_AT_A_TIME_FLAGS
        resolved = flags(configuration)
        observed = (
            resolved.use_scaling,
            resolved.use_adaptive_step,
            resolved.use_adaptive_step_size_weight,
            resolved.use_restart,
            resolved.use_reflection,
            resolved.use_halpern,
        )
        @test observed == expected
    end
end

println("ONE_AT_A_TIME_ABLATION_FLAGS_PASS")
