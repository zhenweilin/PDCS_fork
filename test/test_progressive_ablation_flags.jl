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

const EXPECTED_PROGRESSIVE_FLAGS = Dict(
    "pdhg" => (false, false, false, false, false, false),
    "pdhg_restart" => (false, false, false, true, false, false),
    "pdhg_restart_scaling" => (true, false, false, true, false, false),
    "pdhg_restart_scaling_reflection" =>
        (true, false, false, true, true, false),
    "pdhg_restart_scaling_reflection_adaptive_primal_weight" =>
        (true, false, true, true, true, false),
    "pdhg_restart_scaling_reflection_adaptive" =>
        (true, true, true, true, true, false),
)

@testset "progressive ablation flags" begin
    @test Set(PROGRESSIVE_CONFIGURATIONS) ==
          Set(keys(EXPECTED_PROGRESSIVE_FLAGS))
    for configuration in PROGRESSIVE_CONFIGURATIONS
        resolved = flags(configuration)
        observed = (
            resolved.use_scaling,
            resolved.use_adaptive_step,
            resolved.use_adaptive_step_size_weight,
            resolved.use_restart,
            resolved.use_reflection,
            resolved.use_halpern,
        )
        @test observed == EXPECTED_PROGRESSIVE_FLAGS[configuration]
    end
end

println("PROGRESSIVE_ABLATION_FLAGS_PASS")
