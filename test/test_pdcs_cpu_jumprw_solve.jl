"""
Run from the JumpRW repository root after instantiating the PDCS environment:

    julia --startup-file=no --project=external/PDCS_fork -e \
        'using Pkg; Pkg.instantiate()'
    JULIA_NUM_THREADS=4 julia --startup-file=no \
        --project=external/PDCS_fork \
        external/PDCS_fork/test/test_pdcs_cpu_jumprw_solve.jl
"""

using Test

const RUNNER_PATH = joinpath(
    @__DIR__,
    "..",
    "benchmark",
    "multi_period_port_pdcs_cpu.jl",
)
include(RUNNER_PATH)

pushfirst!(LOAD_PATH, JUMPRW_ROOT)
using JumpRW
using JuMP
import MathOptInterface as MOI

# Loading the PDCS package entry point also loads PDCS_GPU and therefore needs
# a CUDA driver. This integration test exercises only the CPU implementation.
include(PDCS_CPU_SOURCE)

@testset "PDCS CPU solves a CBF through JumpRW" begin
    fixture = joinpath(
        JUMPRW_ROOT,
        "test_data",
        "small_scale",
        "gptest.cbf.gz",
    )
    @test isfile(fixture)

    options = (
        ;
        input_folder=dirname(fixture),
        output_folder="",
        tolerance=1e-6,
        time_limit=30.0,
        verbose=0,
        workers=min(4, Threads.nthreads()),
    )
    model, wall_seconds = solve_model(fixture, options)

    @test JuMP.solver_name(model) == "PDHG-CLP"
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.result_count(model) == 1
    @test JuMP.objective_value(model) ≈ -4.414286350786631 atol = 1e-5
    @test MOI.get(JuMP.backend(model), PDCS_CPU.PDHGIterations()) > 0
    @test haskey(model.ext, :JumpRW_CBF_timings)

    read_seconds = model.ext[:JumpRW_CBF_timings].total_seconds
    @test read_seconds > 0
    @test wall_seconds >= read_seconds
end
