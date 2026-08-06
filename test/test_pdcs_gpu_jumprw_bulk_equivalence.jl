using Test

const GPU_BULK_PDCS_ROOT = normpath(joinpath(@__DIR__, ".."))
const GPU_BULK_JUMPRW_ROOT = joinpath(GPU_BULK_PDCS_ROOT, "external", "Jump_RW")
const GPU_BULK_FIXTURE_ROOT = joinpath(GPU_BULK_PDCS_ROOT, "benchmark", "represent_data")

GPU_BULK_JUMPRW_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, GPU_BULK_JUMPRW_ROOT)

using CUDA
using JumpRW
using JuMP
using PDCS: PDCS_GPU
import MathOptInterface as MOI

# Reuse the strict CPU bulk-equivalence checks with the GPU optimizer module.
# The helper compares the canonical cache and complete primal/dual solutions.
const PDCS_CPU = PDCS_GPU
include(joinpath(@__DIR__, "small_scale_helpers.jl"))

const GPU_BULK_DEFAULT_CASES = (
    "bss1.cbf.gz",
    "gptest.cbf.gz",
    "pp-n10-d10.cbf.gz",
)

function gpu_bulk_cases(arguments)
    names = isempty(arguments) ? GPU_BULK_DEFAULT_CASES : arguments
    return [
        isabspath(name) ? normpath(name) : joinpath(GPU_BULK_FIXTURE_ROOT, name)
        for name in names
    ]
end

CUDA.functional() || error("CUDA is not functional for the cuPDCS bulk test")
println("CUDA_DEVICE=$(CUDA.name(CUDA.device()))")

@testset "JumpRW bulk CBF handoff matches generic MOI on cuPDCS" begin
    for path in gpu_bulk_cases(ARGS)
        @test isfile(path)
        record = verify_small_scale_instance(
            path;
            workers=min(4, Threads.nthreads()),
        )
        @test record.generic.termination == record.bulk.termination
        @test record.generic.primal_status == record.bulk.primal_status
        @test record.generic.dual_status == record.bulk.dual_status
        @test isapprox(
            record.generic.objective,
            record.bulk.objective;
            atol=1e-4,
            rtol=1e-4,
        )
        println(
            "PASS $(basename(path)) " *
            "generic=$(record.generic.termination) " *
            "bulk=$(record.bulk.termination) " *
            "objective=$(record.bulk.objective)",
        )
        flush(stdout)
        GC.gc(true)
        CUDA.reclaim()
    end
end
