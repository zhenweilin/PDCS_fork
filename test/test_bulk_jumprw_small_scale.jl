using Test

const SMALL_SCALE_JUMPRW_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SMALL_SCALE_ROOT = get(
    ENV,
    "JUMPRW_SMALL_SCALE_ROOT",
    joinpath(SMALL_SCALE_JUMPRW_ROOT, "test_data", "small_scale"),
)

SMALL_SCALE_JUMPRW_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, SMALL_SCALE_JUMPRW_ROOT)

using JumpRW
using JuMP
using PDCS: PDCS_CPU
import MathOptInterface as MOI

include(joinpath(@__DIR__, "small_scale_helpers.jl"))

@testset "PDCS bulk handoff covers every small-scale CBF" begin
    files = collect_cbf_paths(SMALL_SCALE_ROOT)
    @test !isempty(files)

    passed = String[]
    failed = Pair{String,String}[]
    for (index, path) in enumerate(files)
        try
            record = verify_small_scale_instance(
                path;
                workers=min(4, Threads.nthreads()),
            )
            push!(passed, path)
            println(
                "PASS $index/$(length(files)) $(basename(path)) " *
                "generic=$(record.generic.termination) " *
                "bulk=$(record.bulk.termination)",
            )
        catch error
            message = sprint(showerror, error, catch_backtrace())
            push!(failed, path => message)
            println("FAIL $index/$(length(files)) $(basename(path)): $message")
        end
        flush(stdout)
    end

    println("SMALL_SCALE_DISCOVERED=$(length(files))")
    println("SMALL_SCALE_PASSED=$(length(passed))")
    println("SMALL_SCALE_FAILED=$(length(failed))")
    foreach(item -> println("FAIL $(item.first): $(item.second)"), failed)
    @test isempty(failed)
end
