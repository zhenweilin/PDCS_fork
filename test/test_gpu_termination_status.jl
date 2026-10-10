using Test

module TerminationStatusRegression

using Test

mutable struct PDHGCLPConvergeInfo
    l_inf_rel_primal_res::Float64
    l_inf_rel_dual_res::Float64
    rel_gap::Float64
    l_inf_abs_primal_res::Float64
    l_inf_abs_dual_res::Float64
    abs_gap::Float64
    status::Symbol
end

# These types are needed by method signatures in termination.jl.  The focused
# regression below exercises only PDHGCLPConvergeInfo.
struct PDHGCLPInfeaInfo end
struct PDHGCLPInfo end
struct PDHGCLPParameters end

include(joinpath(@__DIR__, "..", "src", "pdcs_gpu", "termination.jl"))

@testset "recomputed convergence status is not sticky" begin
    info = PDHGCLPConvergeInfo(
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
        :continue,
    )
    optimality_criteria_met(rel_tol = 1.0e-6, abs_tol = 1.0e-6, info = info)
    @test info.status == :optimal

    info.l_inf_rel_primal_res = 0.1
    info.l_inf_abs_primal_res = 0.1
    optimality_criteria_met(rel_tol = 1.0e-6, abs_tol = 1.0e-6, info = info)
    @test info.status == :continue
end

end
