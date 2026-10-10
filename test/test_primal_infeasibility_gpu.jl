using Test
using CUDA
using LinearAlgebra
using SparseArrays
using Logging

CUDA.functional() || error("A functional CUDA device is required")
include(joinpath(@__DIR__, "..", "src", "pdcs_gpu", "PDCS_GPU.jl"))
CUDA.allowscalar(false)

println("PRIMAL_REGRESSION_GPU=$(CUDA.name(CUDA.device()))")

function solve_primal_regression(x, h, c, rescaling_method; max_outer_iter = 0)
    return with_logger(NullLogger()) do
        PDCS_GPU.rpdhg_gpu_solve(
            n = 2, m = 3, nb = 2,
            G = sparse([1.0 0.0; 0.0 1.0; 1.0 1.0]),
            h = copy(h), c = copy(c), bl = zeros(2), bu = fill(Inf, 2),
            mGzero = 0, mGnonnegative = 3,
            socG = Int[], rsocG = Int[], expG = 0, dual_expG = 0,
            soc_x = Int[], rsoc_x = Int[],
            primal_sol = copy(x), dual_sol = zeros(3), warm_start = true,
            use_preconditioner = true, rescaling_method,
            use_restart = false, use_adaptive_restart = false,
            use_adaptive_step_size_weight = false, use_aggressive = false,
            use_reflection = false, use_resolving = false,
            use_duality_gap_restart = false, use_adaptive_step = false,
            max_outer_iter, max_inner_iter = 2000,
            check_terminate_freq = 10, abs_tol = 1.0e-6, rel_tol = 1.0e-6,
            verbose = 0, time_limit = 60.0,
        )
    end
end

@testset "GPU production primal normalization" begin
    # No iterations: inspect the specified point through the real solver's
    # initialization, rescaling, cuSPARSE products, and CUDA projections.
    cases = (
        (name = "projection dominates", x = [5.0, 0.0], h = [-5.0, -5.0, 6.0],
         abs_res = 1.0, scale_inf = 11.0, scale_l1 = 17.0),
        (name = "Gx dominates", x = [10.0, 1.0], h = [5.0, 2.0, 8.0],
         abs_res = 1.0, scale_inf = 12.0, scale_l1 = 23.0),
        (name = "h dominates", x = [1.0, 2.0], h = [10.0, -1.0, 1.0],
         abs_res = 9.0, scale_inf = 11.0, scale_l1 = 13.0),
        (name = "feasible", x = [1.0, 2.0], h = zeros(3),
         abs_res = 0.0, scale_inf = 4.0, scale_l1 = 7.0),
    )
    for scaling in (:none, :ruiz_pock_chambolle), case in cases,
        c in ([0.0, 10.0], [0.0, 5.0e8])
        @testset "$scaling $(case.name) cost=$(c[2])" begin
            sol = solve_primal_regression(case.x, case.h, c, scaling)
            info = sol.info.convergeInfo[1]
            @test sol.info.iter == 0
            @test Array(sol.x.recovered_primal.primal_sol.x) ≈ case.x
            @test info.l_inf_abs_primal_res ≈ case.abs_res atol = 1.0e-12
            @test info.l_inf_rel_primal_res ≈ case.abs_res / case.scale_inf atol = 1.0e-12
            @test info.l_2_rel_primal_res ≈ case.abs_res / case.scale_l1 atol = 1.0e-12
            if case.name == "projection dominates"
                @test info.l_inf_rel_dual_res == 0.0
                @test info.rel_gap == 0.0
                PDCS_GPU.optimality_criteria_met(
                    rel_tol = 1.0e-6, abs_tol = 1.0e-6, info,
                )
                @test info.status == :continue
            end
        end
    end
end

@testset "GPU solve and independently recomputed primal residual" begin
    G = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    h = [1.0, 2.0, 0.0]
    for scaling in (:none, :ruiz_pock_chambolle)
        sol = solve_primal_regression(
            zeros(2), h, ones(2), scaling; max_outer_iter = 5,
        )
        x = Array(sol.x.recovered_primal.primal_sol.x)
        gx = G * x
        projected = max.(gx - h, 0.0)
        expected = norm(gx - h - projected, Inf) /
                   (1 + max(norm(h, Inf), norm(gx, Inf), norm(projected, Inf)))
        info = sol.info.convergeInfo[1]
        @test sol.info.exit_status == :optimal
        @test x ≈ [1.0, 2.0] atol = 1.0e-4
        @test info.primal_objective ≈ 3.0 atol = 1.0e-4
        @test info.l_inf_rel_primal_res ≈ expected atol = 1.0e-12
        @test expected < 1.0e-6
    end
end
