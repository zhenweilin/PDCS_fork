using Test
using SparseArrays
using LinearAlgebra
using CUDA
using JuMP
import PDCS
import MathOptInterface as MOI

const CHECKOUT_ROOT = realpath(joinpath(@__DIR__, ".."))

@testset "PDCS checkout provenance" begin
    @test realpath(pkgdir(PDCS)) == CHECKOUT_ROOT
end

@testset "PDCS GPU sparse index runtime" begin
    if !CUDA.functional()
        @test_skip CUDA.functional()
    else
        @eval using PDCS: PDCS_GPU

        G = SparseMatrixCSC{Float64,Int64}(sparse(
            [1, 3, 2, 1, 3],
            [1, 1, 2, 3, 3],
            [2.0, 5.0, 3.0, 1.0, 4.0],
            3,
            3,
        ))
        x = [1.0, 2.0, 3.0]
        expected_row_max = [2.0, 3.0, 5.0]
        expected_col_max = [5.0, 3.0, 4.0]
        expected_row_norm = [sqrt(5.0), 3.0, sqrt(41.0)]
        expected_col_norm = [sqrt(29.0), 3.0, sqrt(17.0)]
        row_scaling = [2.0, 3.0, 4.0]
        col_scaling = [5.0, 6.0, 7.0]
        expected_rescaled = Diagonal(row_scaling) * G * Diagonal(col_scaling)

        for Ti in (Int32, Int64)
            coeff = PDCS_GPU.coeffUnion(
                G=G,
                h=zeros(size(G, 1)),
                m=size(G, 1),
                n=size(G, 2),
                sparse_index_type=Ti,
            )
            d_G = coeff.d_G
            @test eltype(d_G.rowPtr) === Ti
            @test eltype(d_G.colVal) === Ti
            @test Array(d_G * CuArray(x)) ≈ G * x

            row_max = CUDA.zeros(Float64, size(G, 1))
            PDCS_GPU.max_abs_row(d_G, row_max)
            @test Array(row_max) == expected_row_max

            col_max = CUDA.zeros(Float64, size(G, 2))
            PDCS_GPU.max_abs_col(d_G, col_max)
            @test Array(col_max) == expected_col_max

            row_idx = similar(d_G.colVal, nnz(d_G))
            @test eltype(row_idx) === Ti
            PDCS_GPU.get_row_index(d_G, row_idx)
            @test Array(row_idx) == Ti[1, 1, 2, 3, 3]

            row_norm = CUDA.zeros(Float64, size(G, 1))
            PDCS_GPU.alpha_norm_row(d_G, 2.0, row_norm)
            @test Array(row_norm) ≈ expected_row_norm

            col_norm = CUDA.zeros(Float64, size(G, 2))
            PDCS_GPU.alpha_norm_col(d_G, 2.0, col_norm)
            @test Array(col_norm) ≈ expected_col_norm

            csr_scaled = PDCS_GPU.coeffUnion(
                G=G,
                h=zeros(size(G, 1)),
                m=size(G, 1),
                n=size(G, 2),
                sparse_index_type=Ti,
            ).d_G
            PDCS_GPU.rescale_csr(
                csr_scaled,
                CuArray(row_scaling),
                CuArray(col_scaling),
                Int64(size(G, 1)),
                Int64(size(G, 2)),
            )
            @test Array(csr_scaled * CuArray(x)) ≈ expected_rescaled * x

            coo_scaled = PDCS_GPU.coeffUnion(
                G=G,
                h=zeros(size(G, 1)),
                m=size(G, 1),
                n=size(G, 2),
                sparse_index_type=Ti,
            ).d_G
            coo_row_idx = similar(coo_scaled.colVal, nnz(coo_scaled))
            PDCS_GPU.get_row_index(coo_scaled, coo_row_idx)
            PDCS_GPU.rescale_coo(
                coo_scaled,
                CuArray(row_scaling),
                CuArray(col_scaling),
                Int64(size(G, 1)),
                Int64(size(G, 2)),
                coo_row_idx,
            )
            @test Array(coo_scaled * CuArray(x)) ≈ expected_rescaled * x

            direct = PDCS_GPU.rpdhg_gpu_solve(
                n=1,
                m=1,
                nb=1,
                c=[1.0],
                G=sparse([1], [1], [1.0], 1, 1),
                h=[1.0],
                mGzero=0,
                mGnonnegative=1,
                socG=Integer[],
                rsocG=Integer[],
                expG=0,
                dual_expG=0,
                bl=[0.0],
                bu=[Inf],
                soc_x=Integer[],
                rsoc_x=Integer[],
                use_preconditioner=false,
                use_adaptive_restart=false,
                use_restart=false,
                use_aggressive=false,
                use_reflection=false,
                use_duality_gap_restart=false,
                max_outer_iter=2,
                max_inner_iter=50,
                check_terminate_freq=1,
                print_freq=100,
                time_limit=10.0,
                verbose=0,
                sparse_index_type=Ti,
            )
            @test direct.info.exit_status in (:optimal, :max_iter, :time_limit)
            @test isfinite(direct.info.pObj)
            @test all(isfinite, Array(direct.x.recovered_primal.primal_sol.x))

            for use_scaling in (true, false)
                model = JuMP.Model(PDCS_GPU.Optimizer)
                JuMP.set_optimizer_attribute(model, "sparse_index_type", Ti)
                JuMP.set_optimizer_attribute(model, "use_scaling", use_scaling)
                JuMP.set_optimizer_attribute(model, "max_outer_iter", 2)
                JuMP.set_optimizer_attribute(model, "max_inner_iter", 50)
                JuMP.set_optimizer_attribute(model, "check_terminate_freq", 1)
                JuMP.set_optimizer_attribute(model, "print_freq", 100)
                JuMP.set_optimizer_attribute(model, "time_limit_secs", 10.0)
                JuMP.set_optimizer_attribute(model, "verbose", 0)
                JuMP.@variable(model, variable >= 0.0)
                JuMP.@objective(model, Min, variable)
                JuMP.@constraint(model, variable >= 1.0)
                JuMP.optimize!(model)

                @test JuMP.termination_status(model) in (
                    MOI.OPTIMAL,
                    MOI.ITERATION_LIMIT,
                    MOI.TIME_LIMIT,
                )
                @test JuMP.result_count(model) == 1
                @test isfinite(JuMP.objective_value(model))
                @test isfinite(JuMP.value(variable))
            end
        end
    end
end
