using Test
using SparseArrays
using CUDA

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
        for Ti in (Int32, Int64)
            d_G = PDCS_GPU._csc_to_gpu_csr(G, Ti)
            @test eltype(d_G.rowPtr) === Ti
            @test eltype(d_G.colVal) === Ti
            @test Array(d_G * CuArray(x)) ≈ G * x

            row_max = CUDA.zeros(Float64, size(G, 1))
            PDCS_GPU.max_abs_row(d_G, row_max)
            @test Array(row_max) == [2.0, 3.0, 5.0]

            col_max = CUDA.zeros(Float64, size(G, 2))
            PDCS_GPU.max_abs_col(d_G, col_max)
            @test Array(col_max) == [5.0, 3.0, 4.0]

            row_idx = similar(d_G.colVal, nnz(d_G))
            PDCS_GPU.get_row_index(d_G, row_idx)
            @test Array(row_idx) == Ti[1, 1, 2, 3, 3]
        end
    end
end
