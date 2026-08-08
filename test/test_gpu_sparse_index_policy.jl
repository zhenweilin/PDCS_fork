using Test
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "pdcs_gpu", "csc_to_csr.jl"))

@testset "GPU sparse index option normalization" begin
    for option in (:auto, "auto")
        @test _normalize_sparse_index_type(option) === :auto
    end
    for option in (Int32, :int32, "int32")
        @test _normalize_sparse_index_type(option) === Int32
    end
    for option in (Int64, :int64, "int64")
        @test _normalize_sparse_index_type(option) === Int64
    end
    @test_throws ArgumentError _normalize_sparse_index_type(:int16)
    @test_throws ArgumentError _normalize_sparse_index_type("Int32")
end

@testset "GPU sparse index auto resolution" begin
    int32_max = typemax(Int32)
    @test _fits_int32_sparse_indices(10, 20, 30)
    @test _fits_int32_sparse_indices(1, 1, int32_max - 1)
    @test !_fits_int32_sparse_indices(1, 1, int32_max)
    @test !_fits_int32_sparse_indices(int32_max + 1, 1, 0)
    @test !_fits_int32_sparse_indices(1, int32_max + 1, 0)
    @test _resolve_sparse_index_type(:auto, 10, 20, 30) === Int32
    @test _resolve_sparse_index_type(:auto, 1, 1, int32_max) === Int64
    @test _resolve_sparse_index_type(Int64, 10, 20, 30) === Int64
    @test_throws ArgumentError _resolve_sparse_index_type(
        Int32,
        1,
        1,
        int32_max,
    )
    @test_throws ArgumentError _resolve_sparse_index_type(:auto, -1, 1, 0)
end

@testset "GPU sparse launch address arithmetic" begin
    padded_position = _sparse_unsigned_thread_index(Int32, 8_388_608, 256, 256)
    @test padded_position == UInt32(2_147_483_648)
    @test typeof(padded_position) === UInt32
    @test padded_position > UInt32(typemax(Int32))
end

@testset "Typed CSC to CSR components" begin
    G = SparseMatrixCSC{Float64,Int64}(sparse(
        [1, 3, 2, 1, 3],
        [1, 1, 2, 3, 3],
        [2.0, 5.0, 3.0, 1.0, 4.0],
        3,
        3,
    ))
    for Ti in (Int32, Int64)
        csr = _csc_to_csr_components(G, Ti)
        @test csr.rowptr == Ti[1, 3, 4, 6]
        @test csr.colval == Ti[1, 3, 2, 1, 3]
        @test csr.nzval == [2.0, 1.0, 3.0, 5.0, 4.0]
        @test csr.dims == (3, 3)
        @test eltype(csr.rowptr) === Ti
        @test eltype(csr.colval) === Ti
    end
end

@testset "Sparse index vector conversion avoids exact-type copies" begin
    @test isdefined(@__MODULE__, :_as_sparse_index_vector)
    if isdefined(@__MODULE__, :_as_sparse_index_vector)
        exact = Int64[1, 3, 4, 6]
        @test _as_sparse_index_vector(exact, Int64) === exact

        narrower = Int32[1, 3, 4, 6]
        converted = _as_sparse_index_vector(narrower, Int64)
        @test converted == exact
        @test eltype(converted) === Int64
        @test converted !== narrower
    end
end

@testset "Forced Int32 rejects oversized sparse metadata" begin
    oversized_rows = Int(typemax(Int32)) + 1
    G = SparseMatrixCSC{Float64,Int64}(
        oversized_rows,
        1,
        Int64[1, 1],
        Int64[],
        Float64[],
    )
    @test size(G) == (oversized_rows, 1)
    @test nnz(G) == 0
    @test isdefined(@__MODULE__, :_csc_to_gpu_csr)
    @test_throws ArgumentError _resolve_sparse_index_type(
        Int32,
        size(G, 1),
        size(G, 2),
        nnz(G),
    )
    if isdefined(@__MODULE__, :_csc_to_gpu_csr)
        @test_throws ArgumentError _csc_to_gpu_csr(G, Int32)
    end
end
