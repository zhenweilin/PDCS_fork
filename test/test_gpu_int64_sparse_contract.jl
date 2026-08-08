using Test
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "pdcs_gpu", "csc_to_csr.jl"))

const GPU_STRUCT_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "def_struct.jl"),
    String,
)
const GPU_KERNEL_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "gpu_kernel.jl"),
    String,
)
const GPU_SOLVE_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "rpdhg_alg_gpu_gen.jl"),
    String,
)
const GPU_MOI_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "MOI_wrapper", "MOI_wrapper.jl"),
    String,
)

@testset "GPU CSC upload preserves Int64 sparse indices" begin
    @test !occursin(
        "d_G = CUDA.CUSPARSE.CuSparseMatrixCSR(G)",
        GPU_STRUCT_SOURCE,
    )
    @test occursin("_csc_to_gpu_csr(G, sparse_index_type)", GPU_STRUCT_SOURCE)

    G = SparseMatrixCSC{Float64,Int32}(sparse(
        [1, 3, 2, 1, 3],
        [1, 1, 2, 3, 3],
        [2.0, 5.0, 3.0, 1.0, 4.0],
        3,
        3,
    ))
    csr = _csc_to_csr_components(G, Int64)
    @test csr.rowptr == Int64[1, 3, 4, 6]
    @test csr.colval == Int64[1, 3, 2, 1, 3]
    @test csr.nzval == [2.0, 1.0, 3.0, 5.0, 4.0]
    @test csr.dims == size(G)
    @test eltype(csr.rowptr) == Int64
    @test eltype(csr.colval) == Int64
end

@testset "GPU sparse index selection reaches upload" begin
    @test occursin("sparse_index_type = :auto", GPU_SOLVE_SOURCE)
    @test occursin("sparse_index_type = sparse_index_type", GPU_SOLVE_SOURCE)
    @test occursin("options[:sparse_index_type] = :auto", GPU_MOI_SOURCE)
    @test occursin("sparse_index_type = options[:sparse_index_type]", GPU_MOI_SOURCE)
    @test occursin("::Type{Int32}", GPU_STRUCT_SOURCE)
    @test occursin("::Type{Int64}", GPU_STRUCT_SOURCE)
    @test occursin("_resolve_sparse_index_type", GPU_STRUCT_SOURCE)
end

@testset "GPU preprocessing kernels accept Int64 CSR indices" begin
    for kernel in (
        "_rescale_csr_index64_kernel!",
        "_max_abs_row_index64_kernel!",
        "_alpha_norm_row_index64_kernel!",
        "_fill_row_index64_kernel!",
        "_rescale_coo_index64_kernel!",
        "_max_abs_col_index64_kernel!",
        "_alpha_norm_col_index64_kernel!",
    )
        @test occursin(kernel, GPU_KERNEL_SOURCE)
    end
end
