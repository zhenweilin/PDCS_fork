using Test
using SparseArrays
using TOML

include(joinpath(@__DIR__, "..", "src", "pdcs_gpu", "csc_to_csr.jl"))

const GPU_STRUCT_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "def_struct.jl"),
    String,
)
const GPU_CSC_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "csc_to_csr.jl"),
    String,
)
const GPU_KERNEL_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "gpu_kernel.jl"),
    String,
)
const GPU_PREPROCESS_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "preprocess.jl"),
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
const README_SOURCE = read(
    joinpath(@__DIR__, "..", "README.md"),
    String,
)
const INSTALL_SOURCE = read(joinpath(@__DIR__, "install.jl"), String)
const PROJECT_CONFIG = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))

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
    @test length(findall(
        "function _csc_to_gpu_csr(",
        GPU_CSC_SOURCE * GPU_STRUCT_SOURCE,
    )) == 2
    @test length(findall("function _upload_csc_to_gpu_csr(", GPU_STRUCT_SOURCE)) == 2
    @test occursin(
        "return _upload_csc_to_gpu_csr(G, Ti)",
        GPU_CSC_SOURCE * GPU_STRUCT_SOURCE,
    )
    @test occursin("::Type{Int32}", GPU_STRUCT_SOURCE)
    @test occursin("::Type{Int64}", GPU_STRUCT_SOURCE)
    @test occursin(
        "_resolve_sparse_index_type",
        GPU_CSC_SOURCE * GPU_STRUCT_SOURCE,
    )
end

@testset "GPU preprocessing specializes on CSR index type" begin
    for kernel in (
        "_rescale_csr_sparse_kernel!",
        "_max_abs_row_sparse_kernel!",
        "_alpha_norm_row_sparse_kernel!",
        "_fill_row_sparse_kernel!",
        "_rescale_coo_sparse_kernel!",
        "_max_abs_indexed_sparse_kernel!",
        "_alpha_norm_indexed_sparse_kernel!",
    )
        @test occursin(kernel, GPU_KERNEL_SOURCE)
    end
    @test occursin("_sparse_thread_index", GPU_KERNEL_SOURCE)
    @test occursin("Tu = unsigned(Ti)", GPU_KERNEL_SOURCE)
    @test occursin("first_position = @inbounds rowptr[row_address]", GPU_KERNEL_SOURCE)
    @test occursin("rowptr[row_address + one(Tu)]", GPU_KERNEL_SOURCE)
    @test !occursin("_index64_thread", GPU_KERNEL_SOURCE)
    @test !occursin("Int64(@inbounds rowptr", GPU_KERNEL_SOURCE)
    @test occursin("similar(data.coeff.d_G.colVal", GPU_PREPROCESS_SOURCE)
    @test !occursin(
        "CUDA.zeros(Int64, nnz(data.coeff.d_G))",
        GPU_PREPROCESS_SOURCE,
    )
end


@testset "GPU sparse index public documentation" begin
    @test occursin("`sparse_index_type`", GPU_SOLVE_SOURCE)
    @test occursin(":auto", GPU_SOLVE_SOURCE)
    @test occursin(":int32", GPU_SOLVE_SOURCE)
    @test occursin(":int64", GPU_SOLVE_SOURCE)
    @test occursin("CUDA 11", GPU_SOLVE_SOURCE)
    @test occursin(
        "set_optimizer_attribute(model, \"sparse_index_type\", :auto)",
        README_SOURCE,
    )
end

@testset "GPU extension dependency and install entry points" begin
    cuda_uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
    @test !haskey(PROJECT_CONFIG["deps"], "CUDA")
    @test PROJECT_CONFIG["weakdeps"]["CUDA"] == cuda_uuid
    @test PROJECT_CONFIG["extensions"]["PDCSGPUExt"] == "CUDA"
    @test occursin("Pkg.add(\"CUDA\")", README_SOURCE)
    @test occursin("Pkg.add(\"CUDA\")", INSTALL_SOURCE)
    @test findfirst("using CUDA", INSTALL_SOURCE) <
          findfirst("using PDCS: PDCS_CPU, PDCS_GPU", INSTALL_SOURCE)
end

@testset "GPU max wrappers document their reset behavior" begin
    @test !occursin("Should be initialized to 1.0 before calling", GPU_KERNEL_SOURCE)
end

@testset "cuBLAS loading remains lazy without changing handle operations" begin
    @test !occursin("CUDA.CUBLAS.libcublas", GPU_KERNEL_SOURCE)
    @test occursin("CUDA.CUBLAS.cublasCreate()", GPU_KERNEL_SOURCE)
    @test occursin("CUDA.CUBLAS.cublasDestroy_v2(ch.handle)", GPU_KERNEL_SOURCE)
    @test occursin("get_gridWise_cublas_handle()", GPU_KERNEL_SOURCE)
end
