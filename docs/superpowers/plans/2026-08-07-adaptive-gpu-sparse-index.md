# Adaptive GPU Sparse Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PDCS_GPU use Int32 sparse indices for ordinary matrices, automatically promote to Int64 at the Int32 boundary, and allow users to force either type.

**Architecture:** A CUDA-independent policy module normalizes and resolves `sparse_index_type` from matrix metadata. The CPU solve path passes that option to a two-path GPU uploader: CUDA.jl's existing Int32 convenience upload for representable matrices and the explicit Int64 CSR upload for oversized matrices. Shared CUDA.jl preprocessing kernels specialize on the CSR array element type, so both storage and index arithmetic match the resolved choice.

**Tech Stack:** Julia 1.10, JuMP, MathOptInterface, CUDA.jl 5/6, CUSPARSE, Julia `Test`.

## Global Constraints

- `sparse_index_type` accepts exactly `:auto`, `:int32`, `:int64`, `"auto"`, `"int32"`, `"int64"`, `Int32`, and `Int64`.
- `:auto` is the default at both the direct solver and JuMP/MOI interfaces.
- Int32 is valid only when `m`, `n`, and the one-based terminal pointer `nnz + 1` fit in `Int32`.
- Forced Int32 must fail before GPU allocation when the matrix is not representable.
- Floating-point matrix values and solver vectors remain `Float64`.
- Existing GPU CSR inputs retain their existing index type without conversion.
- Runtime CUDA.jl kernels replace the historical PTX preprocessing entry points for the affected sparse operations.
- Int64 CUSPARSE execution requires CUDA 11 or newer.
- The local machine has no NVIDIA GPU; actual Int32 and Int64 GPU numerical runs must be completed on a CUDA machine.

---

## File Structure

- `src/pdcs_gpu/csc_to_csr.jl`: CUDA-independent option normalization, Int32 representability checks, type resolution, and typed CPU CSC-to-CSR component conversion.
- `src/pdcs_gpu/PDCS_GPU.jl`: load the CUDA-independent sparse index policy before GPU structures and algorithms.
- `src/pdcs_gpu/def_struct.jl`: Int32/Int64 GPU upload dispatch and propagation through `coeffUnion`.
- `src/pdcs_gpu/rpdhg_alg_gpu_gen.jl`: public `sparse_index_type` keyword on the CPU-input solve path.
- `src/pdcs_gpu/MOI_wrapper/MOI_wrapper.jl`: default and propagation of the JuMP raw optimizer attribute.
- `src/pdcs_gpu/gpu_kernel.jl`: shared type-specialized sparse preprocessing kernels and launch wrappers.
- `src/pdcs_gpu/preprocess.jl`: allocate row-index workspace with the CSR index type.
- `test/test_gpu_sparse_index_policy.jl`: executable host-side policy and CSR conversion tests.
- `test/test_gpu_int64_sparse_contract.jl`: source-level GPU upload, propagation, and kernel contract checks usable without an NVIDIA GPU.
- `test/test_gpu_sparse_index_runtime.jl`: CUDA-gated numerical coverage of both actual GPU CSR index widths.

---

### Task 1: CUDA-Independent Sparse Index Policy

**Files:**
- Create: `test/test_gpu_sparse_index_policy.jl`
- Modify: `src/pdcs_gpu/csc_to_csr.jl`
- Modify: `src/pdcs_gpu/PDCS_GPU.jl`

**Interfaces:**
- Consumes: matrix dimensions and stored-entry count as `Integer` values.
- Produces: `_normalize_sparse_index_type(option)::Union{Symbol,Type{Int32},Type{Int64}}`, `_fits_int32_sparse_indices(m, n, stored_entries)::Bool`, `_resolve_sparse_index_type(option, m, n, stored_entries)::Type{<:Integer}`, and `_csc_to_csr_components(G, ::Type{Ti})`.

- [ ] **Step 1: Write failing policy and boundary tests**

Create `test/test_gpu_sparse_index_policy.jl`:

```julia
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
```

- [ ] **Step 2: Run the policy test and verify RED**

Run:

```bash
julia --startup-file=no test/test_gpu_sparse_index_policy.jl
```

Expected: `UndefVarError` for `_normalize_sparse_index_type` because the policy API does not exist yet.

- [ ] **Step 3: Implement normalization, boundary validation, and typed components**

Replace the contents of `src/pdcs_gpu/csc_to_csr.jl` with:

```julia
const _SPARSE_INDEX_TYPE_ERROR =
    "sparse_index_type must be :auto, Int32, or Int64 " *
    "(aliases :int32, :int64, \"auto\", \"int32\", and \"int64\" are accepted)"

function _normalize_sparse_index_type(option)
    option === :auto && return :auto
    option === "auto" && return :auto
    option === Int32 && return Int32
    option === :int32 && return Int32
    option === "int32" && return Int32
    option === Int64 && return Int64
    option === :int64 && return Int64
    option === "int64" && return Int64
    throw(ArgumentError(_SPARSE_INDEX_TYPE_ERROR))
end

function _fits_int32_sparse_indices(m::Integer, n::Integer, stored_entries::Integer)
    m >= 0 || throw(ArgumentError("matrix row count must be nonnegative"))
    n >= 0 || throw(ArgumentError("matrix column count must be nonnegative"))
    stored_entries >= 0 || throw(ArgumentError("matrix nnz must be nonnegative"))
    limit = typemax(Int32)
    return m <= limit && n <= limit && stored_entries <= limit - 1
end

function _resolve_sparse_index_type(
    option,
    m::Integer,
    n::Integer,
    stored_entries::Integer,
)
    normalized = _normalize_sparse_index_type(option)
    fits_int32 = _fits_int32_sparse_indices(m, n, stored_entries)
    normalized === :auto && return fits_int32 ? Int32 : Int64
    if normalized === Int32 && !fits_int32
        throw(ArgumentError(
            "matrix with m=$m, n=$n, and nnz=$stored_entries does not fit " *
            "in Int32 sparse indices; use sparse_index_type=Int64 or :auto",
        ))
    end
    return normalized
end

function _csc_to_csr_components(
    G::SparseMatrixCSC{Tv,<:Integer},
    ::Type{Ti},
) where {Tv,Ti<:Union{Int32,Int64}}
    transposed = copy(transpose(G))
    return (;
        rowptr = Ti.(transposed.colptr),
        colval = Ti.(transposed.rowval),
        nzval = transposed.nzval,
        dims = size(G),
    )
end
```

Keep this include in `src/pdcs_gpu/PDCS_GPU.jl` immediately after the
`MODULE_DIR` definition so later included source files can use the policy:

```julia
include("./csc_to_csr.jl")
```

- [ ] **Step 4: Run the policy test and verify GREEN**

Run:

```bash
julia --startup-file=no test/test_gpu_sparse_index_policy.jl
```

Expected: all three testsets pass with zero failures.

- [ ] **Step 5: Commit the policy unit**

```bash
git add src/pdcs_gpu/PDCS_GPU.jl src/pdcs_gpu/csc_to_csr.jl test/test_gpu_sparse_index_policy.jl
git commit -m "feat: resolve GPU sparse index width"
```

---

### Task 2: Upload Dispatch and Public Option Propagation

**Files:**
- Modify: `src/pdcs_gpu/def_struct.jl`
- Modify: `src/pdcs_gpu/rpdhg_alg_gpu_gen.jl`
- Modify: `src/pdcs_gpu/MOI_wrapper/MOI_wrapper.jl`
- Modify: `test/test_gpu_int64_sparse_contract.jl`

**Interfaces:**
- Consumes: `_resolve_sparse_index_type(option, m, n, stored_entries)` and `_csc_to_csr_components(G, Int64)` from Task 1.
- Produces: `_csc_to_gpu_csr(G, sparse_index_type=:auto)`, `coeffUnion(...; sparse_index_type=:auto)`, `rpdhg_gpu_solve(...; sparse_index_type=:auto)`, and raw optimizer attribute `"sparse_index_type"`.

- [ ] **Step 1: Write failing source-contract tests for upload and propagation**

Extend `test/test_gpu_int64_sparse_contract.jl` with:

```julia
const GPU_SOLVE_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "rpdhg_alg_gpu_gen.jl"),
    String,
)
const GPU_MOI_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "MOI_wrapper", "MOI_wrapper.jl"),
    String,
)

@testset "GPU sparse index selection reaches upload" begin
    @test occursin("sparse_index_type = :auto", GPU_SOLVE_SOURCE)
    @test occursin("sparse_index_type = sparse_index_type", GPU_SOLVE_SOURCE)
    @test occursin("options[:sparse_index_type] = :auto", GPU_MOI_SOURCE)
    @test occursin("sparse_index_type = options[:sparse_index_type]", GPU_MOI_SOURCE)
    @test occursin("::Type{Int32}", GPU_STRUCT_SOURCE)
    @test occursin("::Type{Int64}", GPU_STRUCT_SOURCE)
    @test occursin("_resolve_sparse_index_type", GPU_STRUCT_SOURCE)
end
```

Update the existing component call from
`_csc_to_int64_csr_components(G)` to `_csc_to_csr_components(G, Int64)`.

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
julia --startup-file=no test/test_gpu_int64_sparse_contract.jl
```

Expected: the new propagation assertions fail because no public option is passed to the uploader.

- [ ] **Step 3: Add two-path upload dispatch**

In `src/pdcs_gpu/def_struct.jl`, replace the fixed Int64 uploader with:

```julia
function _csc_to_gpu_csr(
    G::SparseMatrixCSC{Float64,<:Integer},
    sparse_index_type = :auto,
)
    Ti = _resolve_sparse_index_type(
        sparse_index_type,
        size(G, 1),
        size(G, 2),
        nnz(G),
    )
    return _csc_to_gpu_csr(G, Ti)
end

function _csc_to_gpu_csr(
    G::SparseMatrixCSC{Float64,<:Integer},
    ::Type{Int32},
)
    return CUDA.CUSPARSE.CuSparseMatrixCSR(G)
end

function _csc_to_gpu_csr(
    G::SparseMatrixCSC{Float64,<:Integer},
    ::Type{Int64},
)
    components = _csc_to_csr_components(G, Int64)
    return CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int64}(
        CuArray(components.rowptr),
        CuArray(components.colval),
        CuArray(components.nzval),
        components.dims,
    )
end

function _csc_to_gpu_csr(G::AbstractMatrix{Float64}, sparse_index_type = :auto)
    return _csc_to_gpu_csr(sparse(G), sparse_index_type)
end
```

Add `sparse_index_type = :auto` to the `coeffUnion` inner constructor and use:

```julia
d_G = _csc_to_gpu_csr(G, sparse_index_type)
```

Calls that already pass `d_G` do not invoke the uploader and therefore retain the existing GPU CSR type.

- [ ] **Step 4: Propagate the direct solver keyword**

In the CPU-input `rpdhg_gpu_solve` signature in `src/pdcs_gpu/rpdhg_alg_gpu_gen.jl`, add:

```julia
sparse_index_type = :auto,
```

Pass it only to the CPU-to-GPU `coeffUnion` construction:

```julia
coeff = coeffUnion(
    G = G,
    h = h,
    m = m,
    n = n,
    d_G = nothing,
    d_h = nothing,
    sparse_index_type = sparse_index_type,
)
```

- [ ] **Step 5: Propagate the JuMP/MOI raw optimizer attribute**

In `src/pdcs_gpu/MOI_wrapper/MOI_wrapper.jl`, set the default before either solve branch:

```julia
if !haskey(options, :sparse_index_type)
    options[:sparse_index_type] = :auto
end
```

Pass this keyword in both calls to `PDCS_GPU.rpdhg_gpu_solve`:

```julia
sparse_index_type = options[:sparse_index_type],
```

Do not normalize at `MOI.set`; the shared resolver validates the option before the first GPU allocation and is also used by direct callers.

- [ ] **Step 6: Run policy and propagation tests and verify GREEN**

Run:

```bash
julia --startup-file=no test/test_gpu_sparse_index_policy.jl
julia --startup-file=no test/test_gpu_int64_sparse_contract.jl
```

Expected: both files finish with zero failures.

- [ ] **Step 7: Commit upload and propagation**

```bash
git add src/pdcs_gpu/def_struct.jl src/pdcs_gpu/rpdhg_alg_gpu_gen.jl src/pdcs_gpu/MOI_wrapper/MOI_wrapper.jl test/test_gpu_int64_sparse_contract.jl
git commit -m "feat: select GPU sparse index type"
```

---

### Task 3: Type-Specialized Sparse GPU Preprocessing

**Files:**
- Modify: `src/pdcs_gpu/gpu_kernel.jl`
- Modify: `src/pdcs_gpu/preprocess.jl`
- Modify: `test/test_gpu_int64_sparse_contract.jl`

**Interfaces:**
- Consumes: `CuSparseMatrixCSR{Float64,Int32}` or `CuSparseMatrixCSR{Float64,Int64}` selected in Task 2.
- Produces: shared `_sparse_*_kernel!` definitions specialized by `eltype(rowptr)` or `eltype(indices)`, plus CSR-typed row-index workspace.

- [ ] **Step 1: Write failing generic-kernel contract tests**

Replace the old Int64-kernel-name testset in `test/test_gpu_int64_sparse_contract.jl` with:

```julia
const GPU_PREPROCESS_SOURCE = read(
    joinpath(@__DIR__, "..", "src", "pdcs_gpu", "preprocess.jl"),
    String,
)

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
    @test !occursin("_index64_thread", GPU_KERNEL_SOURCE)
    @test !occursin("Int64(@inbounds rowptr", GPU_KERNEL_SOURCE)
    @test occursin("similar(data.coeff.d_G.colVal", GPU_PREPROCESS_SOURCE)
    @test !occursin(
        "CUDA.zeros(Int64, nnz(data.coeff.d_G))",
        GPU_PREPROCESS_SOURCE,
    )
end
```

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
julia --startup-file=no test/test_gpu_int64_sparse_contract.jl
```

Expected: failures for the generic kernel names and CSR-typed workspace because the current patch hard-codes Int64.

- [ ] **Step 3: Generalize thread and CSR row kernels**

In `src/pdcs_gpu/gpu_kernel.jl`, replace `_index64_thread` and all four
row-oriented Int64 kernel definitions with this complete block:

```julia
@inline function _sparse_thread_index(::Type{Ti}) where {Ti<:Integer}
    return (Ti(CUDA.blockIdx().x) - one(Ti)) * Ti(CUDA.blockDim().x) +
           Ti(CUDA.threadIdx().x)
end
```

Rename the row-oriented kernels and infer `Ti` from `rowptr`:

```julia
function _rescale_csr_sparse_kernel!(
    values,
    rowptr,
    colval,
    row_scaling,
    col_scaling,
    nrows,
)
    Ti = eltype(rowptr)
    row = _sparse_thread_index(Ti)
    if row <= nrows
        first_position = @inbounds rowptr[row]
        last_position = (@inbounds rowptr[row + one(Ti)]) - one(Ti)
        for position in first_position:last_position
            column = @inbounds colval[position]
            @inbounds values[position] /= row_scaling[row] * col_scaling[column]
        end
    end
    return
end

function _max_abs_row_sparse_kernel!(values, rowptr, result, nrows)
    Ti = eltype(rowptr)
    row = _sparse_thread_index(Ti)
    if row <= nrows
        first_position = @inbounds rowptr[row]
        last_position = (@inbounds rowptr[row + one(Ti)]) - one(Ti)
        maximum_value = 0.0
        for position in first_position:last_position
            maximum_value = max(maximum_value, abs(@inbounds values[position]))
        end
        @inbounds result[row] = maximum_value
    end
    return
end

function _alpha_norm_row_sparse_kernel!(values, rowptr, alpha, result, nrows)
    Ti = eltype(rowptr)
    row = _sparse_thread_index(Ti)
    if row <= nrows
        first_position = @inbounds rowptr[row]
        last_position = (@inbounds rowptr[row + one(Ti)]) - one(Ti)
        total = 0.0
        for position in first_position:last_position
            value = abs(@inbounds values[position])
            total += alpha == 1.0 ? value : value^alpha
        end
        @inbounds result[row] = total
    end
    return
end

function _fill_row_sparse_kernel!(rowptr, row_indices, nrows)
    Ti = eltype(rowptr)
    row = _sparse_thread_index(Ti)
    if row <= nrows
        first_position = @inbounds rowptr[row]
        last_position = (@inbounds rowptr[row + one(Ti)]) - one(Ti)
        for position in first_position:last_position
            @inbounds row_indices[position] = row
        end
    end
    return
end
```

- [ ] **Step 4: Generalize COO and atomic kernels**

Replace the three element-oriented Int64 kernel definitions with this complete
block:

```julia
function _rescale_coo_sparse_kernel!(
    values,
    row_indices,
    col_indices,
    row_scaling,
    col_scaling,
    num_entries,
)
    Ti = eltype(row_indices)
    position = _sparse_thread_index(Ti)
    if position <= num_entries
        row = @inbounds row_indices[position]
        column = @inbounds col_indices[position]
        @inbounds values[position] /= row_scaling[row] * col_scaling[column]
    end
    return
end

function _max_abs_indexed_sparse_kernel!(values, indices, result, num_entries)
    Ti = eltype(indices)
    position = _sparse_thread_index(Ti)
    if position <= num_entries
        output_index = @inbounds indices[position]
        value = abs(@inbounds values[position])
        CUDA.@atomic result[output_index] = max(result[output_index], value)
    end
    return
end

function _alpha_norm_indexed_sparse_kernel!(
    values,
    indices,
    alpha,
    result,
    num_entries,
)
    Ti = eltype(indices)
    position = _sparse_thread_index(Ti)
    if position <= num_entries
        output_index = @inbounds indices[position]
        value = abs(@inbounds values[position])
        contribution = alpha == 1.0 ? value : value^alpha
        CUDA.@atomic result[output_index] += contribution
    end
    return
end
```

- [ ] **Step 5: Launch every sparse kernel with a typed bound**

Replace the active bodies of the affected wrapper functions in
`src/pdcs_gpu/gpu_kernel.jl` with these complete definitions:

```julia
function rescale_csr(
    d_G::CUDA.CUSPARSE.CuSparseMatrixCSR,
    row_scaling::CuArray,
    col_scaling::CuArray,
    m::Int64,
    n::Int64,
)
    m == 0 && return
    typed_m = eltype(d_G.rowPtr)(m)
    nblocks = cld(m, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _rescale_csr_sparse_kernel!(
            d_G.nzVal,
            d_G.rowPtr,
            d_G.colVal,
            row_scaling,
            col_scaling,
            typed_m,
        )
    end
    return
end

function max_abs_row(d_G, result)
    result .= 0.0
    nrows = size(d_G, 1)
    nrows == 0 && return
    typed_nrows = eltype(d_G.rowPtr)(nrows)
    nblocks = cld(nrows, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _max_abs_row_sparse_kernel!(
            d_G.nzVal,
            d_G.rowPtr,
            result,
            typed_nrows,
        )
    end
    return
end

max_abs_col(d_G, result) = max_abs_col_elementwise(d_G, result)

function alpha_norm_row(d_G, alpha, result)
    result .= 0.0
    nrows = size(d_G, 1)
    nrows == 0 && return
    typed_nrows = eltype(d_G.rowPtr)(nrows)
    nblocks = cld(nrows, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _alpha_norm_row_sparse_kernel!(
            d_G.nzVal,
            d_G.rowPtr,
            Float64(alpha),
            result,
            typed_nrows,
        )
    end
    return
end

alpha_norm_col(d_G, alpha, result) =
    alpha_norm_col_elementwise(d_G, alpha, result)

function get_row_index(d_G, row_idx)
    nrows = size(d_G, 1)
    nrows == 0 && return
    typed_nrows = eltype(d_G.rowPtr)(nrows)
    nblocks = cld(nrows, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _fill_row_sparse_kernel!(
            d_G.rowPtr,
            row_idx,
            typed_nrows,
        )
    end
    return
end

function rescale_coo(
    d_G::CUDA.CUSPARSE.CuSparseMatrixCSR,
    row_scaling::CuArray,
    col_scaling::CuArray,
    m::Int64,
    n::Int64,
    row_idx::CuArray,
)
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(row_idx)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _rescale_coo_sparse_kernel!(
            d_G.nzVal,
            row_idx,
            d_G.colVal,
            row_scaling,
            col_scaling,
            typed_entries,
        )
    end
    return
end

function max_abs_row_elementwise(d_G, row_idx, result)
    result .= 0.0
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(row_idx)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _max_abs_indexed_sparse_kernel!(
            d_G.nzVal,
            row_idx,
            result,
            typed_entries,
        )
    end
    return
end

function max_abs_col_elementwise(d_G, result)
    result .= 0.0
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(d_G.colVal)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _max_abs_indexed_sparse_kernel!(
            d_G.nzVal,
            d_G.colVal,
            result,
            typed_entries,
        )
    end
    return
end

function alpha_norm_col_elementwise(d_G, alpha, result)
    result .= 0.0
    num_entries = length(d_G.nzVal)
    num_entries == 0 && return
    typed_entries = eltype(d_G.colVal)(num_entries)
    nblocks = cld(num_entries, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.@cuda threads=ThreadPerBlock blocks=nblocks _alpha_norm_indexed_sparse_kernel!(
            d_G.nzVal,
            d_G.colVal,
            Float64(alpha),
            result,
            typed_entries,
        )
    end
    return
end
```

- [ ] **Step 6: Match the row-index workspace to CSR storage**

In `src/pdcs_gpu/preprocess.jl`, replace:

```julia
row_idx = CUDA.zeros(Int64, nnz(data.coeff.d_G))
```

with:

```julia
row_idx = similar(data.coeff.d_G.colVal, nnz(data.coeff.d_G))
```

`get_row_index` fills every entry before the workspace is read, so zero-initialization is unnecessary.

- [ ] **Step 7: Run the contract tests and parse the changed sources**

Run:

```bash
julia --startup-file=no test/test_gpu_sparse_index_policy.jl
julia --startup-file=no test/test_gpu_int64_sparse_contract.jl
julia --startup-file=no -e 'for file in ARGS; Meta.parseall(read(file, String)); end; println("GPU_SOURCES_PARSE")' src/pdcs_gpu/csc_to_csr.jl src/pdcs_gpu/def_struct.jl src/pdcs_gpu/gpu_kernel.jl src/pdcs_gpu/preprocess.jl src/pdcs_gpu/rpdhg_alg_gpu_gen.jl src/pdcs_gpu/MOI_wrapper/MOI_wrapper.jl
```

Expected: both test files pass and the parser prints `GPU_SOURCES_PARSE` with exit code 0.

- [ ] **Step 8: Expand the CUDA.jl kernels without an NVIDIA GPU**

Use a temporary environment so the repository remains clean:

```bash
julia --startup-file=no -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add(PackageSpec(name="CUDA", version="6.2.1")); using CUDA; source = read("src/pdcs_gpu/gpu_kernel.jl", String); first_marker = findfirst("# CUDA.jl\u0027s CPU CSC convenience upload", source); last_marker = findnext("# ----------------------------------------------------------------------------\n# Reflection Update Kernel", source, last(first_marker) + 1); snippet = source[first(first_marker):first(last_marker)-1]; Base.include_string(Main, snippet, "gpu_sparse_kernels.jl"); println("CUDA_SPARSE_KERNEL_DEFINITIONS_LOADED")'
```

Expected: exit code 0 and final line `CUDA_SPARSE_KERNEL_DEFINITIONS_LOADED`.

- [ ] **Step 9: Commit generic GPU preprocessing**

```bash
git add src/pdcs_gpu/gpu_kernel.jl src/pdcs_gpu/preprocess.jl test/test_gpu_int64_sparse_contract.jl
git commit -m "perf: specialize GPU sparse index kernels"
```

---

### Task 4: Regression Verification and CUDA Handoff

**Files:**
- Create: `test/test_gpu_sparse_index_runtime.jl`
- Verify: all files changed in Tasks 1-3.

**Interfaces:**
- Consumes: completed adaptive sparse index implementation from Tasks 1-3.
- Produces: fresh local regression evidence and exact CUDA-machine verification commands.

- [ ] **Step 1: Resolve project dependencies only if the repository has no manifest**

Run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
```

Record whether `Manifest.toml` was untracked before this command. If this command creates it solely for verification, remove only that newly generated file after all tests.

- [ ] **Step 2: Run PDCS bulk-cache regression**

Run:

```bash
julia --startup-file=no --compiled-modules=no --project=. test/test_bulk_cache.jl
```

Expected: all four testsets pass, including optimization without a second generic copy.

- [ ] **Step 3: Run realistic Lasso handoff regression**

Run:

```bash
julia --startup-file=no --compiled-modules=no --project=. /Users/zhenweilin/fork/Jump_RW/external/lasso_realistic_matrix/test_pdcs_bulk_lasso.jl
```

Expected: `PDCS bulk Lasso handoff` passes with zero failures.

- [ ] **Step 4: Run CPU-only import regression**

Run:

```bash
julia --startup-file=no test/test_cpu_only_import.jl
```

Expected: `PDCS CPU import does not load CUDA` passes.

- [ ] **Step 5: Check the final diff and repository cleanliness**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; only intentional adaptive-index changes are present, with no generated `Manifest.toml` or `.CondaPkg` artifact.

- [ ] **Step 6: Add and run a CUDA-gated two-width runtime test**

Create `test/test_gpu_sparse_index_runtime.jl`:

```julia
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
```

Run on both the local machine and a machine with CUDA 11 or newer:

```bash
julia --startup-file=no --project=. test/test_gpu_sparse_index_runtime.jl
```

Expected locally: one skipped assertion because `CUDA.functional()` is false.
Expected on the CUDA machine: all Int32 and Int64 assertions pass with no skips.

- [ ] **Step 7: Re-run the oversized Criteo Lasso case in auto mode**

From `/Users/zhenweilin/fork/Jump_RW/external/lasso_realistic_matrix`, run:

```bash
bash scripts/run_on_new_machine.sh \
  --mode pdcs-gpu \
  --dataset criteo \
  --julia-project /Users/zhenweilin/fork/Jump_RW/external/PDCS_fork \
  --pdcs-root /Users/zhenweilin/fork/Jump_RW/external/PDCS_fork \
  --index-type int64 \
  --skip-download
```

Leave `sparse_index_type` unset so PDCS_GPU uses `:auto`.

Expected: the log passes the former `CuSparseMatrixCSR(G)` conversion point
without `trunc(Int32, 2147483648)` and begins GPU preprocessing/iteration.
Capture `nvidia-smi --query-compute-apps=used_memory --format=csv` while the
model is resident; Int64 CSR requires four additional bytes per row pointer and
column index compared with Int32.

- [ ] **Step 8: Commit the CUDA runtime regression harness**

```bash
git add test/test_gpu_sparse_index_runtime.jl
git commit -m "test: cover GPU sparse index widths"
```
