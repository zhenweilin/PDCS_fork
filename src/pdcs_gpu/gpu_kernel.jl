"""
GPU Kernel Wrapper Functions for PDCS_GPU

This file provides Julia wrappers for CUDA kernels used in the PDCS GPU solver.
It implements lazy loading of PTX (CUDA kernel) files with thread-safe initialization.

Main components:
1. Block projection kernels (massive, moderate, sufficient) - for projecting onto constraint sets
2. cuBLAS handle management - for using cuBLAS library functions
3. Few block projection - uses a shared library (.so) for complex projections
4. Utility kernels - various helper functions for the RPDHG algorithm
5. Matrix scaling and norm computation kernels - for preconditioning and scaling
6. Elementwise operations - for sparse matrix operations

All kernels use a lazy loading pattern with thread-safe initialization to ensure
kernels are loaded only once, even in multi-threaded environments.
"""

# ============================================================================
# Section 1: Block Projection Kernels
# ============================================================================
# These kernels perform projections onto constraint sets for different block sizes.
# The kernels are loaded from PTX files and use lazy initialization.

# ----------------------------------------------------------------------------
# Massive Block Projection Kernel
# ----------------------------------------------------------------------------
# Used for large blocks. Loads the PTX file and provides thread-safe access.
# The kernel projects vectors onto constraint sets for blocks with many elements.

# Storage for the loaded CUDA module (PTX file)
const _massive_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
# Storage for the CUDA function (kernel entry point)
const _massive_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
# Spin lock for thread-safe lazy initialization
const _massive_block_proj_lock   = SpinLock()

# Path to the PTX file containing the compiled CUDA kernel
const _massive_block_proj_path = joinpath(MODULE_DIR, "cuda/massive_block_proj.ptx")
# Name of the kernel function within the PTX file
const _massive_block_proj_name = "massive_block_proj"

"""
    get_massive_block_proj_kernel() -> CuFunction

Lazily loads and returns the massive block projection CUDA kernel.

Uses double-checked locking pattern for thread-safe initialization:
1. First check without lock (fast path)
2. If not loaded, acquire lock
3. Double-check after acquiring lock (another thread might have loaded it)
4. If still not loaded, load the PTX file and cache the kernel

Returns the CuFunction that can be used with CUDA.cudacall().
"""
function get_massive_block_proj_kernel()::CuFunction
    # Fast path: check if already loaded (no lock needed for read)
    k = _massive_block_proj_kernel[]
    k !== nothing && return k

    # Slow path: acquire lock and load kernel
    lock(_massive_block_proj_lock)
    try
        # Double-check after locking (another thread might have loaded it while we waited)
        k = _massive_block_proj_kernel[]
        k !== nothing && return k

        # Verify CUDA is available
        CUDA.functional() || error("CUDA is not functional")
        # Ensure CUDA context exists (creates context if needed)
        CUDA.zeros(Float32, 1)

        # Load the PTX file from disk
        bytes = read(_massive_block_proj_path)     # Read as Vector{UInt8}
        # Create CUDA module from PTX bytes
        mod   = CuModule(bytes)
        # Get the kernel function from the module
        fun   = CuFunction(mod, _massive_block_proj_name)

        # Cache the module and function for future use
        _massive_block_proj_mod[]    = mod
        _massive_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_massive_block_proj_lock)
    end
end


"""
    massive_block_proj(vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol)

Projects vectors onto constraint sets for large blocks (massive blocks).

Arguments:
- vec: Vector to project (modified in-place)
- bl: Lower bounds for the projection
- bu: Upper bounds for the projection
- D_scaled: Scaled diagonal matrix D
- D_scaled_squared: D scaled and squared
- D_scaled_mul_x: D scaled multiplied by x
- temp: Temporary storage array
- t_warm_start: Warm start values for t
- gpu_head_start: GPU array of block head start indices
- gpu_ns: GPU array of block sizes
- blkNum: Number of blocks
- proj_type: Type of projection for each block
- abs_tol: Absolute tolerance for projection
- rel_tol: Relative tolerance for projection

The kernel computes the projection onto constraint sets defined by the bounds
and scaling matrices, using the specified projection types for each block.
"""
function massive_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    # Calculate number of thread blocks needed
    # cld(x, y) = ceil(x/y) = smallest integer >= x/y
    nBlock = cld(blkNum + ThreadPerBlock, ThreadPerBlock)
    
    # Launch kernel and wait for completion
    CUDA.@sync begin
        CUDA.cudacall(
        get_massive_block_proj_kernel(),
        # Kernel function signature: all pointers to device memory
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        # Kernel arguments
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        # Launch configuration: nBlock blocks, ThreadPerBlock threads per block
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

"""Thread-wise projection: one GPU thread processes one cone block."""
threadWise_block_proj(args...) = massive_block_proj(args...)

# ----------------------------------------------------------------------------
# Moderate Block Projection Kernel
# ----------------------------------------------------------------------------
# Used for medium-sized blocks. Similar structure to massive_block_proj but
# optimized for different block size ranges.

const _moderate_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _moderate_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _moderate_block_proj_lock   = SpinLock()

const _moderate_block_proj_path = joinpath(MODULE_DIR, "cuda/moderate_block_proj.ptx")
const _moderate_block_proj_name = "moderate_block_proj"

"""
    get_moderate_block_proj_kernel() -> CuFunction

Lazily loads and returns the moderate block projection CUDA kernel.
Uses the same thread-safe lazy loading pattern as massive_block_proj.
"""
function get_moderate_block_proj_kernel()::CuFunction
    k = _moderate_block_proj_kernel[]
    k !== nothing && return k

    lock(_moderate_block_proj_lock)
    try
        k = _moderate_block_proj_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_moderate_block_proj_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _moderate_block_proj_name)

        _moderate_block_proj_mod[]    = mod
        _moderate_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_moderate_block_proj_lock)
    end
end

"""
    moderate_block_proj(...)

Projects vectors onto constraint sets for medium-sized blocks.
Similar to massive_block_proj but uses a different block configuration:
nBlock = blkNum + 1 (one block per constraint block plus one extra).
"""
function moderate_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    # For moderate blocks, use one block per constraint block plus one
    nBlock = blkNum + 1
    CUDA.@sync begin
        CUDA.cudacall(
        get_moderate_block_proj_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

"""Block-wise projection: one GPU thread block processes one cone block."""
blockWise_block_proj(args...) = moderate_block_proj(args...)


# ----------------------------------------------------------------------------
# Sufficient Block Projection Kernel
# ----------------------------------------------------------------------------
# Used for blocks with sufficient parallelism. Optimized for cases where
# there are enough elements to fully utilize GPU threads.

const _sufficient_block_proj_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _sufficient_block_proj_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _sufficient_block_proj_lock   = SpinLock()

const _sufficient_block_proj_path = joinpath(MODULE_DIR, "cuda/sufficient_block_proj.ptx")
const _sufficient_block_proj_name = "sufficient_block_proj"

"""
    get_sufficient_block_proj_kernel() -> CuFunction

Lazily loads and returns the sufficient block projection CUDA kernel.
Uses the same thread-safe lazy loading pattern.
"""
function get_sufficient_block_proj_kernel()::CuFunction
    k = _sufficient_block_proj_kernel[]
    k !== nothing && return k

    lock(_sufficient_block_proj_lock)
    try
        k = _sufficient_block_proj_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_sufficient_block_proj_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _sufficient_block_proj_name)

        _sufficient_block_proj_mod[]    = mod
        _sufficient_block_proj_kernel[] = fun
        return fun
    finally
        unlock(_sufficient_block_proj_lock)
    end
end

"""
    sufficient_block_proj(...)

Projects vectors onto constraint sets for blocks with sufficient parallelism.
Uses a different block configuration: nBlock = ceil((blkNum + 1) * 32 / ThreadPerBlock),
which provides more blocks for better GPU utilization.
"""
function sufficient_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, gpu_head_start::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, blkNum::Int64, proj_type::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    # Calculate blocks: (blkNum + 1) * 32 elements distributed across thread blocks
    nBlock = cld((blkNum + 1) * 32, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(
        get_sufficient_block_proj_kernel(),
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Int64}, CuPtr{Int64}, Int64, CuPtr{Int64}, Float64, Float64),
        vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, gpu_head_start, gpu_ns, blkNum, proj_type, abs_tol, rel_tol;
        blocks = nBlock, threads = ThreadPerBlock
        )
    end
end

"""Warp-wise projection: one GPU warp processes one cone block."""
warpWise_block_proj(args...) = sufficient_block_proj(args...)

# ============================================================================
# Section 2: cuBLAS Handle Management
# ============================================================================
# cuBLAS is NVIDIA's CUDA Basic Linear Algebra Subroutines library.
# We need to create and manage cuBLAS handles for performing linear algebra
# operations on the GPU (e.g., matrix-vector products, norms).

# cuBLAS type definitions
const cublasStatus_t = Cint                    # cuBLAS status code type
const CUBLAS_STATUS_SUCCESS = 0                # Success status code
const cublasHandle_t = Ptr{Cvoid}              # cuBLAS handle type (opaque pointer)

"""
    CUBLASHandle

Wrapper struct for cuBLAS handle.
A cuBLAS handle is required for all cuBLAS operations and manages the
library's internal state and resources.
"""
mutable struct CUBLASHandle
    handle::cublasHandle_t  # Opaque pointer to cuBLAS context
end

# The grid-wise kernel is the only projection strategy that needs a cuBLAS
# handle. Create it lazily so importing PDCS does not create a CUDA context,
# and keep ownership in this module rather than in individual solver calls.
const _gridWise_cublas_handle = Ref{Union{Nothing,CUBLASHandle}}(nothing)
const _gridWise_cublas_handle_lock = SpinLock()
const _gridWise_cublas_atexit_registered = Ref(false)

"""
    create_cublas_handle() -> CUBLASHandle

Creates a new cuBLAS handle for performing linear algebra operations on GPU.

The handle must be created before using any cuBLAS functions and should be
destroyed when no longer needed using destroy_cublas_handle().

Returns a CUBLASHandle wrapper containing the opaque cuBLAS handle pointer.
"""
function create_cublas_handle()
    # Ensure CUDA context exists (required before creating cuBLAS handle)
    CUDA.zeros(Float32, 1)

    # Create it through CUDA.jl, so it belongs to the current CUDA context
    # and uses exactly the same cuBLAS runtime as the CuArray arguments.
    h = CUDA.CUBLAS.cublasCreate()
    h != C_NULL || error("cublasCreate_v2 returned NULL handle")
    # Keep the handle on CUDA's legacy default stream. The shared grid-wise
    # library launches CUDA C++ kernels and Thrust operations on that stream.
    # Binding this handle to CUDA.jl's task stream would make the C++ kernels
    # and cuBLAS reductions race each other.
    return CUBLASHandle(h)
end

"""
    get_gridWise_cublas_handle() -> CUBLASHandle

Return the process-local cuBLAS handle used by `gridWise_block_proj`, creating
it only on first use. It is subsequently reused by every projection in every
iteration of the solve.
"""
function get_gridWise_cublas_handle()
    current = _gridWise_cublas_handle[]
    current !== nothing && current.handle != C_NULL && return current

    lock(_gridWise_cublas_handle_lock)
    try
        current = _gridWise_cublas_handle[]
        if current === nothing || current.handle == C_NULL
            current = create_cublas_handle()
            _gridWise_cublas_handle[] = current
        end
        return current
    finally
        unlock(_gridWise_cublas_handle_lock)
    end
end

"""
    destroy_cublas_handle(ch::CUBLASHandle)

Destroys a cuBLAS handle, freeing associated resources.

Should be called when the handle is no longer needed. Safe to call multiple
times (idempotent if handle is already C_NULL).
"""
function destroy_cublas_handle(ch::CUBLASHandle)
    # Early return if handle is already null
    ch.handle == C_NULL && return nothing

    # Destroy through CUDA.jl's cuBLAS wrapper, matching the Julia-side
    # creation above.
    CUDA.CUBLAS.cublasDestroy_v2(ch.handle)
    # Mark handle as destroyed
    ch.handle = C_NULL
    return nothing
end

"""
    release_gridWise_cublas_handle!()

Release the cached grid-wise cuBLAS handle. Normal solver calls should not use
this between projections: the handle is intentionally reused for the whole
optimization. It is available for explicit teardown after the final solve.
"""
function release_gridWise_cublas_handle!()
    lock(_gridWise_cublas_handle_lock)
    try
        current = _gridWise_cublas_handle[]
        current === nothing || destroy_cublas_handle(current)
        _gridWise_cublas_handle[] = nothing
    finally
        unlock(_gridWise_cublas_handle_lock)
    end
    return nothing
end

"""Register one Julia-side cleanup callback for the cached grid-wise handle."""
function register_gridWise_cublas_cleanup!()
    _gridWise_cublas_atexit_registered[] && return nothing
    _gridWise_cublas_atexit_registered[] = true
    atexit(release_gridWise_cublas_handle!)
    return nothing
end


# ============================================================================
# Section 3: Few Block Projection (Shared Library Function)
# ============================================================================
# This function calls a C function from the shared library libfew_block_proj.so.
# Unlike the PTX kernels, this uses a function pointer loaded from a .so file.
# The function performs projections for cases with few blocks.

"""
    few_block_proj(vec, bl, bu, D_scaled, D_scaled_squared, D_scaled_mul_x, temp, t_warm_start, cpu_head_start, gpu_ns, cpu_ns, blkNum, cpu_proj_type, abs_tol, rel_tol)

Projects vectors onto constraint sets for cases with few blocks.

This function calls a C function from the shared library libfew_block_proj.so.
The function pointer must be initialized before calling this function.

Arguments:
- vec: Vector to project (GPU array, modified in-place)
- bl, bu: Lower and upper bounds (GPU arrays)
- D_scaled, D_scaled_squared, D_scaled_mul_x: Scaled diagonal matrices (GPU arrays)
- temp: Temporary storage (GPU array)
- t_warm_start: Warm start values (GPU array)
- cpu_head_start: Block head start indices (CPU array)
- gpu_ns: Block sizes (GPU array)
- cpu_ns: Block sizes (CPU array, used for block configuration)
- blkNum: Number of blocks
- cpu_proj_type: Projection type for each block (CPU array)
- abs_tol, rel_tol: Tolerances for projection

The native function pointer is initialized by the module's `__init__()` method.
The cuBLAS handle is initialized lazily on the first grid-wise projection.
"""
function few_block_proj(vec::T, bl::T, bu::T, D_scaled::T, D_scaled_squared::T, D_scaled_mul_x::T, temp::T, t_warm_start::T, cpu_head_start::Vector{Int64}, gpu_ns::CUDA.CuArray{Int64, 1, CUDA.DeviceMemory}, cpu_ns::Vector{Int64}, blkNum::Int64, cpu_proj_type::Vector{Int64}, abs_tol::Float64 = 1e-12, rel_tol::Float64 = 1e-12) where T<:CuArray
    # Calculate number of thread blocks based on maximum block size
    nThread = Int64(ThreadPerBlock)
    nBlock = cld(maximum(cpu_ns) + ThreadPerBlock + 1, ThreadPerBlock)
    
    # Get function pointer (must be initialized elsewhere, e.g., in __init__)
    fptr = few_block_proj_ptr[]
    fptr != C_NULL || error("few_block_proj not initialized. Did __init__() run?")
    # This persistent Julia-owned handle is created only on the first
    # grid-wise projection and reused throughout the optimization.
    gridWise_handle = get_gridWise_cublas_handle()
    # CUDA.jl kernels may have been queued on a task-local stream. Complete
    # them before entering libfew_block_proj.so, whose C++ launches and
    # default-stream cuBLAS handle are ordered with each other.
    CUDA.synchronize()
    
    # Call C function from shared library using @ccall macro
    # The function signature matches the C function in libfew_block_proj.so
    @ccall $fptr(gridWise_handle.handle::Ptr{Nothing}, # Julia-owned cuBLAS handle
                             vec::CuPtr{Cdouble},   # Vector to project
                             bl::CuPtr{Cdouble},    # Lower bounds
                             bu::CuPtr{Cdouble},    # Upper bounds
                             D_scaled::CuPtr{Cdouble}, 
                             D_scaled_squared::CuPtr{Cdouble}, 
                             D_scaled_mul_x::CuPtr{Cdouble}, 
                             temp::CuPtr{Cdouble}, 
                             t_warm_start::CuPtr{Cdouble}, 
                             cpu_head_start::Ptr{Clong},      # CPU array
                             gpu_ns::CuPtr{Clong},            # GPU array
                             cpu_ns::Ptr{Clong},              # CPU array
                             blkNum::Cint, 
                             cpu_proj_type::Ptr{Clong}, 
                             nThread::Cint, 
                             nBlock::Cint,
                             abs_tol::Cdouble,
                             rel_tol::Cdouble)::Cvoid
    
    # Synchronize to ensure kernel completion
    CUDA.synchronize()
end

"""Grid-wise projection: the GPU grid cooperates on a small number of cones."""
gridWise_block_proj(args...) = few_block_proj(args...)



# ============================================================================
# Section 4: Utility Kernels (from utils.ptx)
# ============================================================================
# These kernels implement various utility functions for the RPDHG algorithm,
# including primal/dual updates, reflection operations, averaging, and
# matrix operations. All kernels are loaded from a single PTX file: utils.ptx

# Path to the shared PTX file containing all utility kernels
utils_path = joinpath(MODULE_DIR, "cuda/utils.ptx")

# CUDA.jl's CPU CSC convenience upload and the historical utility PTX both
# assume 32-bit sparse indices. These kernels are compiled by CUDA.jl for the
# actual array types, so matrices with more than typemax(Int32) stored entries
# keep Int64 row pointers and column indices throughout preprocessing.
@inline function _sparse_thread_index(::Type{Ti}) where {Ti<:Integer}
    return _sparse_unsigned_thread_index(
        Ti,
        CUDA.blockIdx().x,
        CUDA.blockDim().x,
        CUDA.threadIdx().x,
    )
end

function _rescale_csr_sparse_kernel!(
    values,
    rowptr,
    colval,
    row_scaling,
    col_scaling,
    nrows,
)
    Ti = eltype(rowptr)
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
        for position in first_position:last_position
            column = @inbounds colval[position]
            @inbounds values[position] /= row_scaling[row] * col_scaling[column]
        end
    end
    return
end

function _max_abs_row_sparse_kernel!(values, rowptr, result, nrows)
    Ti = eltype(rowptr)
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
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
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
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
    Tu = unsigned(Ti)
    row_address = _sparse_thread_index(Ti)
    if row_address <= Tu(nrows)
        row = Ti(row_address)
        first_position = @inbounds rowptr[row_address]
        last_position = (@inbounds rowptr[row_address + one(Tu)]) - one(Ti)
        for position in first_position:last_position
            @inbounds row_indices[position] = row
        end
    end
    return
end

function _rescale_coo_sparse_kernel!(
    values,
    row_indices,
    col_indices,
    row_scaling,
    col_scaling,
    num_entries,
)
    Ti = eltype(row_indices)
    Tu = unsigned(Ti)
    position_address = _sparse_thread_index(Ti)
    if position_address <= Tu(num_entries)
        position = Ti(position_address)
        row = @inbounds row_indices[position]
        column = @inbounds col_indices[position]
        @inbounds values[position] /= row_scaling[row] * col_scaling[column]
    end
    return
end

function _max_abs_indexed_sparse_kernel!(values, indices, result, num_entries)
    Ti = eltype(indices)
    Tu = unsigned(Ti)
    position_address = _sparse_thread_index(Ti)
    if position_address <= Tu(num_entries)
        position = Ti(position_address)
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
    Tu = unsigned(Ti)
    position_address = _sparse_thread_index(Ti)
    if position_address <= Tu(num_entries)
        position = Ti(position_address)
        output_index = @inbounds indices[position]
        value = abs(@inbounds values[position])
        contribution = alpha == 1.0 ? value : value^alpha
        CUDA.@atomic result[output_index] += contribution
    end
    return
end

# ----------------------------------------------------------------------------
# Reflection Update Kernel
# ----------------------------------------------------------------------------
# Updates primal and dual solutions using reflection/extrapolation techniques
# in the aggressive-update path.

const _reflection_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _reflection_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _reflection_update_lock   = SpinLock()

const _reflection_update_path = utils_path
const _reflection_update_name = "reflection_update"

"""
    get_reflection_update_kernel() -> CuFunction

Lazily loads and returns the reflection update CUDA kernel.
Uses thread-safe lazy loading pattern.
"""
function get_reflection_update_kernel()::CuFunction
    k = _reflection_update_kernel[]
    k !== nothing && return k

    lock(_reflection_update_lock)
    try
        k = _reflection_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_reflection_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _reflection_update_name)

        _reflection_update_mod[]    = mod
        _reflection_update_kernel[] = fun
        return fun
    finally
        unlock(_reflection_update_lock)
    end
end

"""
    reflection_update(
        primal_sol,
        primal_sol_lag,
        primal_sol_mean,
        primal_halpern_candidate,
        primal_restart_anchor,
        dual_sol,
        dual_sol_lag,
        dual_sol_mean,
        dual_halpern_candidate,
        dual_restart_anchor,
        extra_coeff,
        primal_n,
        dual_n,
        inner_iter,
        eta_cum,
        eta,
        use_reflection,
        use_halpern,
    )

Updates the reflected main trajectory, its running mean, and an auxiliary
Halpern restart candidate.

This kernel implements the reflection step in the aggressive-update path.
The reflection uses extrapolation
coefficients (eta, eta_cum) to combine current and lagged solutions.

Arguments:
- primal_sol: Current primal solution (GPU array, modified in-place)
- primal_sol_lag: Lagged primal solution (GPU array)
- primal_sol_mean: Running average of primal solution (GPU array, modified in-place)
- primal_halpern_candidate: Auxiliary primal Halpern restart candidate
- primal_restart_anchor: Primal restart anchor from Algorithm 1
- dual_sol: Current dual solution (GPU array, modified in-place)
- dual_sol_lag: Lagged dual solution (GPU array)
- dual_sol_mean: Running average of dual solution (GPU array, modified in-place)
- dual_halpern_candidate: Auxiliary dual Halpern restart candidate
- dual_restart_anchor: Dual restart anchor from Algorithm 1
- extra_coeff: Extra extrapolation coefficient
- primal_n: Dimension of primal variable
- dual_n: Dimension of dual variable
- inner_iter: Current inner iteration number
- eta_cum: Cumulative extrapolation coefficient
- eta: Current extrapolation coefficient
- use_reflection: Whether to apply the reflection/extrapolation term
- use_halpern: Whether to form the auxiliary Halpern candidate. The main
  trajectory is always the reflected point.
"""
function reflection_update(
    primal_sol::T,
    primal_sol_lag::T,
    primal_sol_mean::T,
    primal_halpern_candidate::T,
    primal_restart_anchor::T,
    dual_sol::T,
    dual_sol_lag::T,
    dual_sol_mean::T,
    dual_halpern_candidate::T,
    dual_restart_anchor::T,
    extra_coeff::Float64,
    primal_n::Int64,
    dual_n::Int64,
    inner_iter::Int64,
    eta_cum::Float64,
    eta::Float64,
    use_reflection::Bool,
    use_halpern::Bool,
) where T<:CuArray
    # Calculate blocks based on maximum dimension (primal or dual)
    nBlock = cld(max(primal_n, dual_n), ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(
            get_reflection_update_kernel(),
            (
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                CuPtr{Float64},
                Float64,
                Int64,
                Int64,
                Int64,
                Float64,
                Float64,
                Int32,
                Int32,
            ),
            primal_sol,
            primal_sol_lag,
            primal_sol_mean,
            primal_halpern_candidate,
            primal_restart_anchor,
            dual_sol,
            dual_sol_lag,
            dual_sol_mean,
            dual_halpern_candidate,
            dual_restart_anchor,
            extra_coeff,
            primal_n,
            dual_n,
            inner_iter,
            eta_cum,
            eta,
            Int32(use_reflection),
            Int32(use_halpern);
            blocks = nBlock,
            threads = ThreadPerBlock,
        )
    end
end


# ----------------------------------------------------------------------------
# Primal Update Kernel
# ----------------------------------------------------------------------------
# Updates the primal variable in the RPDHG algorithm.
# Implements: x^{k+1} = x^k - tau * (c + G^T * y^k + d_c)

const _primal_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _primal_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _primal_update_lock   = SpinLock()

const _primal_update_path = utils_path
const _primal_update_name = "primal_update"

"""
    get_primal_update_kernel() -> CuFunction

Lazily loads and returns the primal update CUDA kernel.
"""
function get_primal_update_kernel()::CuFunction
    k = _primal_update_kernel[]
    k !== nothing && return k

    lock(_primal_update_lock)
    try
        k = _primal_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_primal_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _primal_update_name)

        _primal_update_mod[]    = mod
        _primal_update_kernel[] = fun
        return fun
    finally
        unlock(_primal_update_lock)
    end
end

"""
    primal_update(primal_sol, primal_sol_lag, primal_sol_diff, d_c, tau, n)

Updates the primal variable in the RPDHG algorithm.

Performs the primal update step: x^{k+1} = x^k - tau * (c + G^T * y^k + d_c)
where tau is the primal step size.

Arguments:
- primal_sol: Current primal solution (GPU array, modified in-place)
- primal_sol_lag: Previous primal solution (GPU array, modified in-place)
- primal_sol_diff: Difference vector (GPU array, modified in-place)
- d_c: Scaled gradient term (GPU array)
- tau: Primal step size parameter
- n: Dimension of primal variable
"""
function primal_update(primal_sol::T, primal_sol_lag::T, primal_sol_diff::T, d_c::T, tau::Float64, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_primal_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        primal_sol, primal_sol_lag, primal_sol_diff, d_c, tau, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


# ----------------------------------------------------------------------------
# Dual Update Kernel
# ----------------------------------------------------------------------------
# Updates the dual variable in the RPDHG algorithm.
# Implements: y^{k+1} = y^k + sigma * (G * x^{k+1} - h + d_h)

const _dual_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _dual_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _dual_update_lock   = SpinLock()

const _dual_update_path = utils_path
const _dual_update_name = "dual_update"

"""
    get_dual_update_kernel() -> CuFunction

Lazily loads and returns the dual update CUDA kernel.
"""
function get_dual_update_kernel()::CuFunction
    k = _dual_update_kernel[]
    k !== nothing && return k

    lock(_dual_update_lock)
    try
        k = _dual_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_dual_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _dual_update_name)

        _dual_update_mod[]    = mod
        _dual_update_kernel[] = fun
        return fun
    finally
        unlock(_dual_update_lock)
    end
end

"""
    dual_update(dual_sol, dual_sol_lag, dual_sol_diff, d_h, sigma, n)

Updates the dual variable in the RPDHG algorithm.

Performs the dual update step: y^{k+1} = y^k + sigma * (G * x^{k+1} - h + d_h)
where sigma is the dual step size.

Arguments:
- dual_sol: Current dual solution (GPU array, modified in-place)
- dual_sol_lag: Previous dual solution (GPU array, modified in-place)
- dual_sol_diff: Difference vector (GPU array, modified in-place)
- d_h: Scaled constraint violation term (GPU array)
- sigma: Dual step size parameter
- n: Dimension of dual variable
"""
function dual_update(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, d_h::T, sigma::Float64, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_dual_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Float64, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, d_h, sigma, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ----------------------------------------------------------------------------
# Extrapolation Update Kernel
# ----------------------------------------------------------------------------
# Computes the extrapolation/difference between current and lagged primal solutions.
# Used by aggressive updates: x_diff = x - x_lag

const _extrapolation_update_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _extrapolation_update_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _extrapolation_update_lock   = SpinLock()

const _extrapolation_update_path = utils_path
const _extrapolation_update_name = "extrapolation_update"

"""
    get_extrapolation_update_kernel() -> CuFunction

Lazily loads and returns the extrapolation update CUDA kernel.
"""
function get_extrapolation_update_kernel()::CuFunction
    k = _extrapolation_update_kernel[]
    k !== nothing && return k

    lock(_extrapolation_update_lock)
    try
        k = _extrapolation_update_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_extrapolation_update_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _extrapolation_update_name)

        _extrapolation_update_mod[]    = mod
        _extrapolation_update_kernel[] = fun
        return fun
    finally
        unlock(_extrapolation_update_lock)
    end
end

"""
    extrapolation_update(primal_sol_diff, primal_sol, primal_sol_lag, n)

Computes the difference between current and lagged primal solutions.

Calculates: primal_sol_diff = primal_sol - primal_sol_lag
This difference is used by the aggressive-update path.

Arguments:
- primal_sol_diff: Output difference vector (GPU array, modified in-place)
- primal_sol: Current primal solution (GPU array)
- primal_sol_lag: Lagged primal solution (GPU array)
- n: Dimension of primal variable
"""
function extrapolation_update(primal_sol_diff::T, primal_sol::T, primal_sol_lag::T, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_extrapolation_update_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        primal_sol_diff, primal_sol, primal_sol_lag, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end


# ----------------------------------------------------------------------------
# Calculate Difference Kernel
# ----------------------------------------------------------------------------
# Computes differences between current and lagged solutions for both
# primal and dual variables. Used for tracking convergence and restart logic.

const _calculate_diff_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _calculate_diff_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _calculate_diff_lock   = SpinLock()

const _calculate_diff_path = utils_path
const _calculate_diff_name = "calculate_diff"

"""
    get_calculate_diff_kernel() -> CuFunction

Lazily loads and returns the calculate difference CUDA kernel.
"""
function get_calculate_diff_kernel()::CuFunction
    k = _calculate_diff_kernel[]
    k !== nothing && return k

    lock(_calculate_diff_lock)
    try
        k = _calculate_diff_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_calculate_diff_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _calculate_diff_name)

        _calculate_diff_mod[]    = mod
        _calculate_diff_kernel[] = fun
        return fun
    finally
        unlock(_calculate_diff_lock)
    end
end

"""
    calculate_diff(dual_sol, dual_sol_lag, dual_sol_diff, dual_n, primal_sol, primal_sol_lag, primal_sol_diff, primal_n)

Computes differences between current and lagged solutions for both primal and dual variables.

Calculates:
- dual_sol_diff = dual_sol - dual_sol_lag
- primal_sol_diff = primal_sol - primal_sol_lag

These differences are used for convergence monitoring and adaptive restart strategies.

Arguments:
- dual_sol: Current dual solution (GPU array)
- dual_sol_lag: Lagged dual solution (GPU array)
- dual_sol_diff: Dual difference vector (GPU array, modified in-place)
- dual_n: Dimension of dual variable
- primal_sol: Current primal solution (GPU array)
- primal_sol_lag: Lagged primal solution (GPU array)
- primal_sol_diff: Primal difference vector (GPU array, modified in-place)
- primal_n: Dimension of primal variable
"""
function calculate_diff(dual_sol::T, dual_sol_lag::T, dual_sol_diff::T, dual_n::Int64, primal_sol::T, primal_sol_lag::T, primal_sol_diff::T,  primal_n::Int64) where T<:CuArray
    # Use maximum dimension to determine number of blocks
    nBlock = cld(max(dual_n, primal_n) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_calculate_diff_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        dual_sol, dual_sol_lag, dual_sol_diff, dual_n, primal_sol, primal_sol_lag, primal_sol_diff, primal_n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ----------------------------------------------------------------------------
# AXPYZ Kernel (BLAS-like operation)
# ----------------------------------------------------------------------------
# Performs the operation: z = alpha * y + x
# This is a common linear algebra operation used throughout the algorithm.

const _axpyz_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _axpyz_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _axpyz_lock   = SpinLock()

const _axpyz_path = utils_path
const _axpyz_name = "axpyz"

"""
    get_axpyz_kernel() -> CuFunction

Lazily loads and returns the axpyz CUDA kernel.
"""
function get_axpyz_kernel()::CuFunction
    k = _axpyz_kernel[]
    k !== nothing && return k

    lock(_axpyz_lock)
    try
        k = _axpyz_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_axpyz_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _axpyz_name)

        _axpyz_mod[]    = mod
        _axpyz_kernel[] = fun
        return fun
    finally
        unlock(_axpyz_lock)
    end
end

"""
    axpyz(z, alpha, y, x, n)

Performs the BLAS-like operation: z = alpha * y + x

This is equivalent to: z[i] = alpha * y[i] + x[i] for all i.

Arguments:
- z: Output vector (GPU array, modified in-place)
- alpha: Scalar coefficient
- y: First input vector (GPU array)
- x: Second input vector (GPU array)
- n: Length of vectors
"""
function axpyz(z::T, alpha::Float64, y::T, x::T, n::Int64) where T<:CuArray
    nBlock = cld(n + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_axpyz_kernel(), 
        (CuPtr{Float64}, Float64, CuPtr{Float64}, CuPtr{Float64}, Int64), 
        z, alpha, y, x, n;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ----------------------------------------------------------------------------
# Average Sequence Kernel
# ----------------------------------------------------------------------------
# Computes running averages of primal and dual solutions.
# Used for averaging methods in RPDHG: mean = (k * mean + new_value) / (k + 1)

const _average_seq_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _average_seq_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _average_seq_lock   = SpinLock()

const _average_seq_path = utils_path
const _average_seq_name = "average_seq"

"""
    get_average_seq_kernel() -> CuFunction

Lazily loads and returns the average sequence CUDA kernel.
"""
function get_average_seq_kernel()::CuFunction
    k = _average_seq_kernel[]
    k !== nothing && return k

    lock(_average_seq_lock)
    try
        k = _average_seq_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_average_seq_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _average_seq_name)

        _average_seq_mod[]    = mod
        _average_seq_kernel[] = fun
        return fun
    finally
        unlock(_average_seq_lock)
    end
end

"""
    average_seq(; primal_sol_mean, primal_sol, primal_n, dual_sol_mean, dual_sol, dual_n, inner_iter)

Updates running averages of primal and dual solutions.

Implements exponential moving average:
- primal_sol_mean = (inner_iter * primal_sol_mean + primal_sol) / (inner_iter + 1)
- dual_sol_mean = (inner_iter * dual_sol_mean + dual_sol) / (inner_iter + 1)

Arguments:
- primal_sol_mean: Running average of primal solution (GPU array, modified in-place)
- primal_sol: Current primal solution (GPU array)
- primal_n: Dimension of primal variable
- dual_sol_mean: Running average of dual solution (GPU array, modified in-place)
- dual_sol: Current dual solution (GPU array)
- dual_n: Dimension of dual variable
- inner_iter: Current inner iteration number (used as weight)
"""
function average_seq(; primal_sol_mean::T, primal_sol::T, primal_n::Int64, dual_sol_mean::T, dual_sol::T, dual_n::Int64, inner_iter::Int64) where T<:CuArray
    nBlock = cld(max(primal_n, dual_n) + ThreadPerBlock - 1, ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(get_average_seq_kernel(), 
        (CuPtr{Float64}, CuPtr{Float64}, Int64, CuPtr{Float64}, CuPtr{Float64}, Int64, Int64), 
        primal_sol_mean, primal_sol, primal_n, dual_sol_mean, dual_sol, dual_n, inner_iter;
        blocks = nBlock, threads = ThreadPerBlock)
    end
end

# ============================================================================
# Section 5: Matrix Scaling and Norm Computation Kernels
# ============================================================================
# These kernels are used for preconditioning and scaling operations on
# sparse matrices stored in CSR (Compressed Sparse Row) format.

# ----------------------------------------------------------------------------
# Rescale CSR Matrix Kernel
# ----------------------------------------------------------------------------
# Scales a CSR matrix by row and column scaling factors.
# Performs: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]

const _rescale_csr_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _rescale_csr_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _rescale_csr_lock   = SpinLock()

const _rescale_csr_path = utils_path
const _rescale_csr_name = "rescale_csr"

"""
    get_rescale_csr_kernel() -> CuFunction

Lazily loads and returns the rescale CSR matrix CUDA kernel.
"""
function get_rescale_csr_kernel()::CuFunction
    k = _rescale_csr_kernel[]
    k !== nothing && return k

    lock(_rescale_csr_lock)
    try
        k = _rescale_csr_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_rescale_csr_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _rescale_csr_name)

        _rescale_csr_mod[]    = mod
        _rescale_csr_kernel[] = fun
        return fun
    finally
        unlock(_rescale_csr_lock)
    end
end

"""
    rescale_csr(d_G, row_scaling, col_scaling, m, n)

Scales a CSR sparse matrix by row and column scaling factors.

Performs element-wise scaling: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]
This is used for matrix preconditioning to improve numerical conditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR, modified in-place)
- row_scaling: Row scaling factors (GPU array)
- col_scaling: Column scaling factors (GPU array)
- m: Number of rows
- n: Number of columns
"""
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

# ----------------------------------------------------------------------------
# Replace Infinity with Zero Kernel
# ----------------------------------------------------------------------------
# Replaces infinite values in bound arrays with zero.
# Used to handle unbounded variables (Inf bounds) in the optimization problem.

const _replace_inf_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _replace_inf_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _replace_inf_lock   = SpinLock()

const _replace_inf_path = utils_path
const _replace_inf_name = "replace_inf_with_zero"

"""
    get_replace_inf_kernel() -> CuFunction

Lazily loads and returns the replace infinity kernel.
"""
function get_replace_inf_kernel()::CuFunction
    k = _replace_inf_kernel[]
    k !== nothing && return k

    lock(_replace_inf_lock)
    try
        k = _replace_inf_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_replace_inf_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _replace_inf_name)

        _replace_inf_mod[]    = mod
        _replace_inf_kernel[] = fun
        return fun
    finally
        unlock(_replace_inf_lock)
    end
end

"""
    replace_inf_with_zero(bl, bu, n)

Replaces infinite values in bound arrays with zero.

For unbounded variables, bounds are set to Inf. This kernel replaces
Inf values with 0 to avoid numerical issues in GPU computations.

Arguments:
- bl: Lower bounds (GPU array, modified in-place)
- bu: Upper bounds (GPU array, modified in-place)
- n: Length of bound arrays
"""
function replace_inf_with_zero(bl::CuArray{Float64,1}, bu::CuArray{Float64,1}, n::Int)
    threads = ThreadPerBlock
    blocks  = cld(n, threads)

    k = get_replace_inf_kernel()

    CUDA.@sync CUDA.cudacall(
        k,
        (CuPtr{Cdouble}, CuPtr{Cdouble}, Clong),
        bl, bu, Clong(n);
        threads=threads, blocks=blocks
    )
    return nothing
end



# ----------------------------------------------------------------------------
# Max Absolute Row Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each row of a CSR sparse matrix.
# Used for row scaling in preconditioning: result[i] = max_j |G[i,j]|

const _max_abs_row_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_row_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_row_lock   = SpinLock()

const _max_abs_row_path = utils_path
const _max_abs_row_name = "max_abs_row_kernel"

"""
    get_max_abs_row_kernel() -> CuFunction

Lazily loads and returns the max absolute row CUDA kernel.
"""
function get_max_abs_row_kernel()::CuFunction
    k = _max_abs_row_kernel[]
    k !== nothing && return k

    lock(_max_abs_row_lock)
    try
        k = _max_abs_row_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_row_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_row_name)

        _max_abs_row_mod[]    = mod
        _max_abs_row_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_row_lock)
    end
end

"""
    max_abs_row(d_G, result)

Computes the maximum absolute value in each row of a CSR sparse matrix.

For each row i, computes: result[i] = max_j |G[i,j]|
This is used for row scaling in matrix preconditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- result: Output vector of row maxima (GPU array, modified in-place)
          The function resets it to 0.0 before computing the maxima
"""
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


# ----------------------------------------------------------------------------
# Max Absolute Column Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each column of a CSR sparse matrix.
# Used for column scaling in preconditioning: result[j] = max_i |G[i,j]|

const _max_abs_col_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_col_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_col_lock   = SpinLock()

const _max_abs_col_path = utils_path
const _max_abs_col_name = "max_abs_col_kernel"

"""
    get_max_abs_col_kernel() -> CuFunction

Lazily loads and returns the max absolute column CUDA kernel.
"""
function get_max_abs_col_kernel()::CuFunction
    k = _max_abs_col_kernel[]
    k !== nothing && return k

    lock(_max_abs_col_lock)
    try
        k = _max_abs_col_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_col_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_col_name)

        _max_abs_col_mod[]    = mod
        _max_abs_col_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_col_lock)
    end
end

"""
    max_abs_col(d_G, result)

Computes the maximum absolute value in each column of a CSR sparse matrix.

For each column j, computes: result[j] = max_i |G[i,j]|
This is used for column scaling in matrix preconditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- result: Output vector of column maxima (GPU array, modified in-place)
          The function resets it to 0.0 before computing the maxima
"""
function max_abs_col(d_G, result)
    return max_abs_col_elementwise(d_G, result)
end



# ----------------------------------------------------------------------------
# Alpha Norm Row Kernel
# ----------------------------------------------------------------------------
# Computes the alpha-norm (L_alpha norm) for each row of a CSR sparse matrix.
# For alpha=1, this is the L1 norm (sum of absolute values).
# Used for row scaling in preconditioning: result[i] = (sum_j |G[i,j]|^alpha)^(1/alpha)

const _alpha_norm_row_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_row_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_row_lock   = SpinLock()

const _alpha_norm_row_path = utils_path
const _alpha_norm_row_name = "alpha_norm_row_kernel"

"""
    get_alpha_norm_row_kernel() -> CuFunction

Lazily loads and returns the alpha norm row CUDA kernel.
"""
function get_alpha_norm_row_kernel()::CuFunction
    k = _alpha_norm_row_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_row_lock)
    try
        k = _alpha_norm_row_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_alpha_norm_row_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _alpha_norm_row_name)

        _alpha_norm_row_mod[]    = mod
        _alpha_norm_row_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_row_lock)
    end
end

"""
    alpha_norm_row(d_G, alpha, result)

Computes the alpha-norm for each row of a CSR sparse matrix.

For each row i, computes: result[i] = (sum_j |G[i,j]|^alpha)^(1/alpha)
When alpha=1, this is the L1 norm (sum of absolute values).
This is used for row scaling in matrix preconditioning.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- alpha: Norm parameter (typically 1.0 for L1 norm)
- result: Output vector of row norms (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
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



# ----------------------------------------------------------------------------
# Alpha Norm Column Kernel
# ----------------------------------------------------------------------------
# Computes the alpha-norm (L_alpha norm) for each column of a CSR sparse matrix.
# For alpha=1, this is the L1 norm (sum of absolute values).
# Used for column scaling in preconditioning: result[j] = (sum_i |G[i,j]|^alpha)^(1/alpha)

const _alpha_norm_col_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_col_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_col_lock   = SpinLock()

const _alpha_norm_col_path = utils_path
const _alpha_norm_col_name = "alpha_norm_col_kernel"

"""
    get_alpha_norm_col_kernel() -> CuFunction

Lazily loads and returns the alpha norm column CUDA kernel.
"""
function get_alpha_norm_col_kernel()::CuFunction
    k = _alpha_norm_col_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_col_lock)
    try
        k = _alpha_norm_col_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_alpha_norm_col_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _alpha_norm_col_name)

        _alpha_norm_col_mod[]    = mod
        _alpha_norm_col_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_col_lock)
    end
end

"""
    alpha_norm_col(d_G, alpha, result)

Computes the alpha-norm for each column of a CSR sparse matrix.

For each column j, computes: result[j] = (sum_i |G[i,j]|^alpha)^(1/alpha)
When alpha=1, this is the L1 norm (sum of absolute values).
This is used for column scaling in matrix preconditioning.

Note: When alpha=1.0, the final power operation (^(1/alpha)) is skipped
since it's just the identity operation.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- alpha: Norm parameter (typically 1.0 for L1 norm)
- result: Output vector of column norms (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
function alpha_norm_col(d_G, alpha, result)
    return alpha_norm_col_elementwise(d_G, alpha, result)
end



# ----------------------------------------------------------------------------
# Get Row Index Kernel
# ----------------------------------------------------------------------------
# Computes the row index for each non-zero element in a CSR sparse matrix.
# This is useful for elementwise operations that need to know which row
# each non-zero element belongs to.

const _get_row_index_path = utils_path
const _get_row_index_name = "get_row_index"
const _get_row_index_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _get_row_index_lock   = SpinLock()
const _get_row_index_mod    = Ref{Union{Nothing,CuModule}}(nothing)

"""
    get_row_index_kernel() -> CuFunction

Lazily loads and returns the get row index CUDA kernel.
"""
function get_row_index_kernel()::CuFunction
    k = _get_row_index_kernel[]
    k !== nothing && return k

    lock(_get_row_index_lock)
    try
        k = _get_row_index_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_get_row_index_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _get_row_index_name)

        _get_row_index_mod[]    = mod
        _get_row_index_kernel[] = fun
        return fun
    finally
        unlock(_get_row_index_lock)
    end
end

"""
    get_row_index(d_G, row_idx)

Computes the row index for each non-zero element in a CSR sparse matrix.

For each non-zero element at position k in the CSR format, computes which row
it belongs to. This is useful for elementwise operations that need row information.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- row_idx: Output array of row indices (GPU array, modified in-place)
           Length should equal the number of non-zero elements (nnz)
"""
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



# ----------------------------------------------------------------------------
# Rescale COO (Coordinate) Format Kernel
# ----------------------------------------------------------------------------
# Scales a CSR matrix by row and column scaling factors, using row indices
# computed from the CSR format. Similar to rescale_csr but uses precomputed
# row indices for better performance in elementwise operations.
# Performs: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]

const _rescale_coo_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _rescale_coo_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _rescale_coo_lock   = SpinLock()

const _rescale_coo_path = utils_path
const _rescale_coo_name = "rescale_coo"

"""
    get_rescale_coo_kernel() -> CuFunction

Lazily loads and returns the rescale COO CUDA kernel.
"""
function get_rescale_coo_kernel()::CuFunction
    k = _rescale_coo_kernel[]
    k !== nothing && return k

    lock(_rescale_coo_lock)
    try
        k = _rescale_coo_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_rescale_coo_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _rescale_coo_name)

        _rescale_coo_mod[]    = mod
        _rescale_coo_kernel[] = fun
        return fun
    finally
        unlock(_rescale_coo_lock)
    end
end

"""
    rescale_coo(d_G, row_scaling, col_scaling, m, n, row_idx)

Scales a CSR sparse matrix by row and column scaling factors using precomputed row indices.

Performs element-wise scaling: G[i,j] = row_scaling[i] * G[i,j] * col_scaling[j]
This version uses precomputed row indices (from get_row_index) for better performance
when performing multiple scaling operations.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR, modified in-place)
- row_scaling: Row scaling factors (GPU array)
- col_scaling: Column scaling factors (GPU array)
- m: Number of rows
- n: Number of columns
- row_idx: Precomputed row indices for each non-zero element (GPU array)
           Should be computed using get_row_index() before calling this function
"""
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




# ============================================================================
# Section 6: Elementwise Operations
# ============================================================================
# These kernels perform elementwise operations on sparse matrices using
# precomputed row indices. They are optimized for cases where row indices
# are already known, avoiding repeated computation.

# ----------------------------------------------------------------------------
# Max Absolute Row Elementwise Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each row using elementwise operations
# with precomputed row indices. More efficient than max_abs_row when row
# indices are already available.

const _max_abs_row_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_row_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_row_elementwise_lock   = SpinLock()

const _max_abs_row_elementwise_path = utils_path
const _max_abs_row_elementwise_name = "max_abs_row_elementwise_kernel"

"""
    get_max_abs_row_elementwise_kernel() -> CuFunction

Lazily loads and returns the max absolute row elementwise CUDA kernel.
"""
function get_max_abs_row_elementwise_kernel()::CuFunction
    k = _max_abs_row_elementwise_kernel[]
    k !== nothing && return k

    lock(_max_abs_row_elementwise_lock)
    try
        k = _max_abs_row_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_row_elementwise_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_row_elementwise_name)

        _max_abs_row_elementwise_mod[]    = mod
        _max_abs_row_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_row_elementwise_lock)
    end
end

"""
    max_abs_row_elementwise(d_G, row_idx, result)

Computes the maximum absolute value in each row using elementwise operations.

For each row i, computes: result[i] = max_j |G[i,j]|
This version uses precomputed row indices for better performance when
performing multiple row-wise operations.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- row_idx: Precomputed row indices for each non-zero element (GPU array)
           Should be computed using get_row_index() before calling
- result: Output vector of row maxima (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
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


# ----------------------------------------------------------------------------
# Max Absolute Column Elementwise Kernel
# ----------------------------------------------------------------------------
# Computes the maximum absolute value in each column using elementwise operations.
# More efficient than max_abs_col when processing elements directly.

const _max_abs_col_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _max_abs_col_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _max_abs_col_elementwise_lock   = SpinLock()

const _max_abs_col_elementwise_path = utils_path
const _max_abs_col_elementwise_name = "max_abs_col_elementwise_kernel"

"""
    get_max_abs_col_elementwise_kernel() -> CuFunction

Lazily loads and returns the max absolute column elementwise CUDA kernel.
"""
function get_max_abs_col_elementwise_kernel()::CuFunction
    k = _max_abs_col_elementwise_kernel[]
    k !== nothing && return k

    lock(_max_abs_col_elementwise_lock)
    try
        k = _max_abs_col_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_max_abs_col_elementwise_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _max_abs_col_elementwise_name)

        _max_abs_col_elementwise_mod[]    = mod
        _max_abs_col_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_max_abs_col_elementwise_lock)
    end
end

"""
    max_abs_col_elementwise(d_G, result)

Computes the maximum absolute value in each column using elementwise operations.

For each column j, computes: result[j] = max_i |G[i,j]|
This version processes elements directly from the CSR format, which can be
more efficient than the row-based approach for certain matrix structures.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- result: Output vector of column maxima (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
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


# ----------------------------------------------------------------------------
# Alpha Norm Column Elementwise Kernel
# ----------------------------------------------------------------------------
# Computes the alpha-norm for each column using elementwise operations.
# More efficient than alpha_norm_col when processing elements directly.

const _alpha_norm_col_elementwise_mod    = Ref{Union{Nothing,CuModule}}(nothing)
const _alpha_norm_col_elementwise_kernel = Ref{Union{Nothing,CuFunction}}(nothing)
const _alpha_norm_col_elementwise_lock   = SpinLock()

const _alpha_norm_col_elementwise_path = utils_path
const _alpha_norm_col_elementwise_name = "alpha_norm_col_elementwise_kernel"

"""
    get_alpha_norm_col_elementwise_kernel() -> CuFunction

Lazily loads and returns the alpha norm column elementwise CUDA kernel.
"""
function get_alpha_norm_col_elementwise_kernel()::CuFunction
    k = _alpha_norm_col_elementwise_kernel[]
    k !== nothing && return k

    lock(_alpha_norm_col_elementwise_lock)
    try
        k = _alpha_norm_col_elementwise_kernel[]
        k !== nothing && return k

        CUDA.functional() || error("CUDA is not functional")
        CUDA.zeros(Float32, 1)

        bytes = read(_alpha_norm_col_elementwise_path)
        mod   = CuModule(bytes)
        fun   = CuFunction(mod, _alpha_norm_col_elementwise_name)

        _alpha_norm_col_elementwise_mod[]    = mod
        _alpha_norm_col_elementwise_kernel[] = fun
        return fun
    finally
        unlock(_alpha_norm_col_elementwise_lock)
    end
end

"""
    alpha_norm_col_elementwise(d_G, alpha, result)

Computes the alpha-norm for each column using elementwise operations.

For each column j, computes: result[j] = (sum_i |G[i,j]|^alpha)^(1/alpha)
When alpha=1, this is the L1 norm (sum of absolute values).
This version processes elements directly from the CSR format.

Note: When alpha=1.0, the final power operation (^(1/alpha)) is skipped
since it's just the identity operation.

Arguments:
- d_G: CSR sparse matrix on GPU (CuSparseMatrixCSR)
- alpha: Norm parameter (typically 1.0 for L1 norm)
- result: Output vector of column norms (GPU array, modified in-place)
          Should be initialized to 0.0 before calling
"""
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
