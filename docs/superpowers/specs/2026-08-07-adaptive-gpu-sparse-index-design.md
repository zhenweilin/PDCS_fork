# Adaptive GPU Sparse Index Design

## Context

PDCS_GPU previously uploaded a CPU CSC matrix with CUDA.jl's convenience
`CuSparseMatrixCSR(G)` constructor. That constructor uses 32-bit sparse indices
and fails when the one-based CSR end pointer, `nnz(G) + 1`, exceeds
`typemax(Int32)`. The first overflow observed was
`trunc(Int32, 2147483648)`.

The initial overflow fix forced every uploaded CSR matrix and its preprocessing
kernels to use `Int64`. This is correct for very large matrices but costs an
extra four bytes for every CSR row pointer and column index on ordinary models.
For a `Float64` CSR matrix, the main value-plus-column-index storage increases
from approximately 12 bytes to 16 bytes per nonzero.

The dataset loader's existing `index_type` option is not sufficient. It controls
the input LIBSVM CSC representation, while the PDCS MOI handoff currently
materializes its canonical sparse matrix with `Int64` indices. GPU sparse index
selection must therefore happen in PDCS_GPU itself.

## Public Interface

PDCS_GPU will accept a `sparse_index_type` option with three supported values:

- `:auto` (default): choose `Int32` when the matrix is representable, otherwise
  choose `Int64`.
- `Int32`: require 32-bit GPU sparse indices and reject an unrepresentable
  matrix before uploading it.
- `Int64`: always use 64-bit GPU sparse indices.

The option will be available both as a keyword to `rpdhg_gpu_solve` and as the
JuMP/MOI raw optimizer attribute `"sparse_index_type"`:

```julia
set_optimizer_attribute(model, "sparse_index_type", :auto)
```

The accepted aliases are `:auto`, `:int32`, `:int64`, `"auto"`, `"int32"`,
`"int64"`, `Int32`, and `Int64`. They are normalized by one shared resolver;
the internal resolved value is always the type `Int32` or `Int64`.

## Resolution and Validation

A matrix is considered representable with 32-bit sparse indices only when all
of the following fit in `Int32`:

- the row count `m`;
- the column count `n`;
- the one-based terminal pointer `nnz(G) + 1`.

For `:auto`, PDCS_GPU selects `Int32` when these checks pass and `Int64`
otherwise. Forced `Int32` performs the same checks and throws an
`ArgumentError` explaining which matrix dimensions or nonzero count require
`Int64`. Unsupported option values also throw `ArgumentError` before any GPU
allocation.

## Upload Path

For the resolved `Int32` path, PDCS_GPU will use CUDA.jl's existing convenience
CSC-to-CSR upload. This avoids an unnecessary CPU CSR transpose and preserves
the efficient path used before the overflow fix.

For the resolved `Int64` path, PDCS_GPU will convert the CPU CSC matrix into
one-based CSR components with `Int64` row pointers and column indices, upload
those arrays, and explicitly construct
`CuSparseMatrixCSR{Float64,Int64}`. This bypasses CUDA.jl's implicit `Cint`
conversion.

The resolved type is passed through `rpdhg_gpu_solve` to `coeffUnion`; it is not
inferred from the CPU matrix's storage type because the MOI cache may use
`Int64` even when the actual matrix fits in `Int32`.

## GPU Preprocessing Kernels

Sparse preprocessing kernels will be generic over the CSR index element type.
CUDA.jl will compile specialized Int32 and Int64 versions from the same Julia
kernel definitions. Signed CSR payload storage, CSR offsets, column indices,
loop bounds, and atomic output indices use that resolved type rather than being
unconditionally converted to `Int64`. Launch positions and CSR array addresses
instead use the same-width unsigned companion (`UInt32` for `Int32`, `UInt64`
for `Int64`): this safely rejects padded lanes above `typemax(Ti)` and addresses
the `m + 1` CSR row-pointer slot when `m == typemax(Ti)`. A launch position is
converted back to signed `Ti` only after the unsigned bound check succeeds.

Temporary sparse-index arrays, including the COO-style row-index workspace used
by preprocessing, will use `eltype(d_G.rowPtr)`. Floating-point vectors and
matrix values remain `Float64`; this design changes only sparse index storage
and sparse index arithmetic.

The historical PTX preprocessing entry points will not be used for these
operations because their C signatures assume 32-bit `int` indices and the
current Julia wrappers do not consistently match those signatures. Runtime
CUDA.jl kernels provide type-safe specialization without requiring a local
`nvcc` rebuild.

## Compatibility

- Existing users that set no option receive `:auto` and use Int32 for ordinary
  models.
- Models exceeding the Int32 boundary transparently use Int64 under `:auto`.
- Users may force a type for reproducible performance experiments.
- Existing callers that supply an already-created GPU CSR matrix keep its
  existing index type; no conversion is performed.
- CUDA 11 or newer is required for CUSPARSE operations with Int64 sparse
  indices, matching CUDA.jl's documented requirement.

## Testing

Host-side unit tests will cover:

1. `:auto` resolves to `Int32` for a small matrix.
2. Boundary metadata that needs `nnz + 1 == 2^31` resolves to `Int64` without
   allocating such a matrix.
3. Forced `Int32` rejects the same boundary with a clear error.
4. Forced `Int64` remains Int64 for a small matrix.
5. Invalid option values are rejected.
6. CSC-to-CSR components preserve values, dimensions, and the requested index
   type.
7. GPU preprocessing source uses generic typed kernels and allocates temporary
   row indices with the CSR index type.

Regression verification will include the bulk-cache tests, realistic Lasso
handoff test, CPU-only import test, source parsing, and CUDA.jl 6.2.1 macro
expansion on the local CPU-only machine. A CUDA machine is required for the
final numerical tests of both actual GPU CSR variants and the original matrix
whose nonzero count exceeds the Int32 boundary.
