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
