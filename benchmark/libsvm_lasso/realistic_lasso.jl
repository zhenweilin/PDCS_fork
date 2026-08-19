module RealisticLasso

using JuMP
using LinearAlgebra
using SparseArrays
using TOML

const ROOT_DIR = @__DIR__
const DEFAULT_CATALOG = joinpath(ROOT_DIR, "datasets.toml")
const DEFAULT_RAW_DIR = joinpath(ROOT_DIR, "raw")

struct DatasetSpec
    id::String
    display_name::String
    path::String
    url::String
    container::Symbol
    archive_member::Union{Nothing, String}
    expected_rows::Int64
    expected_features::Int64
    expected_bytes::Int64
    label_mode::Symbol
    source::String
    official_page::String
    notes::String
end

struct LassoData{Tv, Ti<:Integer}
    dataset_id::String
    A::SparseMatrixCSC{Tv, Ti}
    b::Vector{Tv}
    lambda_reference::Float64
    original_features::Int64
    row_limit::Union{Nothing, Int64}
end

mutable struct LassoSOCPModel
    model::JuMP.Model
    x::Vector{JuMP.VariableRef}
    u::Vector{JuMP.VariableRef}
    r::JuMP.VariableRef
    lambda::Float64
end

struct LassoConicBlock
    set_type::DataType
    dimension::Int
    first_row::Int
end

struct LassoConicTimings
    assembly_seconds::Float64
end

mutable struct LassoConicData{Ti<:Integer}
    objective_sense::MOI.OptimizationSense
    objective_constant::Float64
    objective_coefficients::Vector{Float64}
    variable_lower::Vector{Float64}
    variable_upper::Vector{Float64}
    num_rows::Int
    num_variables::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Float64}
    affine_constants::Vector{Float64}
    cone_blocks::Vector{LassoConicBlock}
    layout::Nothing
    timings::LassoConicTimings
    num_observations::Int
    num_features::Int
    x_range::UnitRange{Int}
    epigraph_range::UnitRange{Int}
    r_index::Int
    penalty::Float64
    formulation::Symbol
end

function dataset_specs(; catalog::AbstractString = DEFAULT_CATALOG,
                         raw_dir::AbstractString = DEFAULT_RAW_DIR)
    config = TOML.parsefile(catalog)
    specs = Dict{String, DatasetSpec}()
    for (id, entry) in config["datasets"]
        member = get(entry, "archive_member", nothing)
        label_mode_raw = get(entry, "label_mode", "pm_one")
        label_mode = Symbol(label_mode_raw)
        specs[id] = DatasetSpec(
            id,
            entry["display_name"],
            joinpath(raw_dir, entry["file"]),
            entry["url"],
            Symbol(entry["container"]),
            member === nothing ? nothing : String(member),
            Int64(entry["rows"]),
            Int64(entry["features"]),
            Int64(entry["compressed_bytes"]),
            label_mode,
            entry["source"],
            entry["official_page"],
            entry["notes"],
        )
    end
    return specs
end

function dataset_spec(id::AbstractString; kwargs...)
    specs = dataset_specs(; kwargs...)
    haskey(specs, id) || error("Unknown dataset '$id'. Available: $(join(sort!(collect(keys(specs))), ", ")).")
    return specs[id]
end

function validate_download(spec::DatasetSpec)
    isfile(spec.path) || error("Missing $(spec.path). Run scripts/download.sh first.")
    actual = filesize(spec.path)
    actual == spec.expected_bytes || error(
        "Incomplete download $(spec.path): expected $(spec.expected_bytes) bytes, got $actual. " *
        "Run scripts/download.sh again to resume it."
    )
    return true
end

function stream_command(spec::DatasetSpec; validate::Bool = true)
    validate && validate_download(spec)
    if spec.container == :xz
        return `xz -dc $(spec.path)`
    elseif spec.container == :bz2
        return `bzip2 -dc $(spec.path)`
    elseif spec.container == :tar_xz
        spec.archive_member === nothing && error("No archive_member configured for $(spec.id).")
        return `tar -xJOf $(spec.path) $(spec.archive_member)`
    end
    error("Unsupported container $(spec.container) for $(spec.id); expected xz, bz2, or tar_xz.")
end

function with_libsvm_stream(
    f::Function,
    spec::DatasetSpec;
    validate::Bool = true,
    allow_early_close::Bool = false,
)
    command = stream_command(spec; validate)
    # tar reports its expected SIGPIPE as a write error when max_rows stops
    # early. Suppress that noise only on this explicitly allowed path.
    process = open(allow_early_close ? pipeline(command; stderr = devnull) : command, "r")
    result = try
        f(process)
    finally
        # Closing before wait lets xz/tar observe SIGPIPE when max_rows causes
        # an intentional early stop, and ensures the child is always reaped.
        close(process)
        wait(process)
    end
    if !success(process) && !allow_early_close
        error("Decompression failed for $(spec.path): $(process.cmd)")
    end
    return result
end

@inline function normalized_label(raw::Float64, mode::Symbol)
    mode == :pm_one && return raw > 0.0 ? 1.0 : -1.0
    mode == :raw && return raw
    error("label_mode must be :pm_one or :raw, got $mode.")
end

function parse_feature(token::AbstractString)
    separator = findfirst(==(':'), token)
    separator === nothing && error("Malformed LIBSVM feature token '$token'.")
    index = parse(Int64, SubString(token, firstindex(token), prevind(token, separator)))
    value = parse(Float64, SubString(token, nextind(token, separator), lastindex(token)))
    return index, value
end

function foreach_libsvm_row(
    f::Function,
    spec::DatasetSpec;
    max_rows::Union{Nothing, Integer} = nothing,
    label_mode::Symbol = :pm_one,
    validate::Bool = true,
    progress_every::Int64 = 1_000_000,
)
    max_rows !== nothing && max_rows <= 0 && error("max_rows must be positive or nothing.")
    rows = Int64(0)
    with_libsvm_stream(
        spec;
        validate,
        allow_early_close = max_rows !== nothing,
    ) do io
        for line in eachline(io)
            isempty(line) && continue
            tokens = eachsplit(line)
            state = iterate(tokens)
            state === nothing && continue
            label_token, token_state = state
            label = normalized_label(parse(Float64, label_token), label_mode)
            rows += 1
            f(rows, label, tokens, token_state)
            if progress_every > 0 && rows % progress_every == 0
                println("$(spec.id): parsed $rows rows")
            end
            max_rows !== nothing && rows >= max_rows && break
        end
    end
    if max_rows !== nothing
        required_rows = min(Int64(max_rows), spec.expected_rows)
        rows == required_rows || error(
            "$(spec.id) ended after $rows rows; expected at least $required_rows."
        )
    end
    return rows
end

function scan_libsvm(
    spec::DatasetSpec;
    max_rows::Union{Nothing, Integer} = nothing,
    label_mode::Symbol = :pm_one,
    validate::Bool = true,
    keep_column_counts::Bool = true,
    keep_labels::Bool = true,
    progress_every::Int64 = 1_000_000,
)
    n = spec.expected_features
    column_counts = keep_column_counts ? zeros(Int64, n) : Int64[]
    atb = zeros(Float64, n)
    label_capacity = max_rows === nothing ? spec.expected_rows : min(Int64(max_rows), spec.expected_rows)
    labels = keep_labels ? Vector{Float64}(undef, label_capacity) : Float64[]
    nnz_seen = Int64(0)
    max_feature_seen = Int64(0)

    rows = foreach_libsvm_row(
        spec;
        max_rows,
        label_mode,
        validate,
        progress_every,
    ) do row, label, tokens, token_state
        keep_labels && (labels[row] = label)
        state = iterate(tokens, token_state)
        while state !== nothing
            token, token_state = state
            feature, value = parse_feature(token)
            1 <= feature <= n || error(
                "Feature index $feature in row $row is outside the declared range 1:$n."
            )
            keep_column_counts && (column_counts[feature] += 1)
            atb[feature] += value * label
            nnz_seen += 1
            max_feature_seen = max(max_feature_seen, feature)
            state = iterate(tokens, token_state)
        end
    end

    keep_labels && resize!(labels, rows)
    lambda_reference = maximum(abs, atb)
    active_columns = count(!iszero, atb)
    return (;
        rows,
        features = n,
        nnz = nnz_seen,
        max_feature_seen,
        lambda_reference,
        lambda_zero_threshold = 2.0 * lambda_reference,
        active_atb_columns = active_columns,
        column_counts,
        labels,
    )
end

function load_libsvm(
    spec::DatasetSpec;
    max_rows::Union{Nothing, Integer} = nothing,
    label_mode::Symbol = :pm_one,
    value_type::Type{Tv} = Float32,
    index_type::Type{Ti} = Int32,
    compact_zero_columns::Bool = false,
    validate::Bool = true,
    progress_every::Int64 = 1_000_000,
) where {Tv<:AbstractFloat, Ti<:Integer}
    scan = scan_libsvm(
        spec;
        max_rows,
        label_mode,
        validate,
        keep_column_counts = true,
        keep_labels = true,
        progress_every,
    )
    scan.rows <= typemax(Ti) || error("Row count $(scan.rows) does not fit in $Ti.")
    scan.features <= typemax(Ti) || error("Feature count $(scan.features) does not fit in $Ti.")
    scan.nnz + 1 <= typemax(Ti) || error(
        "nnz=$(scan.nnz) does not fit in $Ti CSC pointers; rerun with index_type=Int64."
    )

    original_n = scan.features
    selected_columns = compact_zero_columns ?
        findall(!iszero, scan.column_counts) :
        Base.OneTo(original_n)
    n = length(selected_columns)
    feature_map = if compact_zero_columns
        mapping = zeros(Ti, original_n)
        for (new_column, original_column) in enumerate(selected_columns)
            mapping[original_column] = Ti(new_column)
        end
        mapping
    else
        Ti[]
    end

    colptr = Vector{Ti}(undef, n + 1)
    offset = Int64(1)
    colptr[1] = one(Ti)
    for (new_column, original_column) in enumerate(selected_columns)
        offset += scan.column_counts[original_column]
        colptr[new_column + 1] = Ti(offset)
    end

    rowval = Vector{Ti}(undef, scan.nnz)
    nzval = Vector{Tv}(undef, scan.nnz)
    cursors = Vector{Int64}(undef, n)
    for column in 1:n
        cursors[column] = Int64(colptr[column])
    end

    rows_second_pass = foreach_libsvm_row(
        spec;
        max_rows,
        label_mode,
        validate,
        progress_every,
    ) do row, _, tokens, token_state
        state = iterate(tokens, token_state)
        while state !== nothing
            token, token_state = state
            feature, value = parse_feature(token)
            output_feature = compact_zero_columns ? Int64(feature_map[feature]) : feature
            output_feature > 0 || error("Internal compact-column mapping failure for feature $feature.")
            position = cursors[output_feature]
            rowval[position] = Ti(row)
            nzval[position] = Tv(value)
            cursors[output_feature] = position + 1
            state = iterate(tokens, token_state)
        end
    end
    rows_second_pass == scan.rows || error("The two parsing passes observed different row counts.")

    A = SparseMatrixCSC{Tv, Ti}(scan.rows, n, colptr, rowval, nzval)
    b = Tv.(scan.labels)
    row_limit = max_rows === nothing ? nothing : Int64(max_rows)
    return LassoData(spec.id, A, b, scan.lambda_reference, original_n, row_limit)
end

"""
    penalties(data, ratios)

Return `lambda = ratio * ||A'b||_inf`, matching the reference scale used by
the current PDCS manuscript. For the manuscript's loss `||Ax-b||^2`, the
all-zero solution begins at `lambda >= 2||A'b||_inf`.
"""
function penalties(data::LassoData, ratios)
    return [Float64(ratio) * data.lambda_reference for ratio in ratios]
end

"""
    build_lasso_socp(data; penalty_ratio=1.0, penalty=nothing, optimizer=nothing)

Build, but do not serialize, the compact SOCP:

    min ||A*x-b||_2^2 + lambda*||x||_1.

The reformulation uses `u >= abs(x)` and places `A*x-b` directly in the SOC
`[(1+r)/sqrt(2), (1-r)/sqrt(2), A*x-b] in Q`, with objective
`2r + lambda*sum(u)`.  It introduces neither split variables nor an explicit
residual variable.  If `penalty` is supplied it is the absolute lambda;
otherwise `lambda = penalty_ratio * ||A'b||_inf`.
"""
function build_lasso_socp(
    data::LassoData;
    penalty_ratio::Real = 1.0,
    penalty::Union{Nothing, Real} = nothing,
    optimizer = nothing,
)
    penalty_ratio > 0 || error("penalty_ratio must be positive.")
    lambda = penalty === nothing ? penalty_ratio * data.lambda_reference : Float64(penalty)
    lambda > 0 || error("penalty must be positive.")

    A = data.A
    b = data.b
    m, n = size(A)
    model = optimizer === nothing ? Model() : Model(optimizer)
    @variable(model, x[1:n])
    @variable(model, u[1:n] >= 0.0)
    @variable(model, r)
    @constraint(model, x .<= u)
    @constraint(model, -x .<= u)
    @constraint(
        model,
        vcat((1.0 + r) / sqrt(2.0), (1.0 - r) / sqrt(2.0), A * x - b) in
        SecondOrderCone(),
    )
    @objective(model, Min, 2.0 * r + lambda * sum(u))
    return LassoSOCPModel(model, x, u, r, lambda)
end

function _fill_lasso_feature_columns!(
    rowval,
    nzval,
    colptr,
    A,
    m::Int,
    n::Int,
    workers::Int,
)
    function fill_worker(worker)
        for column in worker:workers:n
            position = Int(colptr[column])
            rowval[position] = column
            nzval[position] = -1.0
            rowval[position + 1] = n + column
            nzval[position + 1] = 1.0
            position += 2
            for source in Int(A.colptr[column]):(Int(A.colptr[column + 1]) - 1)
                rowval[position] = 2 * n + 2 + Int(A.rowval[source])
                nzval[position] = Float64(A.nzval[source])
                position += 1
            end
        end
        return
    end
    if workers == 1
        fill_worker(1)
    else
        @sync for worker in 1:workers
            Threads.@spawn fill_worker(worker)
        end
    end
    return
end

"""
    build_lasso_conic_data(data; penalty_ratio=1.0, penalty=nothing,
                           index_type=Int64, workers=Threads.nthreads())

Build the compact Lasso SOCP directly as finalized CSC conic data for PDCS.
The only variables are `(x, u, r)`, the `2n` nonnegative rows encode
`u - x >= 0` and `u + x >= 0`, and `A*x-b` appears directly in the single SOC.
Thus the design matrix is stored once and there are no residual variables or
residual-defining equality rows.
"""
function build_lasso_conic_data(
    data::LassoData;
    penalty_ratio::Real = 1.0,
    penalty::Union{Nothing, Real} = nothing,
    index_type::Type{Ti} = Int64,
    workers::Integer = Threads.nthreads(),
) where {Ti<:Integer}
    penalty_ratio > 0 || error("penalty_ratio must be positive.")
    lambda = penalty === nothing ? penalty_ratio * data.lambda_reference : Float64(penalty)
    lambda > 0 || error("penalty must be positive.")
    workers > 0 || throw(ArgumentError("workers must be positive"))
    all(!iszero, data.A.nzval) || throw(ArgumentError(
        "the design matrix contains explicit zero coefficients; call dropzeros! first",
    ))

    started = time_ns()
    A = data.A
    b = data.b
    m, n = size(A)
    num_variables = 2 * n + 1
    num_rows = m + 2 * n + 2
    matrix_nnz = nnz(A) + 4 * n + 2
    maximum_dimension = max(num_variables, num_rows, matrix_nnz + 1)
    maximum_dimension <= typemax(Ti) || throw(ArgumentError(
        "canonical Lasso data needs indices up to $maximum_dimension; use a wider index_type",
    ))

    x_range = 1:n
    epigraph_range = (n + 1):(2 * n)
    r_index = num_variables
    soc_first = 2 * n + 1

    colptr = Vector{Ti}(undef, num_variables + 1)
    cursor = 1
    for column in 1:n
        colptr[column] = Ti(cursor)
        cursor += 2 + Int(A.colptr[column + 1]) - Int(A.colptr[column])
    end
    for column in 1:n
        colptr[n + column] = Ti(cursor)
        cursor += 2
    end
    colptr[r_index] = Ti(cursor)
    cursor += 2
    colptr[num_variables + 1] = Ti(cursor)
    cursor == matrix_nnz + 1 || error("internal Lasso CSC size mismatch")

    rowval = Vector{Ti}(undef, matrix_nnz)
    nzval = Vector{Float64}(undef, matrix_nnz)
    active_workers = min(Int(workers), Threads.nthreads(), max(n, 1))
    _fill_lasso_feature_columns!(
        rowval,
        nzval,
        colptr,
        A,
        m,
        n,
        active_workers,
    )

    for column in 1:n
        position = Int(colptr[n + column])
        rowval[position] = Ti(column)
        nzval[position] = 1.0
        rowval[position + 1] = Ti(n + column)
        nzval[position + 1] = 1.0
    end
    position = Int(colptr[r_index])
    inverse_sqrt_two = inv(sqrt(2.0))
    rowval[position] = Ti(soc_first)
    nzval[position] = inverse_sqrt_two
    rowval[position + 1] = Ti(soc_first + 1)
    nzval[position + 1] = -inverse_sqrt_two

    objective = zeros(num_variables)
    objective[epigraph_range] .= lambda
    objective[r_index] = 2.0
    lower = fill(-Inf, num_variables)
    lower[epigraph_range] .= 0.0
    upper = fill(Inf, num_variables)
    constants = zeros(num_rows)
    constants[soc_first] = inverse_sqrt_two
    constants[soc_first + 1] = inverse_sqrt_two
    for row in 1:m
        constants[soc_first + 1 + row] = -Float64(b[row])
    end
    blocks = LassoConicBlock[
        LassoConicBlock(MOI.Nonnegatives, 2 * n, 1),
        LassoConicBlock(MOI.SecondOrderCone, m + 2, soc_first),
    ]
    timings = LassoConicTimings(Float64(time_ns() - started) / 1.0e9)
    return LassoConicData(
        MOI.MIN_SENSE,
        0.0,
        objective,
        lower,
        upper,
        num_rows,
        num_variables,
        colptr,
        rowval,
        nzval,
        constants,
        blocks,
        nothing,
        timings,
        m,
        n,
        x_range,
        epigraph_range,
        r_index,
        lambda,
        :compact_epigraph_direct_soc,
    )
end

"""Update only the L1 penalty coefficients; all matrix constraints are reused."""
function set_penalty!(built::LassoSOCPModel, penalty::Real)
    lambda = Float64(penalty)
    lambda > 0 || error("penalty must be positive.")
    objective = 2.0 * built.r
    sizehint!(objective.terms, length(built.u) + 1)
    for variable in built.u
        add_to_expression!(objective, lambda, variable)
    end
    set_objective_function(built.model, objective)
    built.lambda = lambda
    return lambda
end

function set_penalty!(data::LassoConicData, penalty::Real)
    lambda = Float64(penalty)
    lambda > 0 || error("penalty must be positive.")
    fill!(view(data.objective_coefficients, data.epigraph_range), lambda)
    data.penalty = lambda
    return lambda
end

function set_penalty!(model::JuMP.Model, data::LassoConicData, penalty::Real)
    JuMP.num_variables(model) == data.num_variables || throw(DimensionMismatch(
        "model has $(JuMP.num_variables(model)) variables, expected $(data.num_variables)",
    ))
    lambda = set_penalty!(data, penalty)
    terms = Vector{MOI.ScalarAffineTerm{Float64}}(
        undef,
        length(data.epigraph_range) + 1,
    )
    for (position, column) in enumerate(data.epigraph_range)
        terms[position] = MOI.ScalarAffineTerm(
            lambda,
            MOI.VariableIndex(column),
        )
    end
    terms[end] = MOI.ScalarAffineTerm(2.0, MOI.VariableIndex(data.r_index))
    MOI.set(
        JuMP.backend(model),
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(terms, 0.0),
    )
    JuMP.set_objective_sense(model, MOI.MIN_SENSE)
    return lambda
end

"""Set `lambda = penalty_ratio * ||A'b||_inf` without rebuilding constraints."""
function set_penalty_ratio!(built::LassoSOCPModel, data::LassoData, penalty_ratio::Real)
    penalty_ratio > 0 || error("penalty_ratio must be positive.")
    return set_penalty!(built, penalty_ratio * data.lambda_reference)
end

function parse_cli(args::Vector{String})
    isempty(args) && return ("help", Dict{String, String}())
    command = args[1]
    options = Dict{String, String}()
    i = 2
    while i <= length(args)
        startswith(args[i], "--") || error("Unexpected argument $(args[i]).")
        i == length(args) && error("Option $(args[i]) requires a value.")
        options[args[i][3:end]] = args[i + 1]
        i += 2
    end
    return command, options
end

function option(options, name, default = nothing)
    return get(options, name, default)
end

function write_scan_metadata(path::AbstractString, spec::DatasetSpec, scan, ratios)
    mkpath(dirname(abspath(path)))
    output = Dict{String, Any}(
        "schema_version" => 1,
        "dataset" => spec.id,
        "source_file" => basename(spec.path),
        "rows" => scan.rows,
        "features" => scan.features,
        "nnz" => scan.nnz,
        "max_feature_seen" => scan.max_feature_seen,
        "lambda_reference" => scan.lambda_reference,
        "lambda_zero_threshold" => scan.lambda_zero_threshold,
        "penalty_ratios" => ratios,
        "penalties" => [ratio * scan.lambda_reference for ratio in ratios],
        "label_convention" => "positive label -> +1; nonpositive label -> -1",
        "objective" => "norm(A*x-b, 2)^2 + lambda*norm(x, 1)",
    )
    open(path, "w") do io
        TOML.print(io, output; sorted = true)
    end
    return path
end

function print_help()
    println("""
Usage:
  julia --project=. realistic_lasso.jl scan --dataset DATASET \\
      [--max-rows N] [--penalty-ratios 1,0.1,0.01] [--output FILE]

  julia --project=. realistic_lasso.jl check-model --dataset DATASET \\
      --max-rows N [--penalty-ratio 1.0]

Dataset IDs are defined in datasets.toml. The default reproduction set in
run_penalty_sweep.jl is news20, E2006-log1p, and rcv1-train.

`scan` streams the compressed LIBSVM file and computes ||A'b||_inf; it does
not create CBF or any matrix copy. `check-model` is intended for a bounded row
subset because constructing a JuMP model for a full dataset needs a high-memory
machine. For actual PDCS experiments, use `run_penalty_sweep.jl`, whose default
bulk path calls `build_lasso_conic_data` and reuses one finalized CSC cache.
""")
end

function main(args = ARGS)
    command, options = parse_cli(collect(args))
    command in ("help", "--help", "-h") && return print_help()
    id = option(options, "dataset")
    id === nothing && error("Missing --dataset.")
    spec = dataset_spec(id)
    max_rows_text = option(options, "max-rows")
    max_rows = max_rows_text === nothing ? nothing : parse(Int64, max_rows_text)
    ratios = parse.(Float64, split(option(options, "penalty-ratios", "1,0.1,0.01"), ','))

    if command == "scan"
        scan = scan_libsvm(
            spec;
            max_rows,
            keep_column_counts = false,
            keep_labels = false,
        )
        println("dataset=$(spec.id) rows=$(scan.rows) features=$(scan.features) nnz=$(scan.nnz)")
        println("lambda_reference=$(scan.lambda_reference)")
        println("lambda_zero_threshold=$(scan.lambda_zero_threshold)")
        println("penalty_ratios=$(join(ratios, ','))")
        println("penalties=$(join(ratios .* scan.lambda_reference, ','))")
        output = option(options, "output")
        output === nothing || println("wrote $(write_scan_metadata(output, spec, scan, ratios))")
    elseif command == "check-model"
        max_rows === nothing && error("check-model requires --max-rows to avoid accidental full-model construction.")
        data = load_libsvm(spec; max_rows, compact_zero_columns = true)
        ratio = parse(Float64, option(options, "penalty-ratio", "1.0"))
        built = build_lasso_socp(data; penalty_ratio = ratio)
        println("built $(spec.id) subset: m=$(size(data.A, 1)) n=$(size(data.A, 2)) " *
                "nnz=$(nnz(data.A)) lambda=$(built.lambda)")
    else
        error("Unknown command '$command'. Run with --help.")
    end
    return nothing
end

export DatasetSpec,
       LassoData,
       LassoConicBlock,
       LassoConicData,
       LassoConicTimings,
       LassoSOCPModel,
       build_lasso_conic_data,
       build_lasso_socp,
       dataset_spec,
       dataset_specs,
       load_libsvm,
       penalties,
       scan_libsvm,
       set_penalty!,
       set_penalty_ratio!,
       validate_download

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch exception
        showerror(stderr, exception)
        println(stderr)
        exit(1)
    end
end

end # module
