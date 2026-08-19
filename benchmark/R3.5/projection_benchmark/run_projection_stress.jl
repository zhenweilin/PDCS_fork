using CUDA
using Dates
using LinearAlgebra
using PDCS
using Printf
using Random
using SHA
using Statistics
using TOML

include(joinpath(@__DIR__, "case_matrix.jl"))

const GPU = PDCS.PDCS_GPU
const CSV_COLUMNS = [
    "timestamp_utc", "variant", "artifact_sha256", "device", "shard_index",
    "shard_count", "case_index", "case_id", "category", "strategy",
    "input_key", "natural_strategy", "total_blocks", "structured_cones", "coordinates",
    "max_dimension", "scaling", "condition", "branch", "warm", "amplitude",
    "heterogeneous", "plan_compacted", "plan_native", "plan_serial",
    "plan_thread_soc", "plan_warp_soc", "plan_simple", "samples",
    "latency_median_us", "latency_p10_us", "latency_p90_us",
    "coordinates_per_second", "cones_per_second", "output_finite",
    "reference_finite", "reference_normalized_cone_violation",
    "reference_valid", "max_abs_reference_error", "relative_reference_error",
    "idempotence_error", "max_normalized_cone_violation",
    "nonexpansive_ratio", "correct", "profile_events", "profile_reductions",
    "profile_oracles", "profile_expansions", "profile_bisections",
    "profile_newton_attempts", "profile_newton_accepts",
    "profile_warm_attempts", "profile_warm_accepts", "profile_max_iter",
    "profile_nonfinite", "validation_scope", "checksum", "status", "error",
]

function option(name::String, default::String)
    prefix = "--$(name)="
    for arg in ARGS
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return default
end

parse_bool(value) = lowercase(String(value)) in ("1", "true", "yes", "on")

"""Deterministically keep every same-input hierarchy comparison on one GPU."""
function stable_input_shard(input_key, shard_count)
    digest = sha256(String(input_key))
    value = zero(UInt64)
    for byte in @view digest[1:8]
        value = (value << 8) | UInt64(byte)
    end
    return Int(mod(value, UInt64(shard_count)))
end

function csv_cell(value)
    value === missing && return ""
    if value isa AbstractFloat
        return isfinite(value) ? @sprintf("%.17g", value) : string(value)
    end
    text = string(value)
    occursin(r"[\",\n\r]", text) || return text
    return "\"$(replace(text, "\"" => "\"\""))\""
end

function ensure_csv(path)
    mkpath(dirname(path))
    if !isfile(path) || filesize(path) == 0
        open(path, "w") do io
            println(io, join(CSV_COLUMNS, ','))
        end
    end
    return path
end

function append_row(path, row::Dict{String,Any})
    open(path, "a") do io
        println(io, join((csv_cell(get(row, key, missing)) for key in CSV_COLUMNS), ','))
        flush(io)
    end
    return row
end

function family_code(family, scaling)
    family === :free && return Int64(0)
    family === :zero && return Int64(2)
    family === :nonnegative && return Int64(3)
    family === :soc && return Int64(scaling === :diagonal ? 22 :
                                    scaling === :scalar ? 21 : 20)
    family === :exp && return Int64(scaling === :diagonal ? 27 : 26)
    family === :dual_exp && return Int64(scaling === :diagonal ? 29 : 28)
    error("unsupported projection family $family")
end

function expand_layout(case)
    families = Symbol[]
    dimensions = Int64[]
    for (family, dimension, count) in case.segments
        append!(families, fill(Symbol(family), Int(count)))
        append!(dimensions, fill(Int64(dimension), Int(count)))
    end
    starts = Vector{Int64}(undef, length(dimensions))
    cursor = Int64(0)
    for i in eachindex(dimensions)
        starts[i] = cursor
        cursor += dimensions[i]
    end
    codes = Int64[family_code(family, case.scaling) for family in families]
    return (; families, dimensions, starts, codes, coordinates=Int(cursor))
end

function diagonal_values(rng, dimension, condition, scaling)
    scaling === :identity && return ones(dimension)
    if scaling === :scalar
        value = 10.0 ^ rand(rng, -1:1)
        return fill(value, dimension)
    end
    dimension == 1 && return ones(1)
    half_log = 0.5 * log10(condition)
    values = 10.0 .^ collect(range(-half_log, half_log; length=dimension))
    shuffle!(rng, values)
    return values
end

function soc_input!(x, range, d, branch, amplitude, rng, cone_index, scaling)
    head = first(range)
    tail = head + 1:last(range)
    x[tail] .= randn(rng, length(tail))
    if scaling === :diagonal
        d[head] = 1.0
        weighted = norm(@view(d[tail]) .* @view(x[tail]))
        inverse = norm(@view(x[tail]) ./ @view(d[tail]))
    else
        # Identity/scalar cone scaling uses the closed-form Euclidean SOC
        # projection; a uniform scalar cancels from the cone membership test.
        weighted = norm(@view(x[tail]))
        inverse = weighted
    end
    if branch === :inside
        x[head] = 1.25 * weighted
    elseif branch === :polar
        x[head] = -1.25 * inverse
    elseif branch === :boundary_inside
        x[head] = (1.0 + 1e-11) * weighted
    elseif branch === :boundary_outside
        x[head] = (1.0 - 1e-11) * weighted
    elseif branch === :root_increasing
        x[head] = -0.15 * inverse
    else
        x[head] = 0.15 * weighted
    end
    x[range] .*= amplitude * (1.0 + 0.01 * ((cone_index - 1) % 7))
    return
end

function exp_template(family, branch)
    if family === :exp
        branch === :inside && return [0.0, 1.0, 2.0]
        branch === :polar && return [1.0, 0.0, -1.0]
        branch === :boundary && return [0.0, 0.0, 1.0]
        branch === :root_negative && return [1.0, -0.2, -0.5]
        branch === :near_degenerate && return [1e-12, 1e-14, 1e-16]
        return [1.0, 1.0, 0.05]
    end
    branch === :inside && return [-1.0, 0.0, 1.0]
    branch === :polar && return [0.0, -1.0, -1.0]
    branch === :boundary && return [0.0, 1.0, 1.0]
    branch === :root_negative && return [-0.05, 1.0, 0.01]
    branch === :near_degenerate && return [-1e-14, 1e-12, 1e-16]
    return [1.0, 1.0, 0.05]
end

function make_inputs(case, layout, rng)
    n = layout.coordinates
    x1 = zeros(Float64, n)
    d = ones(Float64, n)
    for i in eachindex(layout.families)
        family = layout.families[i]
        start = Int(layout.starts[i]) + 1
        dimension = Int(layout.dimensions[i])
        range = start:start + dimension - 1
        local_d = diagonal_values(rng, dimension, case.condition, case.scaling)
        d[range] .= local_d
        if family === :free
            x1[range] .= randn(rng, dimension)
        elseif family === :zero
            x1[range] .= case.amplitude .* randn(rng, dimension)
        elseif family === :nonnegative
            x1[range] .= case.amplitude .* randn(rng, dimension)
        elseif family === :soc
            soc_input!(x1, range, d, case.branch, case.amplitude, rng, i,
                       case.scaling)
        else
            template = exp_template(family, case.branch)
            x1[range] .= case.amplitude .* template
            x1[range] .+= abs(case.amplitude) * 1e-4 .* randn(rng, dimension)
        end
    end
    perturbation = 1e-4 .* max.(abs.(x1), abs(case.amplitude)) .* randn(rng, n)
    x2 = x1 .+ perturbation
    return x1, x2, d
end

function buffers(x, d, layout)
    n = length(x)
    return (
        vec = CuArray(x),
        bl = CUDA.fill(-Inf, n),
        bu = CUDA.fill(Inf, n),
        d = CuArray(d),
        d2 = CuArray(d .* d),
        dx = CUDA.zeros(Float64, n),
        temp = CUDA.zeros(Float64, n),
        warm = CUDA.fill(1.0, length(layout.dimensions)),
        heads_gpu = CuArray(layout.starts),
        ns_gpu = CuArray(layout.dimensions),
        codes_gpu = CuArray(layout.codes),
    )
end

function call_projection!(strategy, b, layout, abs_tol, rel_tol)
    block_count = Int64(length(layout.dimensions))
    if strategy === :gridWise
        GPU.gridWise_block_proj(
            b.vec, b.bl, b.bu, b.d, b.d2, b.dx, b.temp, b.warm,
            layout.starts, b.ns_gpu, layout.dimensions, block_count,
            layout.codes, abs_tol, rel_tol,
        )
    elseif strategy === :blockWise
        GPU.blockWise_block_proj(
            b.vec, b.bl, b.bu, b.d, b.d2, b.dx, b.temp, b.warm,
            b.heads_gpu, b.ns_gpu, block_count, b.codes_gpu, abs_tol, rel_tol,
        )
    elseif strategy === :warpWise
        GPU.warpWise_block_proj(
            b.vec, b.bl, b.bu, b.d, b.d2, b.dx, b.temp, b.warm,
            b.heads_gpu, b.ns_gpu, block_count, b.codes_gpu, abs_tol, rel_tol,
        )
    elseif strategy === :threadWise
        GPU.threadWise_block_proj(
            b.vec, b.bl, b.bu, b.d, b.d2, b.dx, b.temp, b.warm,
            b.heads_gpu, b.ns_gpu, block_count, b.codes_gpu, abs_tol, rel_tol,
        )
    else
        error("unknown strategy $strategy")
    end
    return
end

function warm_seed(case, b, layout, source1, abs_tol, rel_tol, heterogeneous)
    fill!(b.warm, 1.0)
    if case.warm in (:good, :perturbed)
        copyto!(b.vec, source1)
        GPU._heterogeneous_projection_enabled[] = false
        call_projection!(:threadWise, b, layout, abs_tol, rel_tol)
        CUDA.synchronize()
        seed = Array(b.warm)
        if case.warm === :perturbed
            seed .+= 1e-3 .* max.(abs.(seed), 1.0)
        end
        return seed
    elseif case.warm === :bad
        return [isodd(i) ? 1e100 : -1e100 for i in eachindex(layout.dimensions)]
    end
    return ones(Float64, length(layout.dimensions))
end

function cone_violation(result, d, layout, scaling)
    worst = 0.0
    for i in eachindex(layout.families)
        family = layout.families[i]
        start = Int(layout.starts[i]) + 1
        dimension = Int(layout.dimensions[i])
        range = start:start + dimension - 1
        values = @view result[range]
        violation = 0.0
        scale = 1.0 + maximum(abs, values)
        if family === :zero
            violation = maximum(abs, values; init=0.0)
        elseif family === :nonnegative
            violation = max(0.0, -minimum(values; init=0.0))
        elseif family === :soc
            tail = start + 1:last(range)
            tail_norm = scaling === :diagonal ?
                norm(@view(d[tail]) .* @view(result[tail])) :
                norm(@view(result[tail]))
            violation = max(0.0,
                tail_norm - result[start])
            scale += tail_norm
        elseif family === :exp
            transformed = case_scaling_view(values, @view(d[range]), :primal)
            x, y, z = transformed
            tol = 1e-13 * scale
            if abs(y) <= tol
                violation = max(y, x, -z, 0.0)
            elseif y > 0.0 && z > 0.0
                violation = max(0.0, x / y - log(z / y))
            else
                violation = max(0.0, -y, -z)
            end
        elseif family === :dual_exp
            transformed = case_scaling_view(values, @view(d[range]), :dual)
            u, v, w = transformed
            tol = 1e-13 * scale
            if abs(u) <= tol
                violation = max(-v, -w, 0.0)
            elseif u < 0.0 && w > 0.0
                violation = max(0.0, log(-u) + v / u - 1.0 - log(w))
            else
                violation = max(0.0, u, -w)
            end
        end
        worst = max(worst, violation / scale)
    end
    return worst
end

@inline function case_scaling_view(values, d, polarity)
    polarity === :primal && return values ./ d
    return values .* d
end

function plan_fields(b, layout, heterogeneous)
    if !heterogeneous
        return (false, 0, 0, 0, 0, 0)
    end
    plan = GPU._get_heterogeneous_projection_plan(
        b.ns_gpu, Int64(length(layout.dimensions)), b.codes_gpu,
    )
    return (plan.fully_compacted, Int(plan.native_cone_count),
            Int(plan.serial_cone_count), Int(plan.compact_soc_cone_count),
            Int(plan.compact_warp_soc_cone_count), Int(plan.simple_cone_count))
end

function profile_projection(case, b, layout, source, warm, abs_tol, rel_tol)
    case.strategy === :gridWise && return nothing
    copyto!(b.vec, source)
    copyto!(b.warm, warm)
    GPU.enable_projection_work_profile!(scope=:all)
    summary = nothing
    try
        call_projection!(case.strategy, b, layout, abs_tol, rel_tol)
        CUDA.synchronize()
    finally
        summary = GPU.disable_projection_work_profile!()
    end
    rows = summary.by_cone
    total(field) = sum(getproperty(row, field) for row in rows; init=0)
    return (
        events=total(:projection_events),
        reductions=total(:vector_vector_reductions),
        oracles=total(:oracle_evaluations),
        expansions=total(:interval_expansion_iterations),
        bisections=total(:bisection_iterations),
        newton_attempts=total(:newton_attempts),
        newton_accepts=total(:newton_accepts),
        warm_attempts=total(:warm_start_attempts),
        warm_accepts=total(:warm_start_accepts),
        max_iter=total(:max_iter_reached),
        nonfinite=total(:nonfinite_outputs),
    )
end

function quantile_linear(values, probability)
    isempty(values) && return NaN
    sorted = sort(values)
    index = 1 + probability * (length(sorted) - 1)
    lower = floor(Int, index)
    upper = ceil(Int, index)
    lower == upper && return sorted[lower]
    fraction = index - lower
    return muladd(fraction, sorted[upper] - sorted[lower], sorted[lower])
end

function run_case(case, case_index; variant, device, shard_index, shard_count,
                  heterogeneous, samples, profile, abs_tol, rel_tol, seed,
                  artifact_hash)
    key_seed = sum(i * Int(byte) for (i, byte) in enumerate(codeunits(case.input_key)))
    rng = MersenneTwister(seed + key_seed)
    layout = expand_layout(case)
    root_cones = count(family -> family in PB_FAMILIES, layout.families)
    pathological_real_grid = case.category === :represent_layout &&
        case.strategy === :gridWise && root_cones > 10_000 &&
        maximum(layout.dimensions; init=0) > 10_000
    if pathological_real_grid
        return Dict{String,Any}(
            "timestamp_utc" => Dates.format(
                now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
            "variant" => variant, "artifact_sha256" => artifact_hash,
            "device" => device, "shard_index" => shard_index,
            "shard_count" => shard_count, "case_index" => case_index,
            "case_id" => case.id, "category" => case.category,
            "strategy" => case.strategy, "input_key" => case.input_key,
            "total_blocks" => length(layout.dimensions),
            "structured_cones" => root_cones,
            "coordinates" => layout.coordinates,
            "max_dimension" => maximum(layout.dimensions; init=0),
            "scaling" => case.scaling, "condition" => case.condition,
            "branch" => case.branch, "warm" => case.warm,
            "amplitude" => case.amplitude, "heterogeneous" => heterogeneous,
            "samples" => 0, "correct" => false,
            "validation_scope" => "selector-rejected pathological real grid",
            "status" => "NOT_APPLICABLE",
            "error" => "host-per-cone grid mapping is excluded when both " *
                "structured_cones>10000 and max_dimension>10000",
        )
    end
    x1, x2, d = make_inputs(case, layout, rng)
    source1 = CuArray(x1)
    source2 = CuArray(x2)
    b = buffers(x1, d, layout)
    initial_warm = warm_seed(case, b, layout, source1, abs_tol, rel_tol,
                             heterogeneous)
    natural = GPU.select_projection_strategy(layout.dimensions, layout.codes)
    compacted, native, serial, tiny_soc, warp_soc, simple =
        plan_fields(b, layout, heterogeneous)

    # A hierarchy-independent correctness reference: the general thread-wise
    # kernel processes every listed block (including simple cones that may
    # occur beyond the leading two), with heterogeneous remapping disabled.
    # This avoids treating the dispatch being tested as its own oracle.
    GPU._heterogeneous_projection_enabled[] = false
    copyto!(b.vec, source1)
    copyto!(b.warm, initial_warm)
    call_projection!(:threadWise, b, layout, abs_tol, rel_tol)
    CUDA.synchronize()
    reference = Array(b.vec)

    # A forced grid mapping is intentionally pathological for solver-shaped
    # layouts containing thousands of cones: the grid implementation invokes
    # the host/cuBLAS projection once per cone. Keep the independent-reference
    # and idempotence gates, but do not repeat this deliberately losing mapping
    # for auxiliary perturbation/profile checks. Synthetic grid cases retain
    # the complete protocol below.
    slow_represent_grid = case.category === :represent_layout &&
                          case.strategy === :gridWise

    GPU._heterogeneous_projection_enabled[] = heterogeneous
    copyto!(b.vec, source1)
    copyto!(b.warm, initial_warm)
    primary_elapsed = if slow_represent_grid
        CUDA.@elapsed call_projection!(
            case.strategy, b, layout, abs_tol, rel_tol,
        )
    else
        call_projection!(case.strategy, b, layout, abs_tol, rel_tol)
        CUDA.synchronize()
        NaN
    end
    output1 = Array(b.vec)
    warm_after = Array(b.warm)

    # Idempotence, plus a nearby input for the nonexpansiveness check.
    copyto!(b.vec, output1)
    copyto!(b.warm, warm_after)
    call_projection!(case.strategy, b, layout, abs_tol, rel_tol)
    CUDA.synchronize()
    output_twice = Array(b.vec)
    output2 = if slow_represent_grid
        copy(output1)
    else
        copyto!(b.vec, source2)
        copyto!(b.warm, initial_warm)
        call_projection!(case.strategy, b, layout, abs_tol, rel_tol)
        CUDA.synchronize()
        Array(b.vec)
    end

    # The grid implementation intentionally handles one cone at a time through
    # the host/cuBLAS path.  For solver-shaped layouts with tens of thousands
    # of cones, repeating a deliberately losing grid mapping 17 times can take
    # tens of minutes.  The dedicated hierarchy A/B supplies stable timing for
    # those layouts; the first correctness call is also their only timing
    # sample, avoiding extra invocations of a mapping the selector rejects.
    timed_samples = slow_represent_grid ? 1 : samples
    timings = if slow_represent_grid
        [primary_elapsed]
    else
        for warmup in 1:3
            copyto!(b.vec, isodd(warmup) ? source1 : source2)
            copyto!(b.warm, initial_warm)
            call_projection!(case.strategy, b, layout, abs_tol, rel_tol)
        end
        CUDA.synchronize()
        values = Float64[]
        for sample in 1:timed_samples
            copyto!(b.vec, isodd(sample) ? source1 : source2)
            copyto!(b.warm, initial_warm)
            CUDA.synchronize()
            elapsed = CUDA.@elapsed call_projection!(
                case.strategy, b, layout, abs_tol, rel_tol,
            )
            push!(values, elapsed)
        end
        CUDA.synchronize()
        values
    end

    work = profile && !slow_represent_grid ?
        profile_projection(case, b, layout, source1, initial_warm,
                           abs_tol, rel_tol) : nothing
    scale = max(1.0, maximum(abs, reference; init=0.0))
    max_abs_error = maximum(abs, output1 .- reference; init=0.0)
    relative_error = norm(output1 .- reference) / max(1.0, norm(reference))
    idempotence = maximum(abs, output_twice .- output1; init=0.0) / scale
    nonexpansive = if slow_represent_grid
        NaN
    else
        input_distance = norm((x1 .- x2) .* d)
        output_distance = norm((output1 .- output2) .* d)
        output_distance / max(input_distance, eps(Float64))
    end
    violation = cone_violation(output1, d, layout, case.scaling)
    finite = all(isfinite, output1) && all(isfinite, output2)
    reference_finite = all(isfinite, reference)
    reference_violation = reference_finite ?
        cone_violation(reference, d, layout, case.scaling) : Inf
    reference_valid = reference_finite && reference_violation <= 5e-7
    agreement = !reference_valid || max_abs_error / scale <= 5e-7
    # Nonexpansiveness is reported but is not a hard gate here: primal EXP,
    # dual EXP and diagonal SOC use different transformed metrics internally.
    # A single D-weighted ratio is useful for spotting outliers, but cannot be
    # compared to one uniformly without reconstructing each solver metric.
    correct = finite && agreement &&
              idempotence <= 5e-7 && violation <= 5e-7
    median_seconds = median(timings)
    row = Dict{String,Any}(
        "timestamp_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
        "variant" => variant, "artifact_sha256" => artifact_hash,
        "device" => device, "shard_index" => shard_index,
        "shard_count" => shard_count, "case_index" => case_index,
        "case_id" => case.id, "category" => case.category,
        "strategy" => case.strategy, "natural_strategy" => natural,
        "input_key" => case.input_key,
        "total_blocks" => length(layout.dimensions),
        "structured_cones" => root_cones, "coordinates" => layout.coordinates,
        "max_dimension" => maximum(layout.dimensions; init=0),
        "scaling" => case.scaling, "condition" => case.condition,
        "branch" => case.branch, "warm" => case.warm,
        "amplitude" => case.amplitude, "heterogeneous" => heterogeneous,
        "plan_compacted" => compacted, "plan_native" => native,
        "plan_serial" => serial, "plan_thread_soc" => tiny_soc,
        "plan_warp_soc" => warp_soc, "plan_simple" => simple,
        "samples" => timed_samples,
        "latency_median_us" => 1e6 * median_seconds,
        "latency_p10_us" => 1e6 * quantile_linear(timings, 0.1),
        "latency_p90_us" => 1e6 * quantile_linear(timings, 0.9),
        "coordinates_per_second" => layout.coordinates / median_seconds,
        "cones_per_second" => root_cones / median_seconds,
        "output_finite" => finite, "reference_finite" => reference_finite,
        "reference_normalized_cone_violation" => reference_violation,
        "reference_valid" => reference_valid,
        "max_abs_reference_error" => max_abs_error,
        "relative_reference_error" => relative_error,
        "idempotence_error" => idempotence,
        "max_normalized_cone_violation" => violation,
        "nonexpansive_ratio" => nonexpansive, "correct" => correct,
        "validation_scope" => slow_represent_grid ?
            "reference+feasibility+idempotence" :
            "reference+feasibility+idempotence+perturbation+profile",
        "checksum" => sum(output1), "status" => "OK", "error" => "",
    )
    if work !== nothing
        for (column, field) in (
            ("profile_events", :events), ("profile_reductions", :reductions),
            ("profile_oracles", :oracles), ("profile_expansions", :expansions),
            ("profile_bisections", :bisections),
            ("profile_newton_attempts", :newton_attempts),
            ("profile_newton_accepts", :newton_accepts),
            ("profile_warm_attempts", :warm_attempts),
            ("profile_warm_accepts", :warm_accepts),
            ("profile_max_iter", :max_iter),
            ("profile_nonfinite", :nonfinite),
        )
            row[column] = getproperty(work, field)
        end
    end
    return row
end

function artifact_digest(directory)
    names = ("moderate_block_proj.ptx", "massive_block_proj.ptx",
             "sufficient_block_proj.ptx", "libfew_block_proj.so")
    entries = String[]
    for name in names
        path = joinpath(directory, name)
        isfile(path) || continue
        push!(entries, "$(name):$(bytes2hex(SHA.sha256(read(path))))")
    end
    return bytes2hex(SHA.sha256(join(entries, '\n')))
end

file_digest(path) = isfile(path) ? bytes2hex(SHA.sha256(read(path))) : "missing"

function main()
    CUDA.functional() || error("CUDA is not functional")
    device = parse(Int, option("device", "0"))
    shard_index = parse(Int, option("shard-index", "0"))
    shard_count = parse(Int, option("shard-count", "1"))
    0 <= shard_index < shard_count || error("invalid shard index/count")
    tier = Symbol(option("tier", "full"))
    variant = option("variant", "enhanced")
    heterogeneous = parse_bool(option("heterogeneous", "true"))
    samples = parse(Int, option("samples", tier === :full ? "11" : "5"))
    profile = parse_bool(option("profile", "true"))
    case_regex = option("case-regex", ".*")
    abs_tol = parse(Float64, option("abs-tol", "1e-12"))
    rel_tol = parse(Float64, option("rel-tol", "1e-12"))
    seed = parse(Int, option("seed", "20260815"))
    output = abspath(option("output", joinpath(
        @__DIR__, "results", "$(variant)_shard$(shard_index).csv")))
    CUDA.device!(device)
    repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    artifact_dir = abspath(get(
        ENV, "PDCS_CUDA_PROJECTION_ARTIFACT_DIR",
        joinpath(repo_root, "src", "pdcs_gpu", "cuda"),
    ))
    artifact_hash = artifact_digest(artifact_dir)
    all_cases = projection_stress_cases(; tier)
    selected = [(i, case) for (i, case) in enumerate(all_cases)
                if stable_input_shard(case.input_key, shard_count) == shard_index &&
                   occursin(Regex(case_regex), case.id)]
    ensure_csv(output)
    metadata = Dict(
        "timestamp_utc" => string(now(UTC)), "variant" => variant,
        "tier" => String(tier), "device" => device,
        "shard_index" => shard_index, "shard_count" => shard_count,
        "sharding" => "sha256(input_key)",
        "selected_cases" => length(selected), "total_cases" => length(all_cases),
        "case_regex" => case_regex,
        "heterogeneous" => heterogeneous, "profile" => profile,
        "samples" => samples, "abs_tol" => abs_tol, "rel_tol" => rel_tol,
        "slow_represent_grid_samples" => 1,
        "slow_represent_grid_validation" =>
            "reference+feasibility+idempotence; no perturbation/profile repeat",
        "seed" => seed, "artifact_dir" => artifact_dir,
        "artifact_sha256" => artifact_hash,
        "runner_sha256" => file_digest(@__FILE__),
        "case_matrix_sha256" => file_digest(joinpath(@__DIR__, "case_matrix.jl")),
        "projection_strategy_sha256" => file_digest(joinpath(
            repo_root, "src", "pdcs_gpu", "projection_strategy.jl")),
        "gpu_kernel_sha256" => file_digest(joinpath(
            repo_root, "src", "pdcs_gpu", "gpu_kernel.jl")),
        "projection_strategy_override" => get(
            ENV, "PDCS_PROJECTION_STRATEGY_OVERRIDE", "",
        ),
        "cuda_device" => string(CUDA.device()), "output" => output,
    )
    open(replace(output, r"\.csv$" => ".toml"), "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    @info "projection stress shard started" variant=variant tier=tier device=device shard_index=shard_index shard_count=shard_count selected=length(selected) total=length(all_cases)
    for (case_index, case) in selected
        started = time()
        try
            row = run_case(
                case, case_index; variant, device, shard_index, shard_count,
                heterogeneous, samples, profile, abs_tol, rel_tol, seed,
                artifact_hash,
            )
            append_row(output, row)
            @info "projection stress case complete" case=case.id status=get(row, "status", "OK") correct=get(row, "correct", false) median_us=get(row, "latency_median_us", missing) elapsed=time() - started
        catch err
            message = sprint(showerror, err, catch_backtrace())
            row = Dict{String,Any}(
                "timestamp_utc" => string(now(UTC)), "variant" => variant,
                "artifact_sha256" => artifact_hash, "device" => device,
                "shard_index" => shard_index, "shard_count" => shard_count,
                "case_index" => case_index, "case_id" => case.id,
                "category" => case.category, "strategy" => case.strategy,
                "input_key" => case.input_key,
                "status" => "ERROR", "correct" => false,
                "error" => message,
            )
            append_row(output, row)
            @error "projection stress case failed" case=case.id exception=(err, catch_backtrace())
        end
        GC.gc(false)
        CUDA.reclaim()
    end
    println(output)
    return output
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
