module SOCDivergenceCases

using Random
using LinearAlgebra
using SHA
using Statistics

export ConeSet, initial_grid, expanded_grid, generate_candidates,
       generate_branch_case, generate_parametric_case, select_quartiles,
       paired_layouts, permutation_hash, multiset_hash, validate_pair

struct ConeSet
    x::Matrix{Float64}          # dimension × cones
    diagonal::Matrix{Float64}   # dimension × cones
    ids::Vector{Int64}
    boundary_ratio::Vector{Float64}
    log_sigma::Vector{Float64}
    construction_class::Vector{Int8}
end

initial_grid() = (
    [0.02, 0.10, 0.20, 0.50, 0.80, 0.95],
    [0.0, 0.25, 0.50, 1.00, 1.50, 2.00],
)
expanded_grid() = (
    [0.005, 0.01, 0.02, 0.10, 0.50, 0.90, 0.99],
    [0.0, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00],
)

# A separate deterministic stream for every cone and purpose makes generation
# independent of chunking and execution order.
function stream(seed::Integer, id::Integer, tag::Integer)
    # SplitMix64-style integer mixing; unlike Base.hash, this is stable across
    # Julia processes and does not depend on a session hash salt.
    z = UInt64(seed) ⊻ (UInt64(id) << 21) ⊻ (UInt64(tag) << 42) ⊻ 0x5044435352433133
    z ⊻= z >> 30; z *= 0xbf58476d1ce4e5b9
    z ⊻= z >> 27; z *= 0x94d049bb133111eb
    z ⊻= z >> 31
    MersenneTwister(UInt32(z & 0xffffffff))
end

function mix64(z::UInt64)
    z ⊻= z >> 30; z *= 0xbf58476d1ce4e5b9
    z ⊻= z >> 27; z *= 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

function normal_vector(seed::Integer, id::Integer, tag::Integer, n::Integer,
                       attempt::Integer=0)
    out=Vector{Float64}(undef,n)
    base=UInt64(seed) ⊻ (UInt64(id)<<21) ⊻ (UInt64(tag)<<42) ⊻
         (UInt64(attempt)<<48) ⊻ 0x5044435352433133
    j=1; counter=UInt64(0)
    while j<=n
        u1=(Float64((mix64(base+counter)>>11))+0.5)*0x1.0p-53
        u2=(Float64((mix64(base+counter+1)>>11))+0.5)*0x1.0p-53
        radius=sqrt(-2log(u1)); angle=2pi*u2
        out[j]=radius*cos(angle)
        j<n && (out[j+1]=radius*sin(angle))
        j+=2; counter+=2
    end
    out
end

function unit_direction(seed, id, q; tag=1)
    attempt=0
    while true
        z = normal_vector(seed,id,tag,q,attempt)
        n = norm(z)
        n > 0 && return z ./ n
        attempt += 1
    end
end

function centered_diagonal(seed, id, q, sigma; tag=2)
    attempt=0
    while true
        eta = normal_vector(seed,id,tag,q,attempt)
        ell = sigma .* (eta .- sum(eta) / q)
        d = exp.(ell)
        all(v -> 1e-3 <= v <= 1e3, d) && return d
        attempt += 1
    end
end

function generate_candidates(count::Integer, dimension::Integer, seed::Integer;
                             family::Symbol=:positive, expanded::Bool=false)
    dimension >= 3 || error("cone dimension must be at least three")
    family in (:positive, :negative) || error("family must be positive or negative")
    ratios, sigmas = expanded ? expanded_grid() : initial_grid()
    q = dimension - 1
    x = Matrix{Float64}(undef, dimension, count)
    diagonal = Matrix{Float64}(undef, dimension, count)
    ids = Int64.(1:count)
    ratio_col = Vector{Float64}(undef, count)
    sigma_col = Vector{Float64}(undef, count)
    classes = fill(Int8(family === :positive ? 3 : 4), count)
    cells = collect(Iterators.product(ratios, sigmas))
    for i in 1:count
        r, s = cells[mod1(i, length(cells))]
        u = unit_direction(seed, i, q)
        d = centered_diagonal(seed, i, q, s)
        a = norm(d .* u)
        b = norm(u ./ d)
        x[1, i] = family === :positive ? r * a : -r * b
        x[2:end, i] .= u
        diagonal[1, i] = 1.0
        diagonal[2:end, i] .= d
        ratio_col[i] = r
        sigma_col[i] = s
    end
    ConeSet(x, diagonal, ids, ratio_col, sigma_col, classes)
end

function generate_branch_case(count::Integer, dimension::Integer, seed::Integer)
    count % 4 == 0 || error("branch experiment requires cone count divisible by four")
    q = dimension - 1
    x = Matrix{Float64}(undef, dimension, count)
    diagonal = Matrix{Float64}(undef, dimension, count)
    classes = Vector{Int8}(undef, count)
    for i in 1:count
        u = unit_direction(seed, i, q; tag=11)
        d = centered_diagonal(seed, i, q, 1.0; tag=12)
        a, b = norm(d .* u), norm(u ./ d)
        c = Int8(mod1(i, 4))
        x[1, i] = c == 1 ? 1.25a : c == 2 ? -1.25b : c == 3 ? 0.20a : -0.20b
        x[2:end, i] .= u
        diagonal[1, i] = 1.0
        diagonal[2:end, i] .= d
        classes[i] = c
    end
    ConeSet(x, diagonal, Int64.(1:count), fill(NaN, count), fill(1.0, count), classes)
end

function generate_parametric_case(count::Integer, dimension::Integer, seed::Integer,
                                  delta::Real)
    delta in (0.0, 1e-4, 1e-3, 1e-2) ||
        error("delta must be one of 0, 1e-4, 1e-3, 1e-2")
    q = dimension - 1
    ustar = unit_direction(seed, 0, q; tag=21)
    dstar = centered_diagonal(seed, 0, q, 1.0; tag=22)
    ellstar = log.(dstar)
    x = Matrix{Float64}(undef, dimension, count)
    diagonal = Matrix{Float64}(undef, dimension, count)
    for i in 1:count
        xi = unit_direction(seed, i, q; tag=23)
        u = ustar .+ delta * norm(ustar) .* xi
        u ./= norm(u)
        d = nothing
        attempt=0
        while d === nothing
            ell = ellstar .+ delta .* normal_vector(seed,i,24,q,attempt)
            ell .-= sum(ell) / q
            candidate = exp.(ell)
            all(v -> 1e-3 <= v <= 1e3, candidate) && (d = candidate)
            attempt += 1
        end
        x[1, i] = 0.20norm(d .* u)
        x[2:end, i] .= u
        diagonal[1, i] = 1.0
        diagonal[2:end, i] .= d
    end
    ConeSet(x, diagonal, Int64.(1:count), fill(0.20, count),
            fill(Float64(delta), count), fill(Int8(3), count))
end

function select_quartiles(case::ConeSet, work::AbstractVector{<:Integer},
                          target::Integer; seed::Integer, family::Symbol)
    length(work) == length(case.ids) || error("work-count length mismatch")
    target % 128 == 0 || error("target cone count must be divisible by 128")
    retained = findall(>=(0), work)
    length(retained) >= target || error("fewer retained candidates than target")
    key(i) = family === :positive ? (work[i], case.ids[i]) :
             (work[i], case.ids[i])
    sorted = sort(retained; by=key)
    n = 4fld(length(sorted), 4)
    resize!(sorted, n)
    m = n ÷ 4
    nq = target ÷ 4
    selected = Vector{Vector{Int}}(undef, 4)
    for c in 1:4
        group = @view sorted[(c-1)*m+1:c*m]
        ranks = [floor(Int, (a + 0.5) * m / nq) + 1 for a in 0:nq-1]
        chosen = collect(group[ranks])
        shuffle!(stream(seed, c, 31), chosen)
        selected[c] = chosen
    end
    med1 = median(work[selected[1]])
    med4 = median(work[selected[4]])
    med4 - med1 >= 4 || error("quartile separation gate failed: Q4-Q1=$(med4-med1)")
    selected
end

function exact_orders(groups::Vector{Vector{Int}})
    length(groups) == 4 || error("exact layouts require four classes")
    n = length(groups[1])
    all(g -> length(g) == n, groups) || error("class sizes differ")
    n % 32 == 0 || error("each class size must be divisible by 32")
    total = 4n
    grouped = Vector{Int}(undef, total)
    interleaved = Vector{Int}(undef, total)
    for s in 0:(total ÷ 128 - 1), c in 0:3, lane in 0:31
        grouped[32(4s+c)+lane+1] = groups[c+1][32s+lane+1]
    end
    for w in 0:(total ÷ 32 - 1), r in 0:7, c in 0:3
        interleaved[32w+4r+c+1] = groups[c+1][8w+r+1]
    end
    grouped, interleaved
end

subset(case::ConeSet, order) = ConeSet(case.x[:, order], case.diagonal[:, order],
    case.ids[order], case.boundary_ratio[order], case.log_sigma[order],
    case.construction_class[order])

function paired_layouts(case::ConeSet, groups::Vector{Vector{Int}})
    g, i = exact_orders(groups)
    grouped, interleaved = subset(case, g), subset(case, i)
    validate_pair(grouped, interleaved)
    # The candidate pool may be larger than the selected workload, so g/i are
    # not permutations of 1:target. sortperm recovers the common canonical
    # selected-source order from either layout.
    grouped, interleaved, g, i, sortperm(g), sortperm(i)
end

function permutation_hash(case::ConeSet)
    bytes2hex(sha256(vcat(reinterpret(UInt8, vec(case.x)),
                         reinterpret(UInt8, vec(case.diagonal)),
                         reinterpret(UInt8, case.ids))))
end

function multiset_hash(case::ConeSet)
    p = sortperm(case.ids)
    permutation_hash(subset(case, p))
end

function validate_pair(grouped::ConeSet, interleaved::ConeSet)
    sort(grouped.ids) == sort(interleaved.ids) || error("layout ID multisets differ")
    multiset_hash(grouped) == multiset_hash(interleaved) ||
        error("layout cone-tuple hashes differ")
    true
end

end
