module GenerateSOCCases

using Random
include("common.jl")
using .RebuttalCommon

export ordinary_soc, rescaled_soc, permute_cones

function ordinary_soc(count, dimension; sigma=2.0, seed=BASE_SEED)
    count > 0 && dimension >= 2 || error("invalid SOC shape")
    rng = MersenneTwister(seed)
    sigma .* randn(rng, Float64, count * dimension)
end

function rescaled_soc(count, dimension; sigma_x=1.0, sigma_d=1.0,
                      boundary_ratio=0.2, seed=BASE_SEED, classes=nothing)
    dimension >= 3 || error("rescaled SOC dimension must be at least 3")
    rng_x = MersenneTwister(seed + dimension)
    rng_d = MersenneTwister(seed + dimension + 1)
    x = zeros(Float64, count * dimension)
    d = ones(Float64, count * dimension)
    labels = classes === nothing ? fill(:positive_root, count) : collect(classes)
    length(labels) == count || error("class count mismatch")
    for cone in 1:count
        r = (cone-1)*dimension+1:cone*dimension
        tail = @view x[first(r)+1:last(r)]
        diagonal = @view d[first(r)+1:last(r)]
        tail .= sigma_x .* randn(rng_x, dimension-1)
        diagonal .= clamp.(abs.(sigma_d .* randn(rng_d, dimension-1)), 1e-3, 1e3)
        ns = sqrt(sum(abs2, diagonal .* tail))
        ni = sqrt(sum(abs2, tail ./ diagonal))
        x[first(r)] = labels[cone] === :feasible ? 1.2ns :
                      labels[cone] === :polar ? -1.2ni :
                      labels[cone] === :negative_root ? -boundary_ratio*ni :
                      boundary_ratio*ns
    end
    (; x, diagonal=d, classes=labels)
end

function permute_cones(values, dimension, p)
    verify_permutation(p, length(p))
    output = similar(values)
    for (dst, src) in pairs(p)
        output[(dst-1)*dimension+1:dst*dimension] .=
            values[(src-1)*dimension+1:src*dimension]
    end
    output
end

end
