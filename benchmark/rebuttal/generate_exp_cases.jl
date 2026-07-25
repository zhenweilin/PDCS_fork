module GenerateExpCases

using Random
include("common.jl")
using .RebuttalCommon

export random_exp, permute_exp

function random_exp(count; sigma_x=1.0, sigma_d=1.0, seed=BASE_SEED)
    rng_x = MersenneTwister(seed + count + 27)
    rng_d = MersenneTwister(seed + count + 91_337)
    x = sigma_x .* randn(rng_x, Float64, 3count)
    d = clamp.(abs.(sigma_d .* randn(rng_d, Float64, 3count)), 1e-3, 1e3)
    (; x, diagonal=d)
end

function permute_exp(values, p)
    output = similar(values)
    for (dst, src) in pairs(p)
        output[3dst-2:3dst] .= values[3src-2:3src]
    end
    output
end

end
