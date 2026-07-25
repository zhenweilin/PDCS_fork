module RebuttalCommon

using Dates
using Random
using Statistics

export option, flag, int_list, float_list, symbol_list, csv, write_csv,
       BASE_SEED, STRATEGIES, strategy_code, paired_seed, environment_rows,
       permutation, inverse_permutation, verify_permutation

const BASE_SEED = 2026
const STRATEGIES = (:gridWise, :blockWise, :warpWise, :threadWise)

function option(name, default=nothing; args=ARGS)
    prefix = "--$name="
    for (i, value) in pairs(args)
        startswith(value, prefix) && return value[length(prefix)+1:end]
        value == "--$name" && i < length(args) && return args[i+1]
    end
    default
end
flag(name; args=ARGS) = any(==("--$name"), args)
int_list(x) = parse.(Int, filter(!isempty, split(x, ',')))
float_list(x) = parse.(Float64, filter(!isempty, split(x, ',')))
symbol_list(x) = Symbol.(filter(!isempty, split(x, ',')))
csv(x) = '"' * replace(string(x), '"' => "\"\"") * '"'
paired_seed(cell, trial; base=BASE_SEED) = base + 10_000cell + trial

strategy_code(strategy) =
    strategy === :gridWise ? 1 : strategy === :blockWise ? 2 :
    strategy === :warpWise ? 3 : strategy === :threadWise ? 4 :
    error("unknown strategy $strategy")

function write_csv(path, header, rows)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(csv.(row), ','))
        end
    end
end

function permutation(classes, layout; seed=BASE_SEED)
    n = length(classes)
    if layout == :grouped
        return sortperm(eachindex(classes); by=i -> (classes[i], i))
    elseif layout == :random
        return randperm(MersenneTwister(seed), n)
    elseif layout == :interleaved
        groups = Dict(c => findall(==(c), classes) for c in sort(unique(classes)))
        output = Int[]
        while length(output) < n
            for c in sort(collect(keys(groups)))
                isempty(groups[c]) || push!(output, popfirst!(groups[c]))
            end
        end
        return output
    end
    error("layout must be grouped, random, or interleaved")
end

function inverse_permutation(p)
    inv = similar(p)
    for (destination, source) in pairs(p)
        inv[source] = destination
    end
    inv
end

function verify_permutation(p, n)
    length(p) == n && sort(p) == collect(1:n) || error("invalid permutation")
    inverse_permutation(p)[p] == collect(1:n) || error("inverse permutation failed")
    true
end

function environment_rows()
    [
        ("timestamp_utc", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")),
        ("julia_version", string(VERSION)),
        ("git_commit", get(ENV, "PDCS_GIT_COMMIT", "unknown")),
        ("cuda_toolkit", get(ENV, "PDCS_CUDA_TOOLKIT", "unknown")),
    ]
end

end
