"""
    select_projection_strategy(block_sizes, projection_types) -> Symbol

Select the GPU hierarchy used to project a collection of cone blocks.  The
return value is one of `:gridWise`, `:blockWise`, `:warpWise`, or
`:threadWise`.

`block_sizes` and `projection_types` use the solver's complete block layout.
The first two blocks are scalar/simple cones in that layout, so only entries
from the third block onward determine the largest structured-cone dimension.

The heuristic is:

1. use `:gridWise` for at most three blocks, unless an RSOC kernel (codes 23--25)
   is present;
2. use `:blockWise` for at most 1,000 blocks or a structured cone of dimension
   at least 2,000;
3. use `:warpWise` for at most 60,000 blocks or a structured cone of
   dimension at least 150;
4. otherwise use `:threadWise`.

The thresholds are empirical and can be re-tuned with
`benchmark/random_soc_projection.jl` on the target GPU.
"""
function select_projection_strategy(
    block_sizes::AbstractVector{<:Integer},
    projection_types::AbstractVector{<:Integer},
)::Symbol
    length(block_sizes) == length(projection_types) ||
        throw(DimensionMismatch("block_sizes and projection_types must have equal length"))

    block_count = length(block_sizes)
    largest_structured_cone = block_count >= 3 ? maximum(@view block_sizes[3:end]) : 0
    has_rsoc = any(code -> code in (8, 9, 10, 23, 24, 25), projection_types)

    if block_count <= 3 && !has_rsoc
        return :gridWise
    elseif block_count <= 1_000 || largest_structured_cone >= 2_000
        return :blockWise
    elseif block_count <= 60_000 || largest_structured_cone >= 150
        return :warpWise
    else
        return :threadWise
    end
end
