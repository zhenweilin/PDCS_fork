"""Case construction for the PDCS projection stress benchmark.

The matrix intentionally contains no RSOC cases: the rebuttal data and the
requested scope exercise SOC, primal exponential, and dual exponential cones.
Every synthetic homogeneous layout is padded with the two leading simple
blocks used by the solver, so hierarchy selection and kernel indexing match a
real PDCS layout.
"""

const PB_STRATEGIES = (:gridWise, :blockWise, :warpWise, :threadWise)
const PB_FAMILIES = (:soc, :exp, :dual_exp)

function pb_case(id; category, strategy, segments, scaling=:diagonal,
                 condition=1e2, branch=:root, warm=:cold, amplitude=1.0,
                 input_key=id)
    return (; id=String(id), category=Symbol(category), strategy=Symbol(strategy),
            segments=collect(segments), scaling=Symbol(scaling),
            condition=Float64(condition), branch=Symbol(branch),
            warm=Symbol(warm), amplitude=Float64(amplitude),
            input_key=String(input_key))
end

function homogeneous_case(id, family, strategy, count, dimension;
                          category=:homogeneous, kwargs...)
    segments = [(:free, 1, 2), (family, dimension, count)]
    return pb_case(id; category, strategy, segments, kwargs...)
end

function append_unique!(cases, seen, case)
    case.id in seen && return cases
    push!(cases, case)
    push!(seen, case.id)
    return cases
end

function projection_stress_cases(; tier=:full)
    tier in (:smoke, :quick, :full) ||
        throw(ArgumentError("tier must be smoke, quick, or full"))
    cases = NamedTuple[]
    seen = Set{String}()
    add(case) = append_unique!(cases, seen, case)

    # 1. Root/closed-form branch coverage under every scaling mode and every
    # execution hierarchy.  This is deliberately a Cartesian coverage gate:
    # a branch passing in the block kernel is not used as evidence that its
    # grid/warp/thread implementation was exercised.  Counts are sized to
    # fill the corresponding hierarchy without making the branch gate the
    # dominant part of the full stress run.  SOC uses dimension 201, the
    # large-SOC size discussed in R3.5.
    family_branches = Dict(
        :soc => (:inside, :polar, :boundary_inside, :boundary_outside,
                 :root_decreasing, :root_increasing),
        :exp => (:inside, :polar, :boundary, :root_positive, :root_negative,
                 :near_degenerate),
        :dual_exp => (:inside, :polar, :boundary, :root_positive,
                      :root_negative, :near_degenerate),
    )
    branch_counts = Dict(:gridWise => 1, :blockWise => 256,
                         :warpWise => 1024, :threadWise => 4096)
    for family in PB_FAMILIES
        dimension = family === :soc ? 201 : 3
        for branch in family_branches[family]
            for scaling in (:identity, :scalar, :diagonal)
                for strategy in PB_STRATEGIES
                    id = "branch_$(family)_$(branch)_$(scaling)_$(strategy)"
                    add(homogeneous_case(
                        id, family, strategy, branch_counts[strategy], dimension;
                        category=:branch, scaling, condition=1e4,
                        branch, warm=:cold,
                    ))
                end
            end
        end
    end

    # 2. Warm-start behavior: cold, exact previous root, a nearby previous
    # root, and a deliberately unusable value.
    for family in PB_FAMILIES, warm in (:cold, :good, :perturbed, :bad)
        dimension = family === :soc ? 201 : 3
        branch = family === :soc ? :root_decreasing : :root_positive
        add(homogeneous_case("warm_$(family)_$(warm)", family, :blockWise,
                             512, dimension; category=:warm_start,
                             scaling=:diagonal, condition=1e6, branch, warm))
    end

    # 3. SOC dimension boundaries.  Each dimension is forced through every
    # hierarchy; counts are capped to keep the largest case below ~8M doubles.
    soc_dimensions = (2, 3, 7, 8, 9, 31, 32, 33, 149, 150, 200, 201,
                      1999, 2000, 2048, 8192, 22289, 105589)
    nominal_counts = Dict(:gridWise => 1, :blockWise => 512,
                          :warpWise => 4096, :threadWise => 65536)
    for dimension in soc_dimensions, strategy in PB_STRATEGIES
        count = max(1, min(nominal_counts[strategy], fld(8_000_000, dimension)))
        add(homogeneous_case("dimension_soc_d$(dimension)_$(strategy)",
                             :soc, strategy, count, dimension;
                             category=:dimension, scaling=:diagonal,
                             condition=1e4, branch=:root_decreasing,
                             warm=:perturbed))
    end

    # 4. Dispatch boundaries: EXP serial packing begins at 8 cones, tiny SOC
    # thread packing at 256, and medium SOC warp packing at 64.
    for count in (6, 7, 8, 9, 254, 255, 256, 257)
        add(homogeneous_case("dispatch_exp_n$(count)", :exp, :blockWise,
                             count, 3; category=:dispatch, scaling=:diagonal,
                             condition=1e4, branch=:root_positive))
        add(homogeneous_case("dispatch_dual_exp_n$(count)", :dual_exp,
                             :blockWise, count, 3; category=:dispatch,
                             scaling=:diagonal, condition=1e4,
                             branch=:root_negative))
    end
    for dimension in (3, 8), count in (254, 255, 256, 257)
        add(homogeneous_case("dispatch_soc_thread_d$(dimension)_n$(count)",
                             :soc, :blockWise, count, dimension;
                             category=:dispatch, scaling=:diagonal,
                             condition=1e4, branch=:root_decreasing))
    end
    for dimension in (9, 32), count in (62, 63, 64, 65)
        add(homogeneous_case("dispatch_soc_warp_d$(dimension)_n$(count)",
                             :soc, :blockWise, count, dimension;
                             category=:dispatch, scaling=:diagonal,
                             condition=1e4, branch=:root_decreasing))
    end
    for dimension in (8, 9, 32, 33), count in (63, 64, 255, 256)
        add(homogeneous_case("dispatch_cross_d$(dimension)_n$(count)",
                             :soc, :warpWise, count, dimension;
                             category=:dispatch_cross, scaling=:diagonal,
                             condition=1e2, branch=:root_increasing))
    end
    # Throughput sweeps distinguish a launch-bound threshold win from a rule
    # that remains profitable once enough cones are available to fill the GPU.
    for family in (:exp, :dual_exp), count in (512, 1024, 4096, 60000)
        branch = family === :exp ? :root_positive : :root_negative
        add(homogeneous_case("dispatch_throughput_$(family)_n$(count)",
                             family, :blockWise, count, 3;
                             category=:dispatch_cross, scaling=:diagonal,
                             condition=1e4, branch))
    end
    for dimension in (3, 4, 8), count in (512, 1024, 4096, 60000)
        add(homogeneous_case("dispatch_throughput_soc_d$(dimension)_n$(count)",
                             :soc, :blockWise, count, dimension;
                             category=:dispatch_cross, scaling=:diagonal,
                             condition=1e4, branch=:root_decreasing))
    end
    for dimension in (9, 32), count in (128, 512, 4096, 60000)
        add(homogeneous_case("dispatch_throughput_soc_d$(dimension)_n$(count)",
                             :soc, :blockWise, count, dimension;
                             category=:dispatch_cross, scaling=:diagonal,
                             condition=1e4, branch=:root_decreasing))
    end
    # Fine-grained boundaries for the selected family/dimension thresholds.
    for family in (:exp, :dual_exp), count in
        (255, 256, 257, 383, 384, 511, 512, 513, 767, 768, 1023, 1024)
        branch = family === :exp ? :root_positive : :root_negative
        add(homogeneous_case("dispatch_threshold_$(family)_n$(count)",
                             family, :blockWise, count, 3;
                             category=:dispatch_cross, scaling=:diagonal,
                             condition=1e4, branch))
    end
    for dimension in (3, 4, 5, 8), count in (255, 256, 257, 511, 512, 513)
        add(homogeneous_case("dispatch_threshold_soc_d$(dimension)_n$(count)",
                             :soc, :blockWise, count, dimension;
                             category=:dispatch_cross, scaling=:diagonal,
                             condition=1e4, branch=:root_decreasing))
    end
    for dimension in (5, 8, 9, 16, 32, 33), count in (63, 64, 65)
        add(homogeneous_case("dispatch_threshold_warp_d$(dimension)_n$(count)",
                             :soc, :blockWise, count, dimension;
                             category=:dispatch_cross, scaling=:diagonal,
                             condition=1e4, branch=:root_decreasing))
    end

    # 5. Automatic hierarchy-selection boundaries. Counts below are structured
    # counts; the two leading free blocks make total block counts exactly
    # 3/4, 999/1000/1001, and 59999/60000/60001.
    for total_count in (3, 4, 999, 1000, 1001, 59999, 60000, 60001)
        count = total_count - 2
        add(homogeneous_case("selection_count_$(total_count)", :soc,
                             :threadWise, count, 3;
                             category=:selection, scaling=:diagonal,
                             condition=1e2, branch=:root_decreasing))
    end
    for dimension in (149, 150, 151, 1999, 2000, 2001)
        add(homogeneous_case("selection_dimension_$(dimension)", :soc,
                             :threadWise, 1001, dimension;
                             category=:selection, scaling=:diagonal,
                             condition=1e2, branch=:root_decreasing))
    end

    # 6. Same-input hierarchy crossovers.  Unlike the capacity-oriented cases
    # above, every strategy for a workload shares an input_key and therefore
    # receives bit-identical vectors and diagonals. Obviously dominated grid
    # mappings are retained through 256 cones to measure their crossover, but
    # are not forced over tens of thousands of host-orchestrated calls.
    crossover_layouts = [
        (:soc, 3, 1), (:soc, 4, 1), (:soc, 8, 1), (:soc, 9, 1),
        (:soc, 16, 1), (:soc, 31, 1), (:soc, 32, 1), (:soc, 33, 1),
        (:soc, 64, 1), (:soc, 128, 1), (:soc, 201, 1),
        (:soc, 2048, 1), (:soc, 8192, 1), (:soc, 16384, 1),
        (:soc, 22289, 1), (:soc, 24576, 1), (:soc, 32768, 1),
        (:soc, 49152, 1), (:soc, 65536, 1), (:soc, 105589, 1),
        (:soc, 201, 8), (:soc, 32, 64),
        (:soc, 33, 64), (:soc, 8, 256), (:soc, 9, 256),
        (:soc, 201, 1000), (:soc, 32, 4096), (:soc, 33, 4096),
        (:soc, 3, 60000), (:soc, 8, 60000),
        (:soc, 256, 1001), (:soc, 512, 1001),
        (:soc, 1024, 1001), (:soc, 1536, 1001),
        (:soc, 1999, 1001), (:soc, 2000, 1001),
        (:exp, 3, 1), (:exp, 3, 8), (:exp, 3, 64), (:exp, 3, 256),
        (:exp, 3, 1000), (:exp, 3, 4096), (:exp, 3, 60000),
        (:dual_exp, 3, 1), (:dual_exp, 3, 8), (:dual_exp, 3, 64),
        (:dual_exp, 3, 256), (:dual_exp, 3, 1000),
        (:dual_exp, 3, 4096), (:dual_exp, 3, 60000),
    ]
    # Denser SOC selector surface near the one-thread/one-warp/one-block
    # crossovers.  This prevents fitting a threshold from only two dimensions
    # or counts, while keeping each layout below the benchmark memory cap.
    for (dimension, counts) in (
        (3, (64, 256, 1000, 4096)),
        (8, (8, 64, 1000, 4096)),
        (9, (1, 8, 64, 1000, 4096)),
        (32, (1, 8, 256, 1000)),
        (33, (1, 8, 256, 1000)),
        (64, (8, 64, 256, 1000, 4096)),
        (128, (8, 64, 256, 1000, 4096)),
        (201, (64, 256, 4096)),
        (512, (1, 64, 256, 1000, 4096)),
        (1024, (1, 64, 256, 1000, 4096)),
        (1536, (1, 64, 256, 1000, 4096)),
        (3, (8192, 16384, 32768)),
        (5, (8192, 16384, 32768)),
        (8, (8192, 16384, 32768)),
        (16, (8192, 16384, 32768)),
        (32, (8192, 16384, 32768)),
    )
        for count in counts
            item = (:soc, dimension, count)
            item in crossover_layouts || push!(crossover_layouts, item)
        end
    end
    for (family, dimension, count) in crossover_layouts
        key = "crossover_$(family)_d$(dimension)_n$(count)"
        strategies = (count <= 256 && dimension <= 201) || count <= 8 ? PB_STRATEGIES :
            count <= 4096 ? (:blockWise, :warpWise, :threadWise) :
            (:warpWise, :threadWise)
        branch = family === :soc ? :root_decreasing : :root_positive
        for strategy in strategies
            add(homogeneous_case("$(key)_$(strategy)", family, strategy,
                                 count, dimension; category=:crossover,
                                 scaling=:diagonal, condition=1e4, branch,
                                 warm=:perturbed, input_key=key))
        end
    end

    # 7. Conditioning and scale.  These expose overflow, underflow and the
    # shifted/log-coordinate regimes without changing the requested accuracy.
    for family in PB_FAMILIES, condition in (1.0, 1e2, 1e4, 1e8, 1e12)
        dimension = family === :soc ? 201 : 3
        branch = family === :soc ? :root_decreasing : :root_positive
        add(homogeneous_case("condition_$(family)_$(condition)", family,
                             :blockWise, 512, dimension;
                             category=:conditioning, scaling=:diagonal,
                             condition, branch, warm=:perturbed))
    end
    for family in PB_FAMILIES, amplitude in (1e-12, 1e-9, 1e-6, 1.0,
                                             1e6, 1e9, 1e12)
        dimension = family === :soc ? 201 : 3
        branch = family === :soc ? :root_increasing : :root_negative
        add(homogeneous_case("amplitude_$(family)_$(amplitude)", family,
                             :blockWise, 256, dimension;
                             category=:amplitude, scaling=:diagonal,
                             condition=1e6, branch, warm=:perturbed,
                             amplitude))
    end

    # 8. Exact hard-case inventories (order-normalized). They reproduce the
    # relevant amount and dimensions of work seen in represent_data, including
    # simple cones because dispatch profitability depends on the complete mix.
    hard_layouts = [
        ("represent_ravem", :blockWise,
         [(:zero,41,1),(:nonnegative,405,1),
          (:soc,6,1),(:exp,3,16)]),
        ("represent_gams01", :blockWise,
         [(:zero,241,1),(:nonnegative,1516,1),
          (:soc,3,110),(:exp,3,10)]),
        ("represent_batch", :blockWise,
         [(:zero,28,1),(:nonnegative,173,1),(:soc,6,1),(:exp,3,11)]),
        ("represent_batchs101006m", :blockWise,
         [(:zero,59,1),(:nonnegative,1543,1),(:soc,21,1),(:exp,3,29)]),
        ("represent_varun", :blockWise,
         [(:zero,334,1),(:nonnegative,671,1),
          (:soc,28,1),(:exp,3,327)]),
        ("represent_integrated", :blockWise,
         [(:zero,4489,1),(:nonnegative,40993,1),
          (:soc,1860,1),(:soc,14,326)]),
        ("represent_qssp180", :threadWise,
         [(:zero,64799,1),(:nonnegative,130682,1),(:soc,4,65341)]),
        ("represent_cx02_100", :warpWise,
         [(:zero,10098,1),(:nonnegative,10495,1),(:soc,5052,1),
          (:exp,3,5148)]),
        ("represent_db_plane_strain_prism", :threadWise,
         [(:zero,57536,1),(:nonnegative,44800,1),
          (:soc,22289,1),(:soc,3,22400)]),
        ("represent_joint_FC_12", :threadWise,
         [(:zero,189396,1),(:nonnegative,267028,1),
          (:soc,3,53112),(:soc,105589,1)]),
    ]
    for (id, strategy, segments) in hard_layouts
        # Keep the historical/natural row name for compatibility with the
        # smoke tier, then add the other three mappings with a shared
        # input_key.  This makes the solver-shaped inventories genuine
        # same-input hierarchy A/B cases instead of one-off throughput rows.
        for candidate in PB_STRATEGIES
            case_id = candidate === strategy ? id : "$(id)_$(candidate)"
            add(pb_case(case_id; category=:represent_layout,
                        strategy=candidate, segments,
                        scaling=:diagonal, condition=1e4,
                        branch=:root_decreasing, warm=:perturbed,
                        input_key=id))
        end
    end

    if tier === :smoke
        keep = Set(("branch_soc_root_decreasing_diagonal_blockWise",
                    "branch_exp_root_positive_diagonal_blockWise",
                    "branch_dual_exp_root_negative_diagonal_blockWise",
                    "dimension_soc_d201_gridWise",
                    "dimension_soc_d201_blockWise",
                    "dimension_soc_d33_warpWise",
                    "dimension_soc_d3_threadWise",
                    "dispatch_exp_n8", "dispatch_soc_thread_d3_n256",
                    "dispatch_soc_warp_d9_n64", "represent_varun",
                    "represent_qssp180"))
        return [case for case in cases if case.id in keep]
    elseif tier === :quick
        return [case for (i, case) in enumerate(cases)
                if case.category in (:dispatch, :dispatch_cross, :selection, :crossover,
                                     :represent_layout) ||
                   i % 7 == 1]
    end
    return cases
end
