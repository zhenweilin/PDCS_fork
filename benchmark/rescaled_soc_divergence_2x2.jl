#!/usr/bin/env julia

# Paired 2×2 cold-start benchmark for diagonally rescaled SOC projection.
# Publication timing uses only production kernels. Diagnostic PTX is loaded
# solely while constructing/validating the iteration-stratified workload.

using CUDA
using PDCS: PDCS_GPU, PDCS_CPU
using NVTX
using Dates
using Printf
using Random
using SHA
using Serialization
using Statistics

include("rebuttal/common.jl")
include("rebuttal/soc_divergence_cases.jl")
include("rebuttal/diagnostic_root_profile.jl")
using .RebuttalCommon
using .SOCDivergenceCases
using .DiagnosticRootProfile

opt(name, default) = option(name, default)
const EXPERIMENT = Symbol(opt("experiment", "iteration"))
const MODE = Symbol(opt("mode", flag("profile-one") ? "profile-one" :
                              parse(Float64, opt("duration", "0")) > 0 ? "duration" : "timing"))
const LAYOUT_OPT = Symbol(opt("layout", "all"))
const STRATEGY_OPT = Symbol(opt("strategy", "all"))
const COUNT = parse(Int, opt("cone-count", "1048576"))
const DIMENSION = parse(Int, opt("cone-dimension", "10"))
const SEED = parse(Int, opt("seed", "2026"))
const PILOT_SEED = parse(Int, opt("pilot-seed", "2001"))
const CANDIDATE_FACTOR = parse(Int, opt("candidate-factor", "4"))
const FAMILY_OPT = Symbol(opt("family", "auto"))
const GRID_OPT = Symbol(opt("grid", "auto"))
const WARMUPS = parse(Int, opt("warmups", "5"))
const ROUNDS = parse(Int, opt("rounds", "10"))
const DURATION = parse(Float64, opt("duration", "0"))
const DELTA = parse(Float64, opt("delta", "0.001"))
const OUTPUT_DIR = abspath(opt("output-dir",
    "benchmark/results/rebuttal/soc_divergence_2x2/$(Dates.format(now(UTC), "yyyymmddTHHMMSS"))"))
const CASE_CACHE = opt("case-cache", "")
const DRY_RUN = flag("dry-run")
const ABS_TOL = 1e-12
const REL_TOL = 1e-12

EXPERIMENT in (:iteration, :branch, :parametric) ||
    error("--experiment must be iteration, branch, or parametric")
MODE in (:timing, Symbol("profile-one"), :duration, :generate) ||
    error("--mode must be timing, profile-one, duration, or generate")
LAYOUT_OPT in (:all, :grouped, :interleaved) || error("invalid --layout")
STRATEGY_OPT in (:all, :threadWise, :warpWise) || error("invalid --strategy")
COUNT % 128 == 0 || error("--cone-count must be divisible by 128")
DIMENSION >= 3 || error("--cone-dimension must be at least 3")
CANDIDATE_FACTOR >= 1 || error("--candidate-factor must be positive")

function write_rows(path, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function branch_groups(case)
    [findall(==(Int8(c)), case.construction_class) for c in 1:4]
end

function gpu_buffers(case)
    input = vec(case.x)
    diagonal = vec(case.diagonal)
    total = length(input)
    count = length(case.ids)
    starts = Int64.(0:DIMENSION:total-DIMENSION)
    sizes = fill(Int64(DIMENSION), count)
    types = fill(Int64(22), count)
    (; x=CuArray(input), immutable=CuArray(input),
       bl=CUDA.zeros(Float64, total), bu=CUDA.zeros(Float64, total),
       diagonal=CuArray(diagonal), diagonal2=CuArray(diagonal.^2),
       scaled_x=CUDA.zeros(Float64, total), temp=CUDA.zeros(Float64, total),
       warm=CUDA.zeros(Float64, count), starts=CuArray(starts),
       sizes=CuArray(sizes), types=CuArray(types), count=Int64(count))
end

args(b) = (b.x, b.bl, b.bu, b.diagonal, b.diagonal2, b.scaled_x, b.temp,
           b.warm, b.starts, b.sizes, b.count, b.types)

function restore!(b)
    copyto!(b.x, b.immutable)
    fill!(b.warm, 0.0)
    fill!(b.scaled_x, 0.0)
    fill!(b.temp, 0.0)
end

function launch!(strategy, b)
    strategy === :threadWise &&
        return PDCS_GPU.threadWise_block_proj(args(b)..., ABS_TOL, REL_TOL)
    strategy === :warpWise &&
        return PDCS_GPU.warpWise_block_proj(args(b)..., ABS_TOL, REL_TOL)
    error("unsupported strategy")
end

function event_time!(strategy, b)
    restore!(b)
    start, stop = CUDA.Event(), CUDA.Event()
    CUDA.record(start)
    launch!(strategy, b)
    CUDA.record(stop)
    CUDA.synchronize(stop)
    1000CUDA.elapsed(start, stop)
end

function diagnostic_records(case, strategy)
    b = gpu_buffers(case)
    restore!(b)
    rec = profile_project!(strategy, args(b)...; abs_tol=ABS_TOL, rel_tol=REL_TOL)
    output = Array(b.x)
    restore!(b)
    launch!(strategy, b)
    CUDA.synchronize()
    production = Array(b.x)
    err = maximum(abs.(output .- production); init=0.0)
    err <= 5e-8 || error("diagnostic/production output mismatch: $err")
    rec=[RootProfileRecord(r.branch_code,r.interval_expansion_iterations,
      r.newton_attempts,r.newton_accepts,r.bisection_iterations,
      r.oracle_evaluations,r.warm_start_attempted,r.warm_start_accepted,
      r.max_iter_reached,r.termination_reason,
      Int32(all(isfinite,@view output[(i-1)*DIMENSION+1:i*DIMENSION])),
      r.reserved,r.final_residual,r.final_bracket_left,r.final_bracket_right,
      r.normalized_bracket_width) for (i,r) in pairs(rec)]
    rec, err
end

function freeze_family()
    FAMILY_OPT !== :auto && return FAMILY_OPT, GRID_OPT === :expanded
    # Pilot selection is based only on diagnostic work counts, never runtime.
    for (family, expanded) in ((:positive, false), (:positive, true),
                               (:negative, false), (:negative, true))
        pilot_count = CANDIDATE_FACTOR * COUNT
        candidate = generate_candidates(pilot_count, DIMENSION, PILOT_SEED;
                                        family, expanded)
        records, _ = diagnostic_records(candidate, :threadWise)
        expected = family === :positive ? 2 : 3
        work = [r.branch_code == expected && r.max_iter_reached == 0 &&
                r.output_finite==1 && isfinite(r.final_residual) &&
                r.termination_reason in (1,2) &&
                (abs(r.final_residual)<=ABS_TOL ||
                 r.normalized_bracket_width<=REL_TOL) ?
                (family === :positive ? r.bisection_iterations : r.oracle_evaluations) : -1
                for r in records]
        try
            select_quartiles(candidate, work, COUNT; seed=PILOT_SEED, family)
            GC.gc(true); CUDA.reclaim()
            return family, expanded
        catch e
            @warn "pilot family/grid failed" family expanded exception=e
            GC.gc(true); CUDA.reclaim()
        end
    end
    error("no predeclared candidate family/grid passed the pilot separation gate")
end

function iteration_case()
    family, expanded = freeze_family()
    candidate = generate_candidates(CANDIDATE_FACTOR * COUNT, DIMENSION, SEED;
                                    family, expanded)
    thread_records, instrumentation_error = diagnostic_records(candidate, :threadWise)
    expected = family === :positive ? 2 : 3
    work = [r.branch_code == expected && r.max_iter_reached == 0 &&
            r.output_finite==1 && isfinite(r.final_residual) &&
            r.termination_reason in (1,2) &&
            (abs(r.final_residual)<=ABS_TOL ||
             r.normalized_bracket_width<=REL_TOL) ?
            (family === :positive ? r.bisection_iterations : r.oracle_evaluations) : -1
            for r in thread_records]
    groups = select_quartiles(candidate, work, COUNT; seed=SEED, family)
    grouped, interleaved, gp, ip, gi, ii = paired_layouts(candidate, groups)
    warp_records,warp_instrumentation_error=diagnostic_records(grouped,:warpWise)
    all(r->r.branch_code==expected,warp_records) ||
      error("warpWise diagnostic path differs from selected candidate family")
    all(r->r.max_iter_reached==0,warp_records) ||
      error("warpWise diagnostic reached MAX_ITER")
    all(r->isfinite(r.final_residual),warp_records) ||
      error("warpWise diagnostic produced nonfinite final residual")
    all(r->r.output_finite==1,warp_records) ||
      error("warpWise diagnostic produced nonfinite projection output")
    all(r->r.termination_reason in (1,2) &&
           (abs(r.final_residual)<=ABS_TOL ||
            r.normalized_bracket_width<=REL_TOL),warp_records) ||
      error("warpWise diagnostic failed the common production stopping rule")
    selected_work = Dict(candidate.ids[j] => work[j] for group in groups for j in group)
    metadata = (; family, expanded, instrumentation_error, selected_work,
                warp_instrumentation_error,records=thread_records,
                candidate, groups, gp, ip, gi, ii)
    grouped, interleaved, metadata
end

function branch_case()
    case = generate_branch_case(COUNT, DIMENSION, SEED)
    thread_records,thread_error=diagnostic_records(case,:threadWise)
    warp_records,warp_error=diagnostic_records(case,:warpWise)
    expected=Int32[c-1 for c in case.construction_class]
    all(thread_records[i].branch_code==expected[i] for i in eachindex(expected)) ||
      error("threadWise diagnostic branch differs from construction label")
    all(warp_records[i].branch_code==expected[i] for i in eachindex(expected)) ||
      error("warpWise diagnostic branch differs from construction label")
    all(r->r.max_iter_reached==0,thread_records) ||
      error("threadWise diagnostic reached MAX_ITER")
    all(r->r.max_iter_reached==0,warp_records) ||
      error("warpWise diagnostic reached MAX_ITER")
    groups = branch_groups(case)
    for (c, group) in pairs(groups)
        shuffle!(MersenneTwister(SEED + 10_000c), group)
    end
    grouped, interleaved, gp, ip, gi, ii = paired_layouts(case, groups)
    grouped, interleaved, (; family=:branches, expanded=false,
                           gp, ip, gi, ii, candidate=case, groups,
                           thread_error,warp_error)
end

function parametric_case()
    case = generate_parametric_case(COUNT, DIMENSION, SEED, DELTA)
    base=(; family=:parametric, expanded=false,
           gp=collect(1:COUNT), ip=collect(1:COUNT),
           gi=collect(1:COUNT), ii=collect(1:COUNT), candidate=case)
    if MODE === :generate
        thread_records,thread_error=diagnostic_records(case,:threadWise)
        warp_records,warp_error=diagnostic_records(case,:warpWise)
        all(r->r.branch_code==2 && r.max_iter_reached==0,thread_records) ||
          error("parametric threadWise diagnostic path/termination failed")
        all(r->r.branch_code==2 && r.max_iter_reached==0,warp_records) ||
          error("parametric warpWise diagnostic path/termination failed")
        return case,case,merge(base,(;thread_records,warp_records,thread_error,warp_error))
    end
    case,case,base
end

function build_case()
    if !isempty(CASE_CACHE)
        isfile(CASE_CACHE) || error("case cache does not exist: $CASE_CACHE")
        return deserialize(CASE_CACHE)
    end
    EXPERIMENT === :iteration ? iteration_case() :
    EXPERIMENT === :branch ? branch_case() : parametric_case()
end

function save_manifest(grouped, interleaved, meta)
    mkpath(OUTPUT_DIR)
    write_rows(joinpath(OUTPUT_DIR, "case_manifest.csv"),
      ("experiment","seed","pilot_seed","cone_count","cone_dimension",
       "candidate_factor","family","expanded_grid","abs_tol","rel_tol",
       "grouped_hash","interleaved_hash","grouped_multiset_hash",
       "interleaved_multiset_hash"),
      [(EXPERIMENT,SEED,PILOT_SEED,COUNT,DIMENSION,CANDIDATE_FACTOR,
        meta.family,meta.expanded,ABS_TOL,REL_TOL,
        permutation_hash(grouped),permutation_hash(interleaved),
        multiset_hash(grouped),multiset_hash(interleaved))])
    write_rows(joinpath(OUTPUT_DIR, "permutations.csv"),
      ("position","grouped_source","interleaved_source",
       "grouped_inverse","interleaved_inverse"),
      [(i,meta.gp[i],meta.ip[i],meta.gi[i],meta.ii[i]) for i in 1:COUNT])
    if hasproperty(meta, :selected_work)
        checks=Any[]
        medians=Dict{Symbol,Tuple{Float64,Float64}}()
        root_path=joinpath(OUTPUT_DIR,"root_work_raw.csv")
        open(root_path,"w") do io
          println(io,"experiment,seed,layout,position,cone_id,primary_work,expansion_iterations,bisection_iterations,oracle_evaluations,newton_attempts,newton_accepts,termination_reason,max_iter_reached,output_finite,final_residual,final_bracket_left,final_bracket_right,normalized_bracket_width,warp,warp_spread,modeled_active_lane_efficiency")
          for (layout,case) in ((:grouped,grouped),(:interleaved,interleaved))
            h=[meta.selected_work[id] for id in case.ids]
            spreads=Float64[]; efficiencies=Float64[]
            for first in 1:32:COUNT
              values=@view h[first:first+31]
              hi=maximum(values)
              hi>0 || error("zero-work warp in $layout")
              push!(spreads,hi-minimum(values))
              push!(efficiencies,sum(values)/(32hi))
            end
            medians[layout]=(median(spreads),median(efficiencies))
            for i in 1:COUNT
              r=meta.records[case.ids[i]]
              println(io,join((EXPERIMENT,SEED,layout,i,case.ids[i],h[i],
                r.interval_expansion_iterations,r.bisection_iterations,
                r.oracle_evaluations,r.newton_attempts,r.newton_accepts,
                r.termination_reason,r.max_iter_reached,r.output_finite,
                r.final_residual,
                r.final_bracket_left,r.final_bracket_right,
                r.normalized_bracket_width,cld(i,32),spreads[cld(i,32)],
                efficiencies[cld(i,32)]),','))
            end
          end
        end
        sg,eg=medians[:grouped]; si,ei=medians[:interleaved]
        pass=si>sg && eg>=0.90 && ei<=eg-0.10
        write_rows(joinpath(OUTPUT_DIR,"manipulation_checks.csv"),
          ("experiment","seed","grouped_median_spread","interleaved_median_spread",
           "grouped_median_efficiency","interleaved_median_efficiency","status"),
          [(EXPERIMENT,SEED,sg,si,eg,ei,pass ? "PASS" : "FAIL")])
        summary_rows=Any[]
        for (quartile,group) in pairs(meta.groups)
            for (metric,getter) in (("expansion",r->r.interval_expansion_iterations),
                                    ("bisection",r->r.bisection_iterations),
                                    ("oracle",r->r.oracle_evaluations))
                values=sort([getter(meta.records[j]) for j in group])
                p90=values[clamp(ceil(Int,0.90length(values)),1,length(values))]
                push!(summary_rows,(EXPERIMENT,SEED,meta.family,quartile,metric,
                                    median(values),p90,maximum(values),length(values)))
            end
        end
        write_rows(joinpath(OUTPUT_DIR,"root_work_summary.csv"),
          ("experiment","seed","family","quartile","metric","median","p90","maximum","cones"),
          summary_rows)
        pass || error("Section 3.9 manipulation gate failed")
    elseif EXPERIMENT === :branch
        grouped_codes=[length(unique(grouped.construction_class[i:i+31])) for i in 1:32:COUNT]
        interleaved_codes=[length(unique(interleaved.construction_class[i:i+31])) for i in 1:32:COUNT]
        pass=all(==(1),grouped_codes) && all(==(4),interleaved_codes)
        write_rows(joinpath(OUTPUT_DIR,"manipulation_checks.csv"),
          ("experiment","seed","grouped_distinct_codes_min","grouped_distinct_codes_max",
           "interleaved_distinct_codes_min","interleaved_distinct_codes_max","status"),
          [(EXPERIMENT,SEED,minimum(grouped_codes),maximum(grouped_codes),
            minimum(interleaved_codes),maximum(interleaved_codes),pass ? "PASS" : "FAIL")])
        pass || error("branch manipulation gate failed")
    elseif EXPERIMENT === :parametric && hasproperty(meta,:thread_records)
        rows=Any[]
        for (strategy,records,error) in ((:threadWise,meta.thread_records,meta.thread_error),
                                        (:warpWise,meta.warp_records,meta.warp_error))
            b=[r.bisection_iterations for r in records]
            spreads=[maximum(b[first:first+31])-minimum(b[first:first+31])
                     for first in 1:32:length(b)]
            sort!(b)
            p90=b[clamp(ceil(Int,0.90length(b)),1,length(b))]
            push!(rows,(EXPERIMENT,SEED,DELTA,strategy,median(b),p90,maximum(b),
                        median(spreads),error))
        end
        write_rows(joinpath(OUTPUT_DIR,"root_work_summary.csv"),
          ("experiment","seed","delta","strategy","bisection_median","bisection_p90",
           "bisection_max","within_warp_spread_median","instrumentation_error"),rows)
    end
end

const WILLIAMS = (
    ((:threadWise,:grouped),(:threadWise,:interleaved),
     (:warpWise,:interleaved),(:warpWise,:grouped)),
    ((:threadWise,:interleaved),(:warpWise,:grouped),
     (:threadWise,:grouped),(:warpWise,:interleaved)),
    ((:warpWise,:grouped),(:warpWise,:interleaved),
     (:threadWise,:interleaved),(:threadWise,:grouped)),
    ((:warpWise,:interleaved),(:threadWise,:grouped),
     (:warpWise,:grouped),(:threadWise,:interleaved)),
)

function sampled_cpu_error(case, output; samples=min(1024,length(case.ids)))
    indices=unique(round.(Int,range(1,length(case.ids);length=samples)))
    worst=0.0
    for i in indices
        expected=copy(@view case.x[:,i])
        tail=@view expected[2:end]
        d=copy(@view case.diagonal[2:end,i])
        d2=d.^2
        product=zeros(Float64,length(tail))
        temp=zeros(Float64,length(tail))
        warm=zeros(Float64,1)
        PDCS_CPU.soc_proj_diagonal!(expected,tail,d,d2,product,temp,warm,1,
                                    PDCS_CPU.timesInfo())
        actual=@view output[(i-1)*DIMENSION+1:i*DIMENSION]
        for j in eachindex(expected)
            scaled=abs(actual[j]-expected[j]) /
              (5e-8 + 5e-8max(abs(actual[j]),abs(expected[j])))
            worst=max(worst,scaled)
        end
    end
    worst
end

function correctness!(buffers, grouped, interleaved)
    for ((layout, strategy), b) in buffers
        restore!(b); launch!(strategy, b); CUDA.synchronize()
    end
    tg = Array(buffers[(:grouped,:threadWise)].x)
    wg = Array(buffers[(:grouped,:warpWise)].x)
    ti = Array(buffers[(:interleaved,:threadWise)].x)
    wi = Array(buffers[(:interleaved,:warpWise)].x)
    mixed_error(a,b)=maximum(abs.(a.-b) ./
      (5e-8 .+ 5e-8 .* max.(abs.(a),abs.(b)));init=0.0)
    errors = (mixed_error(tg,wg),mixed_error(ti,wi))
    maxerr = maximum(errors)
    all(isfinite,tg) && all(isfinite,wg) && all(isfinite,ti) && all(isfinite,wi) ||
      error("nonfinite projection output")
    maxerr <= 1 || error("threadWise/warpWise mixed tolerance failed: $maxerr")
    cpu_grouped=sampled_cpu_error(grouped,tg)
    cpu_interleaved=sampled_cpu_error(interleaved,ti)
    max(cpu_grouped,cpu_interleaved)<=1 ||
      error("sampled CPU/GPU mixed tolerance failed: grouped=$cpu_grouped interleaved=$cpu_interleaved")
    write_rows(joinpath(OUTPUT_DIR, "correctness.csv"),
      ("experiment","seed","grouped_strategy_error","interleaved_strategy_error",
       "grouped_cpu_scaled_error","interleaved_cpu_scaled_error","sampled_cones","status"),
      [(EXPERIMENT,SEED,errors[1],errors[2],cpu_grouped,cpu_interleaved,
        min(1024,COUNT),"PASS")])
end

function timing!(grouped, interleaved)
    buffers = Dict(
      (layout,strategy) => gpu_buffers(layout === :grouped ? grouped : interleaved)
      for layout in (:grouped,:interleaved), strategy in (:threadWise,:warpWise))
    for (cell,b) in buffers, _ in 1:WARMUPS
        restore!(b); launch!(cell[2],b)
    end
    CUDA.synchronize()
    correctness!(buffers,grouped,interleaved)
    rows = Any[]
    seed_index = SEED - 2026
    for round in 0:ROUNDS-1
        order_id = mod(seed_index + round, 4) + 1
        for (position,(strategy,layout)) in pairs(WILLIAMS[order_id])
            ms = event_time!(strategy, buffers[(layout,strategy)])
            push!(rows,(EXPERIMENT,SEED,round+1,order_id-1,position,
                        strategy,layout,ms,COUNT,DIMENSION))
        end
    end
    path = joinpath(OUTPUT_DIR, "timings_raw.csv")
    write_rows(path,("experiment","seed","round","order","position","strategy",
                     "layout","runtime_ms","cone_count","cone_dimension"),rows)
    summaries = Any[]
    for strategy in (:threadWise,:warpWise), layout in (:grouped,:interleaved)
        values = [r[8] for r in rows if r[6]===strategy && r[7]===layout]
        push!(summaries,(EXPERIMENT,SEED,strategy,layout,length(values),
                         mean(values),median(values),std(values)))
    end
    write_rows(joinpath(OUTPUT_DIR,"timings_seed_summary.csv"),
      ("experiment","seed","strategy","layout","replicates","mean_ms","median_ms","std_ms"),
      summaries)
end

function one_or_duration!(grouped, interleaved)
    layout = LAYOUT_OPT === :all ? :interleaved : LAYOUT_OPT
    strategy = STRATEGY_OPT === :all ? :threadWise : STRATEGY_OPT
    case = layout === :grouped ? grouped : interleaved
    b = gpu_buffers(case)
    for _ in 1:WARMUPS
        restore!(b); launch!(strategy,b)
    end
    CUDA.synchronize()
    if MODE === Symbol("profile-one")
        restore!(b)
        NVTX.range_push("PDCS_PROJECTION")
        try
            launch!(strategy,b)
            CUDA.synchronize()
        finally
            NVTX.range_pop()
        end
        println("PROFILE_COMPLETE experiment=$EXPERIMENT layout=$layout strategy=$strategy launches=1")
    else
        launches=0; restore_ns=0; projection_ms=0.0; started=time()
        while time()-started < DURATION
            t=time_ns(); restore!(b); CUDA.synchronize(); restore_ns += time_ns()-t
            projection_ms += event_time_without_restore!(strategy,b)
            launches += 1
        end
        @printf("UTILIZATION_COMPLETE experiment=%s layout=%s strategy=%s launches=%d elapsed_seconds=%.6f restore_seconds=%.6f projection_seconds=%.6f\n",
          EXPERIMENT,layout,strategy,launches,time()-started,restore_ns/1e9,projection_ms/1e3)
    end
end

function event_time_without_restore!(strategy,b)
    start,stop=CUDA.Event(),CUDA.Event()
    CUDA.record(start); launch!(strategy,b); CUDA.record(stop); CUDA.synchronize(stop)
    1000CUDA.elapsed(start,stop)
end

if DRY_RUN
    println("DRY_RUN experiment=$EXPERIMENT mode=$MODE seed=$SEED count=$COUNT dimension=$DIMENSION layout=$LAYOUT_OPT strategy=$STRATEGY_OPT output=$OUTPUT_DIR")
    exit()
end
CUDA.functional() || error("CUDA is not functional")
grouped, interleaved, metadata = build_case()
save_manifest(grouped, interleaved, metadata)
if MODE === :generate
    cache=joinpath(OUTPUT_DIR,"case_cache.jls")
    cache_meta=(; family=metadata.family,expanded=metadata.expanded,
                 gp=metadata.gp,ip=metadata.ip,gi=metadata.gi,ii=metadata.ii)
    serialize(cache,(grouped,interleaved,cache_meta))
    println("CASE_CACHE=$cache")
    println("GENERATE_COMPLETE output=$OUTPUT_DIR")
elseif MODE === :timing
    timing!(grouped, interleaved)
    println("TIMING_COMPLETE output=$OUTPUT_DIR")
else
    one_or_duration!(grouped, interleaved)
end
