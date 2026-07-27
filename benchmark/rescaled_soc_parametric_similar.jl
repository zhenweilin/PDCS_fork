#!/usr/bin/env julia

using CUDA
using PDCS: PDCS_GPU, PDCS_CPU
using Dates
using LinearAlgebra
using Printf
using Serialization
using SHA
using Statistics

include("rebuttal/common.jl")
include("rebuttal/soc_divergence_cases.jl")
include("rebuttal/diagnostic_root_profile.jl")
using .RebuttalCommon
using .SOCDivergenceCases
using .DiagnosticRootProfile

const MODE=Symbol(option("mode","generate"))
const COUNT=parse(Int,option("cone-count","1048576"))
const DIMENSION=parse(Int,option("cone-dimension","10"))
const SEED=parse(Int,option("seed","2026"))
const DELTAS=Tuple(float_list(option("deltas","0,0.0001,0.001,0.01")))
const DELTA=parse(Float64,option("delta","0"))
const STRATEGY=Symbol(option("strategy","threadWise"))
const WARMUPS=parse(Int,option("warmups","5"))
const ROUNDS=parse(Int,option("rounds","10"))
const DURATION=parse(Float64,option("duration","35"))
const OUTPUT_DIR=abspath(option("output-dir",
    "benchmark/results/rebuttal/soc_parametric_similarity/seed_$SEED"))
const CACHE=option("case-cache","")
const WAIT_FOR_START=flag("wait-for-start")
const SAVE_PER_CONE=flag("save-per-cone")
const DRY_RUN=flag("dry-run")
const ABS_TOL=1e-12
const REL_TOL=1e-12

MODE in (:pilot,:generate,:timing,:duration) || error("invalid --mode")
STRATEGY in (:threadWise,:warpWise) || error("invalid --strategy")
COUNT>0 && COUNT%32==0 || error("--cone-count must be positive and divisible by 32")
DIMENSION>=3 || error("--cone-dimension must be at least 3")
all(d->d in (0.0,1e-4,1e-3,1e-2),DELTAS) || error("invalid --deltas")

function write_rows(path,header,rows)
    mkpath(dirname(path))
    open(path,"w") do io
        println(io,join(header,','))
        for row in rows
            println(io,join(row,','))
        end
    end
end

hash_array(x)=bytes2hex(sha256(reinterpret(UInt8,vec(x))))
qvalue(v,p)=sort(v)[clamp(ceil(Int,p*length(v)),1,length(v))]

function gpu_buffers(case)
    input=vec(case.x); diagonal=vec(case.diagonal); total=length(input)
    starts=CuArray(Int64.(0:DIMENSION:total-DIMENSION))
    (;x=CuArray(input),immutable=CuArray(input),bl=CUDA.zeros(Float64,total),
      bu=CUDA.zeros(Float64,total),diagonal=CuArray(diagonal),
      diagonal2=CuArray(diagonal.^2),scaled=CUDA.zeros(Float64,total),
      temp=CUDA.zeros(Float64,total),warm=CUDA.zeros(Float64,COUNT),
      starts,sizes=CuArray(fill(Int64(DIMENSION),COUNT)),
      types=CuArray(fill(Int64(22),COUNT)),count=Int64(COUNT))
end
args(b)=(b.x,b.bl,b.bu,b.diagonal,b.diagonal2,b.scaled,b.temp,b.warm,
         b.starts,b.sizes,b.count,b.types)
function restore!(b)
    copyto!(b.x,b.immutable)
    fill!(b.warm,0); fill!(b.scaled,0); fill!(b.temp,0)
end
function launch!(strategy,b)
    strategy===:threadWise &&
      return PDCS_GPU.threadWise_block_proj(args(b)...,ABS_TOL,REL_TOL)
    PDCS_GPU.warpWise_block_proj(args(b)...,ABS_TOL,REL_TOL)
end
function event_time!(strategy,b;restore=true)
    restore && restore!(b)
    start,stop=CUDA.CuEvent(),CUDA.CuEvent()
    CUDA.record(start); launch!(strategy,b); CUDA.record(stop)
    CUDA.synchronize(stop)
    1000CUDA.elapsed(start,stop)
end

function sampled_cpu_error(case,output;samples=min(1024,COUNT))
    indices=unique(round.(Int,range(1,COUNT;length=samples)))
    worst_abs=0.0; worst_scaled=0.0
    for i in indices
        expected=copy(@view case.x[:,i])
        tail=@view expected[2:end]
        d=copy(@view case.diagonal[2:end,i])
        PDCS_CPU.soc_proj_diagonal!(expected,tail,d,d.^2,zeros(DIMENSION-1),
          zeros(DIMENSION-1),zeros(1),1,PDCS_CPU.timesInfo())
        actual=@view output[(i-1)*DIMENSION+1:i*DIMENSION]
        for j in eachindex(expected)
            e=abs(actual[j]-expected[j])
            worst_abs=max(worst_abs,e)
            worst_scaled=max(worst_scaled,e/(5e-8+5e-8max(abs(actual[j]),abs(expected[j]))))
        end
    end
    (;worst_abs,worst_scaled)
end

function family_from_cache()
    isempty(CACHE) && return generate_parametric_family(COUNT,DIMENSION,SEED;deltas=DELTAS)
    isfile(CACHE) || error("missing case cache: $CACHE")
    deserialize(CACHE)
end

function generator_artifacts(family)
    mkpath(joinpath(OUTPUT_DIR,"generator"))
    write_rows(joinpath(OUTPUT_DIR,"generator","input_hashes.csv"),
      ("seed","delta","x_sha256","diagonal_sha256"),
      ((SEED,d,hash_array(family.cases[d].x),hash_array(family.cases[d].diagonal))
       for d in DELTAS))
    rows=Any[]
    for d in DELTAS
        case=family.cases[d]
        angles=[acos(clamp(dot(family.ustar,@view(case.x[2:end,i])),-1,1))
                for i in 1:COUNT]
        rms=[sqrt(mean(abs2,log.(@view(case.diagonal[2:end,i])).-family.ellstar))
             for i in 1:COUNT]
        identical=d==0 ? maximum(abs.(case.x.-case.x[:,1])) : NaN
        push!(rows,(SEED,d,mean(angles),maximum(angles),mean(rms),maximum(rms),
                    identical,minimum(case.diagonal),maximum(case.diagonal)))
    end
    write_rows(joinpath(OUTPUT_DIR,"generator","seed_level_similarity.csv"),
      ("seed","delta","mean_angle","max_angle","mean_rms_log_diagonal",
       "max_rms_log_diagonal","delta_zero_max_vector_difference",
       "diagonal_min","diagonal_max"),rows)
    write_rows(joinpath(OUTPUT_DIR,"generator","rejection_counts.csv"),
      ("seed","delta","rejections"),
      ((SEED,d,family.rejection_counts[d]) for d in DELTAS))
    write_rows(joinpath(OUTPUT_DIR,"generator","base_hashes.csv"),
      ("seed","ustar_sha256","ellstar_sha256","direction_perturbations_sha256",
       "diagonal_perturbations_sha256"),
      [(SEED,hash_array(family.ustar),hash_array(family.ellstar),
        hash_array(family.direction_perturbations),
        hash_array(family.diagonal_perturbations))])
end

function diagnostic!(family)
    summary=Any[]; warp_rows=Any[]; term_rows=Any[]
    per_dir=joinpath(OUTPUT_DIR,"diagnostic","per_cone")
    SAVE_PER_CONE && mkpath(per_dir)
    for delta in DELTAS
        case=family.cases[delta]
        for strategy in (:threadWise,:warpWise)
            b=gpu_buffers(case); restore!(b)
            records=profile_project!(strategy,args(b)...;abs_tol=ABS_TOL,rel_tol=REL_TOL)
            diagnostic_output=Array(b.x)
            restore!(b); launch!(strategy,b); CUDA.synchronize()
            production_output=Array(b.x)
            invariant=maximum(abs.(diagnostic_output.-production_output);init=0.0)
            invariant<=5e-8 || error("diagnostic invariance failed: $invariant")
            all(r->r.branch_code==2,records) ||
              error("positive-root path gate failed seed=$SEED delta=$delta strategy=$strategy")
            all(r->r.max_iter_reached==0 && r.output_finite!=0,records) ||
              error("termination/nonfinite gate failed seed=$SEED delta=$delta")
            for (metric,getter) in
                (("expansion",r->Int(r.interval_expansion_iterations)),
                 ("bisection",r->Int(r.bisection_iterations)),
                 ("oracle",r->Int(r.oracle_evaluations)))
                values=getter.(records)
                mu=mean(values); sd=std(values)
                push!(summary,(SEED,delta,strategy,metric,minimum(values),
                  qvalue(values,.10),qvalue(values,.25),median(values),
                  qvalue(values,.75),qvalue(values,.90),qvalue(values,.99),
                  maximum(values),mu,sd,mu==0 ? 0.0 : sd/mu))
                spreads=Int[]; efficiencies=Float64[]
                for first in 1:32:COUNT
                    v=@view values[first:first+31]; hi=maximum(v)
                    push!(spreads,hi-minimum(v))
                    push!(efficiencies,hi==0 ? 1.0 : sum(v)/(32hi))
                end
                push!(warp_rows,(SEED,delta,strategy,metric,median(spreads),
                  qvalue(spreads,.90),qvalue(spreads,.99),maximum(spreads),
                  median(efficiencies),qvalue(efficiencies,.10),
                  qvalue(efficiencies,.01),minimum(efficiencies)))
            end
            distinct=[length(unique(r.branch_code for r in records[f:f+31]))
                      for f in 1:32:COUNT]
            modal=[maximum(count(==(code),records[f:f+31]) for
                           code in unique(r.branch_code for r in records[f:f+31]))/32
                   for f in 1:32:COUNT]
            push!(term_rows,(SEED,delta,strategy,minimum(distinct),maximum(distinct),
              minimum(modal),count(r->r.max_iter_reached!=0,records),
              count(r->r.output_finite==0,records),invariant))
            if SAVE_PER_CONE
                path=joinpath(per_dir,"seed_$(SEED)_delta_$(replace(string(delta),'.'=>'p'))_$(strategy).csv")
                write_rows(path,("seed","delta","strategy","cone_index","path_code",
                  "expansion_iterations","newton_attempts","newton_accepts",
                  "bisection_iterations","oracle_evaluations","termination_reason",
                  "final_bracket_width","final_residual","max_iter","nonfinite"),
                  ((SEED,delta,strategy,i,r.branch_code,
                    r.interval_expansion_iterations,r.newton_attempts,r.newton_accepts,
                    r.bisection_iterations,r.oracle_evaluations,r.termination_reason,
                    r.normalized_bracket_width,r.final_residual,r.max_iter_reached,
                    1-r.output_finite) for (i,r) in pairs(records)))
            end
        end
    end
    write_rows(joinpath(OUTPUT_DIR,"diagnostic","per_cone_summary.csv"),
      ("seed","delta","strategy","metric","minimum","q10","q25","median","q75",
       "q90","q99","maximum","mean","std","coefficient_of_variation"),summary)
    write_rows(joinpath(OUTPUT_DIR,"diagnostic","warp_root_work.csv"),
      ("seed","delta","strategy","metric","spread_median","spread_p90",
       "spread_p99","spread_max","modeled_efficiency_median",
       "modeled_efficiency_p10","modeled_efficiency_p1",
       "modeled_efficiency_min"),warp_rows)
    write_rows(joinpath(OUTPUT_DIR,"diagnostic","termination_summary.csv"),
      ("seed","delta","strategy","distinct_path_min","distinct_path_max",
       "modal_path_fraction_min","max_iter_count","nonfinite_count",
       "diagnostic_production_max_abs_error"),term_rows)
end

const CELLS=Tuple((strategy,delta) for strategy in (:threadWise,:warpWise)
                              for delta in DELTAS)
# Standard even-condition Williams rows. Rounds 1--8 use all eight rows;
# rounds 9--10 repeat the first two rows after the seed-index shift.
order_for(seed_index,round)=collect(CELLS)[
    williams_indices(length(CELLS),seed_index+mod(round-1,length(CELLS)))]

function telemetry()
    gpu=get(ENV,"PDCS_GPU_PHYSICAL",get(ENV,"CUDA_VISIBLE_DEVICES","0"))
    cmd=`nvidia-smi -i $gpu --query-gpu=uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits`
    try
        split(strip(read(cmd,String)),',') .|> strip
    catch
        ["unknown","NaN","NaN","NaN","NaN"]
    end
end

function timing!(family)
    buffers=Dict(cell=>gpu_buffers(family.cases[cell[2]]) for cell in CELLS)
    launch_rows=Any[]
    for cell in CELLS, w in 1:WARMUPS
        ms=event_time!(cell[1],buffers[cell])
        push!(launch_rows,(SEED,cell[2],cell[1],"warmup",w,0,ms,
                           "not_sampled","NaN","NaN","NaN","NaN","PASS"))
    end
    correctness=Any[]
    for delta in DELTAS
        outputs=Dict{Symbol,Vector{Float64}}()
        for strategy in (:threadWise,:warpWise)
            b=buffers[(strategy,delta)]; restore!(b); launch!(strategy,b)
            CUDA.synchronize(); outputs[strategy]=Array(b.x)
        end
        cross=maximum(abs.(outputs[:threadWise].-outputs[:warpWise]);init=0.0)
        t=sampled_cpu_error(family.cases[delta],outputs[:threadWise])
        w=sampled_cpu_error(family.cases[delta],outputs[:warpWise])
        status=all(isfinite,outputs[:threadWise]) &&
               all(isfinite,outputs[:warpWise]) &&
               max(t.worst_scaled,w.worst_scaled)<=1 && cross<=5e-8 ? "PASS" : "FAIL"
        push!(correctness,(SEED,delta,cross,t.worst_abs,t.worst_scaled,
                           w.worst_abs,w.worst_scaled,min(1024,COUNT),status))
        status=="PASS" || error("correctness gate failed seed=$SEED delta=$delta")
    end
    for round in 1:ROUNDS
        order=order_for(SEED-2026,round)
        for (position,cell) in pairs(order)
            ms=event_time!(cell[1],buffers[cell]); env=telemetry()
            push!(launch_rows,(SEED,cell[2],cell[1],"measured",round,position,ms,
                               env...,"PASS"))
        end
    end
    write_rows(joinpath(OUTPUT_DIR,"timing","launch_level.csv"),
      ("seed","delta","strategy","status","round","order_position","kernel_ms",
       "gpu_uuid","sm_clock_mhz","memory_clock_mhz","power_w","temperature_c",
       "gate_status"),launch_rows)
    seed_rows=Any[]
    for cell in CELLS
        v=[r[7] for r in launch_rows if r[4]=="measured" &&
                                      r[2]==cell[2] && r[3]==cell[1]]
        push!(seed_rows,(SEED,cell[2],cell[1],length(v),mean(v),median(v),
                         std(v),minimum(v),maximum(v)))
    end
    write_rows(joinpath(OUTPUT_DIR,"timing","seed_level.csv"),
      ("seed","delta","strategy","launches","mean_ms","median_ms","std_ms",
       "min_ms","max_ms"),seed_rows)
    write_rows(joinpath(OUTPUT_DIR,"correctness","cpu_gpu.csv"),
      ("seed","delta","cross_strategy_max_abs","thread_cpu_max_abs",
       "thread_cpu_max_scaled","warp_cpu_max_abs","warp_cpu_max_scaled",
       "sampled_cones","status"),correctness)
end

function duration!(family)
    DELTA in keys(family.cases) || error("--delta is not in cached family")
    b=gpu_buffers(family.cases[DELTA])
    for _ in 1:WARMUPS
        event_time!(STRATEGY,b)
    end
    println("READY monotonic_ns=$(time_ns()) seed=$SEED delta=$DELTA strategy=$STRATEGY")
    flush(stdout)
    if WAIT_FOR_START
        strip(readline(stdin))=="START" || error("expected START")
    end
    println("START monotonic_ns=$(time_ns())"); flush(stdout)
    first=time_ns()
    println("FIRST_PROJECTION_START monotonic_ns=$first"); flush(stdout)
    launches=0; restore_ns=0; projection_ms=0.0; started=time()
    while time()-started<DURATION
        t=time_ns(); restore!(b); CUDA.synchronize(); restore_ns+=time_ns()-t
        projection_ms+=event_time!(STRATEGY,b;restore=false); launches+=1
    end
    last=time_ns(); wall=time()-started
    println("LAST_PROJECTION_STOP monotonic_ns=$last")
    @printf("UTILIZATION_COMPLETE seed=%d delta=%.10g strategy=%s launches=%d elapsed_seconds=%.6f restore_seconds=%.6f projection_seconds=%.6f\n",
      SEED,DELTA,STRATEGY,launches,wall,restore_ns/1e9,projection_ms/1e3)
    println("DONE monotonic_ns=$(time_ns())"); flush(stdout)
    write_rows(joinpath(OUTPUT_DIR,"duration_seed$(SEED)_delta$(replace(string(DELTA),'.'=>'p'))_$(STRATEGY)_ledger.csv"),
      ("seed","delta","strategy","launches","wall_seconds","restore_seconds",
       "projection_seconds","host_gap_seconds","first_projection_monotonic_ns",
       "last_projection_monotonic_ns"),
      [(SEED,DELTA,STRATEGY,launches,wall,restore_ns/1e9,projection_ms/1e3,
        max(0,wall-restore_ns/1e9-projection_ms/1e3),first,last)])
end

if DRY_RUN
    println("DRY_RUN mode=$MODE seed=$SEED count=$COUNT dimension=$DIMENSION deltas=$(join(DELTAS,',')) delta=$DELTA strategy=$STRATEGY output=$OUTPUT_DIR")
    exit()
end
CUDA.functional() || error("CUDA is not functional")
family=family_from_cache()
if MODE in (:pilot,:generate)
    isfile(joinpath(OUTPUT_DIR,"case_cache.jls")) &&
      error("refusing to overwrite existing case cache in $OUTPUT_DIR")
    generator_artifacts(family)
    diagnostic!(family)
    mkpath(OUTPUT_DIR)
    serialize(joinpath(OUTPUT_DIR,"case_cache.jls"),family)
    println("GENERATE_COMPLETE seed=$SEED cache=$(joinpath(OUTPUT_DIR,"case_cache.jls"))")
elseif MODE===:timing
    isfile(joinpath(OUTPUT_DIR,"timing","launch_level.csv")) &&
      error("refusing to overwrite existing timing output in $OUTPUT_DIR")
    timing!(family)
    println("TIMING_COMPLETE seed=$SEED output=$OUTPUT_DIR")
else
    ledger=joinpath(OUTPUT_DIR,
      "duration_seed$(SEED)_delta$(replace(string(DELTA),'.'=>'p'))_$(STRATEGY)_ledger.csv")
    isfile(ledger) && error("refusing to overwrite existing duration ledger: $ledger")
    duration!(family)
end
