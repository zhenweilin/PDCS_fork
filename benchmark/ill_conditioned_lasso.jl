#!/usr/bin/env julia

using Dates
using JuMP
using LinearAlgebra
using Printf
using Serialization
using SparseArrays
using Statistics
using PDCS: PDCS_CPU, PDCS_GPU
using MosekTools
using SCS
using Clarabel
import MathOptInterface as MOI

include("rebuttal/common.jl")
include("rebuttal/ill_conditioned_lasso_cases.jl")
using .RebuttalCommon
using .IllConditionedLassoCases

const MODE=Symbol(option("mode","generate"))
const M=parse(Int,option("m","20000"))
const N=parse(Int,option("n","100000"))
const S=parse(Int,option("support","200"))
const D=parse(Int,option("sparsity","20"))
const KVALUES=float_list(option("k-values","1,100,10000,1000000"))
const SEED=parse(Int,option("seed","2026"))
const PANEL=Symbol(option("panel","fixed_lambda"))
const LAMBDA=parse(Float64,option("lambda","0.01"))
const RESIDUAL_NORM=parse(Float64,option("residual-norm","1"))
const SOLVER=Symbol(option("solver","pdcs_cpu"))
const TOL=parse(Float64,option("tol","1e-3"))
const TIME_LIMIT=parse(Float64,option("time-limit","600"))
const MOSEK_LICENSE=option("mosek-license",get(ENV,"MOSEKLM_LICENSE_FILE",joinpath(@__DIR__,"..","mosek_lic","mosek.lic")))
const OUTPUT=abspath(option("output-dir","benchmark/results/rebuttal/ill_conditioned_lasso"))
const CACHE=option("cache","")
const WARMUP_CACHE=option("warmup-cache","")
const DRY_RUN=flag("dry-run")
const VERBOSE=flag("verbose")
const VERBOSE_LEVEL=parse(Int,option("verbose-level",VERBOSE ? "1" : "0"))
const PRINT_FREQ=parse(Int,option("print-freq","2000"))
function bool_option(name,default)
    value=lowercase(option(name,default ? "true" : "false"))
    value in ("true","1","yes") && return true
    value in ("false","0","no") && return false
    error("--$name must be true or false")
end
# These options make CPU/GPU algorithm-parity experiments possible without
# changing a solver default.  The GPU-only duality-gap restart is the first
# controlled factor to test on the same cached instance.
const USE_AGGRESSIVE=bool_option("use-aggressive",true)
const USE_DUALITY_GAP_RESTART=bool_option("use-duality-gap-restart",true)
const USE_KKT_RESTART=bool_option("use-kkt-restart",false)

MODE in (:generate,:solve,:smoke) || error("--mode must be generate, solve, or smoke")
PANEL in (:fixed_lambda,:fixed_residual) || error("invalid --panel")
VERBOSE_LEVEL in 0:2 || error("--verbose-level must be 0, 1, or 2")
PRINT_FREQ > 0 || error("--print-freq must be positive")

function rows(path,header,values)
    mkpath(dirname(path))
    open(path,"w") do io
        println(io,join(header,','))
        for value in values
            println(io,join(value,','))
        end
    end
end
function append_rows(path,header,values)
    mkpath(dirname(path)); exists=isfile(path)
    open(path,"a") do io
        exists || println(io,join(header,','))
        for value in values
            println(io,join(value,','))
        end
    end
end
label(K)=replace(@sprintf("%.0f",K),".0"=>"")
cache_path(K)=isempty(CACHE) ? joinpath(OUTPUT,"instances","seed_$(SEED)_K_$(label(K)).jls") : CACHE

function save_instance(inst)
    path=cache_path(inst.K)
    isfile(path) && error("refusing to overwrite instance cache: $path")
    mkpath(dirname(path)); serialize(path,inst)
    v=verify_instance(inst); h=instance_hashes(inst)
    λmax=2norm(inst.A' * inst.b,Inf)
    append_rows(joinpath(OUTPUT,"instance_manifest.csv"),
      ("instance_id","seed","panel","m","n","s","d","K","rho","nnz",
       "lambda","lambda_max","b_norm2","rstar_norm2","xstar_norm2",
       "kappa_theory","kappa_measured","kkt_active","kkt_inactive",
       "matrix_hash","b_hash","xstar_hash"),
      [("seed$(inst.seed)_K$(label(inst.K))",inst.seed,inst.panel,size(inst.A,1),
        size(inst.A,2),length(inst.support),inst.sparsity,inst.K,inst.rho,nnz(inst.A),
        inst.lambda,λmax,norm(inst.b),norm(inst.rstar),norm(inst.xstar),
        v.kappa_theory,v.kappa_measured,v.kkt_active,v.kkt_inactive,
        h.matrix,h.b,h.xstar)])
    println("INSTANCE_GENERATED cache=$path K=$(inst.K) kappa=$(v.kappa_measured)")
end

"""Build SCS with MOI's standard bridges only.

SCS 2.6.4 advertises several optional PSD-specific bridge types whose
partially-parameterized signatures are incompatible with the current MOI
bridge graph. This Lasso model uses only linear, nonnegative, equality, and
SOC constraints, so those PSD bridges are neither needed nor appropriate.
"""
function scs_standard_bridge_optimizer()
    raw_optimizer=SCS.Optimizer()
    # The cache index type is chosen from the linear solver.  In particular,
    # SCS's GPU artifact uses Cint, whereas the default direct solver uses
    # Int64.  Set this before calling `MOI.default_cache`; changing it later
    # produces `T == scsint_t(linear_solver)` assertion failures.
    if SOLVER===:scs_indirect
        MOI.set(raw_optimizer,MOI.RawOptimizerAttribute("linear_solver"),
                SCS.IndirectSolver)
    elseif SOLVER===:scs_gpu
        MOI.set(raw_optimizer,MOI.RawOptimizerAttribute("linear_solver"),
                SCS.GpuIndirectSolver)
    end
    cache=MOI.default_cache(raw_optimizer,Float64)
    cached_optimizer=MOI.Utilities.CachingOptimizer(cache,raw_optimizer)
    optimizer=MOI.Bridges.LazyBridgeOptimizer(cached_optimizer)
    MOI.Bridges.Variable.add_all_bridges(optimizer,Float64)
    MOI.Bridges.Constraint.add_all_bridges(optimizer,Float64)
    MOI.Bridges.Objective.add_all_bridges(optimizer,Float64)
    return optimizer
end

function optimizer_factory(solver::Symbol)
    if solver===:pdcs_cpu
        return PDCS_CPU.Optimizer
    elseif solver===:pdcs_gpu
        Base.eval(Main,:(using CUDA))
        Base.eval(Main,:(CUDA.functional())) || error("CUDA is not functional")
        return PDCS_GPU.Optimizer
    elseif solver in (:scs,:scs_indirect)
        return scs_standard_bridge_optimizer
    elseif solver===:scs_gpu
        error(
          "SCS GPU requires the isolated CUDA 11.8 environment; use " *
          "benchmark/run_scs_gpu_lasso.sh"
        )
    elseif solver===:clarabel
        return Clarabel.Optimizer
    elseif solver===:cuclarabel
        error(
          "CuClarabel is the GPU branch of the Clarabel module and cannot " *
          "coexist with CPU Clarabel in this project; use " *
          "benchmark/ill_conditioned_lasso_cuclarabel.jl with " *
          "--project=benchmark/cuclarabel_env"
        )
    elseif solver in (:mosek,:mosek_nopresolve)
        isfile(MOSEK_LICENSE) || error("MOSEK license file is missing: $MOSEK_LICENSE")
        ENV["MOSEKLM_LICENSE_FILE"]=MOSEK_LICENSE
        return MosekTools.Optimizer
    end
    error("unsupported solver: $solver")
end

"""Build the exact SOCP epigraph in the manuscript normalization.
The returned x is always recovered as x_pos - x_neg."""
function build_jump_model(inst,optimizer)
    m,n=size(inst.A)
    # External optimizer modules can be loaded dynamically.  Construct JuMP's
    # model in the latest world so MOI methods from that module are visible.
    # SCS already receives an explicit standard-only bridge optimizer above.
    model=Base.invokelatest(
      Model,optimizer;
      add_bridges=!(SOLVER in (:scs,:scs_indirect,:scs_gpu)))
    if VERBOSE_LEVEL >= 2
        objective_type=MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
        println("JUMP_MODEL solver=$SOLVER backend=$(typeof(backend(model))) " *
                "affine_objective=$(MOI.supports(backend(model),objective_type))")
    end
    VERBOSE_LEVEL == 0 && (try set_silent(model) catch; end)
    try set_time_limit_sec(model,TIME_LIMIT) catch; end
    attributes = SOLVER in (:pdcs_cpu,:pdcs_gpu) ?
      (("abs_tol",TOL),("rel_tol",TOL),("time_limit_secs",TIME_LIMIT),
       ("verbose",VERBOSE_LEVEL),("print_freq",PRINT_FREQ),
       ("use_aggressive",USE_AGGRESSIVE)) :
      SOLVER in (:scs,:scs_indirect,:scs_gpu) ?
        (("eps_abs",TOL),("eps_rel",TOL),) :
      SOLVER in (:clarabel,:cuclarabel) ?
        (("tol_gap_abs",TOL),("tol_gap_rel",TOL),("tol_feas",TOL)) :
      SOLVER in (:mosek,:mosek_nopresolve) ?
        (("MSK_DPAR_INTPNT_CO_TOL_PFEAS",TOL),
         ("MSK_DPAR_INTPNT_CO_TOL_DFEAS",TOL),
         ("MSK_DPAR_INTPNT_CO_TOL_REL_GAP",TOL)) : ()
    for (name,value) in attributes
        try set_optimizer_attribute(model,name,value) catch; end
    end
    if SOLVER===:scs_indirect
        set_optimizer_attribute(model,"linear_solver",SCS.IndirectSolver)
    elseif SOLVER===:scs_gpu
        set_optimizer_attribute(model,"linear_solver",SCS.GpuIndirectSolver)
    elseif SOLVER===:mosek_nopresolve
        set_optimizer_attribute(
          model,"MSK_IPAR_PRESOLVE_USE",
          MosekTools.Mosek.MSK_PRESOLVE_MODE_OFF)
    end
    if SOLVER===:pdcs_gpu
        for (name,value) in (("use_duality_gap_restart",USE_DUALITY_GAP_RESTART),
                             ("use_kkt_restart",USE_KKT_RESTART))
            try set_optimizer_attribute(model,name,value) catch; end
        end
    end
    @variable(model,xpos[1:n]>=0)
    @variable(model,xneg[1:n]>=0)
    @variable(model,y[1:m])
    @variable(model,r>=0)
    @objective(model,Min,2r+inst.lambda*sum(xpos[j]+xneg[j] for j in 1:n))
    @constraint(model,y .== inst.A*(xpos-xneg)-inst.b)
    @constraint(model,[(1+r)/sqrt(2);(1-r)/sqrt(2);y] in SecondOrderCone())
    model,xpos,xneg
end

function solver_iteration_count(model)
    attr = if SOLVER in (:scs,:scs_indirect,:scs_gpu)
        SCS.ADMMIterations()
    elseif SOLVER===:pdcs_cpu
        PDCS_CPU.PDHGIterations()
    elseif SOLVER===:pdcs_gpu
        PDCS_GPU.PDHGIterations()
    elseif SOLVER in (:clarabel,:cuclarabel,:mosek,:mosek_nopresolve)
        MOI.BarrierIterations()
    else
        MOI.SimplexIterations()
    end
    return try
        MOI.get(backend(model),attr)
    catch
        missing
    end
end

function solver_native_solve_time(model)
    return try
        solve_time(model)
    catch
        try
            MOI.get(backend(model),MOI.SolveTimeSec())
        catch
            NaN
        end
    end
end

function synchronize_solver()
    SOLVER in (:pdcs_gpu,:scs_gpu) || return
    Base.invokelatest(Base.eval(Main,:(CUDA.synchronize)))
end

"""Compile and initialize the selected solver in the same Julia process.

The warm-up result is intentionally not written to the formal result CSV.
It uses a separate, smaller cached instance with the same JuMP cone types.
"""
function warmup_solver(inst)
    println("WARMUP_START solver=$SOLVER cache=$WARMUP_CACHE")
    started=time()
    optimizer=optimizer_factory(SOLVER)
    model,_,_=build_jump_model(inst,optimizer)
    optimize!(model)
    synchronize_solver()
    println("WARMUP_COMPLETE solver=$SOLVER status=$(termination_status(model)) seconds=$(time()-started)")
    return nothing
end

function solve_instance(inst;input_seconds=NaN)
    hashes=instance_hashes(inst)
    println("INSTANCE cache=$CACHE seed=$(inst.seed) K=$(inst.K) " *
            "m=$(size(inst.A,1)) n=$(size(inst.A,2)) nnz=$(nnz(inst.A)) " *
            "matrix_hash=$(hashes.matrix) b_hash=$(hashes.b) " *
            "xstar_hash=$(hashes.xstar) input_seconds=$input_seconds")
    started=time(); status="numerical_failure"; message=""; x=fill(NaN,size(inst.A,2))
    iterations=missing; objective=NaN; setup_seconds=NaN
    optimize_wall_seconds=NaN; native_solve_seconds=NaN
    try
        optimizer=optimizer_factory(SOLVER)
        model,xpos,xneg=build_jump_model(inst,optimizer)
        setup_seconds=time()-started
        optimize_started=time()
        optimize!(model)
        synchronize_solver()
        optimize_wall_seconds=time()-optimize_started
        term=termination_status(model); message=string(term)
        iterations=solver_iteration_count(model)
        native_solve_seconds=solver_native_solve_time(model)
        if term==MOI.TIME_LIMIT
            # Some solvers expose an incumbent at the limit.  Keep its
            # diagnostics, but do not misrepresent it as a completed solve.
            has_values(model) && (x=value.(xpos).-value.(xneg); objective=objective_value(model))
            status="timeout"
        elseif has_values(model)
            x=value.(xpos).-value.(xneg)
            objective=objective_value(model)
            status=term in (MOI.OPTIMAL,MOI.ALMOST_OPTIMAL) ? "returned" : "returned_nonoptimal"
        end
    catch err
        bt=catch_backtrace()
        message=sprint(showerror,err)
        if VERBOSE_LEVEL > 0
            showerror(stderr,err,bt)
            println(stderr)
        end
        if occursin("not installed",message)
            status="input_or_conversion_failure"
        elseif occursin("CUDA is not functional",message)
            status="not_attempted_hardware_limit"
        end
    end
    elapsed=time()-started
    metric=all(isfinite,x) ? lasso_metrics(inst,x) : nothing
    if status in ("returned","returned_nonoptimal")
        status=(metric.normalized_stationarity<=TOL && metric.x_error<=max(10TOL,1e-4)) ?
          "solved_verified" : "solver_claimed_solved_but_inaccurate"
    end
    append_rows(joinpath(OUTPUT,"seed_level_results.csv"),
      ("seed","panel","K","solver","tolerance","status","solve_seconds",
       "iterations","objective","normalized_kkt","x_error","objective_error",
       "precision","recall","message","setup_seconds","optimize_wall_seconds",
       "native_solve_seconds","input_seconds","matrix_hash","b_hash",
       "xstar_hash"),
      [(inst.seed,inst.panel,inst.K,SOLVER,TOL,status,elapsed,iterations,objective,
        metric===nothing ? NaN : metric.normalized_stationarity,
        metric===nothing ? NaN : metric.x_error,
        metric===nothing ? NaN : metric.objective_error,
        metric===nothing ? NaN : metric.precision,
        metric===nothing ? NaN : metric.recall,
        replace(replace(message,','=>';'),'\n'=>' '),setup_seconds,
        optimize_wall_seconds,native_solve_seconds,input_seconds,hashes.matrix,
        hashes.b,hashes.xstar)])
    println("SOLVE_COMPLETE solver=$SOLVER K=$(inst.K) status=$status " *
            "seconds=$elapsed setup_seconds=$setup_seconds " *
            "optimize_wall_seconds=$optimize_wall_seconds " *
            "native_solve_seconds=$native_solve_seconds")
end

if DRY_RUN
    println("DRY_RUN mode=$MODE m=$M n=$N s=$S d=$D K=$(join(KVALUES,',')) seed=$SEED solver=$SOLVER")
    exit()
end
if MODE===:generate
    for K in KVALUES
        let inst=generate_instance(M,N,S,D,K,SEED;panel=PANEL,lambda=LAMBDA,
          residual_norm=RESIDUAL_NORM)
            save_instance(inst)
        end
    end
elseif MODE===:solve
    isfile(CACHE) || error("--cache is required for solve mode")
    if !isempty(WARMUP_CACHE)
        isfile(WARMUP_CACHE) || error("--warmup-cache does not exist: $WARMUP_CACHE")
        warmup_solver(deserialize(WARMUP_CACHE))
    end
    input_started=time()
    inst=deserialize(CACHE)
    input_seconds=time()-input_started
    solve_instance(inst;input_seconds)
else
    inst=generate_instance(min(M,1000),min(N,5000),min(S,40),min(D,10),
      first(KVALUES),SEED;panel=PANEL,lambda=LAMBDA,residual_norm=RESIDUAL_NORM)
    v=verify_instance(inst)
    println("SMOKE_GENERATOR_PASS K=$(inst.K) kappa=$(v.kappa_measured) kkt=$(v.normalized_stationarity)")
    solve_instance(inst)
end
