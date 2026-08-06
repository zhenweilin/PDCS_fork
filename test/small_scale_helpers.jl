const _SMALL_SCALE_CBF_EXTENSIONS = (".cbf", ".cbf.gz", ".cbf.bz2")
const _SMALL_SCALE_CONE_TYPES = (
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.DualExponentialCone,
)

is_cbf_path(path::AbstractString) = isfile(path) && any(
    extension -> endswith(lowercase(path), extension),
    _SMALL_SCALE_CBF_EXTENSIONS,
)

function collect_cbf_paths(root::AbstractString)
    isdir(root) || throw(ArgumentError("small-scale directory does not exist: $root"))
    paths = String[]
    for (directory, _, names) in walkdir(root)
        for name in names
            path = joinpath(directory, name)
            is_cbf_path(path) && push!(paths, path)
        end
    end
    sort!(paths)
    return paths
end

mutable struct _PDCSBaselineCapture <: MOI.AbstractOptimizer
    source::Any
    prototype::PDCS_CPU.Optimizer
end

_PDCSBaselineCapture() = _PDCSBaselineCapture(nothing, PDCS_CPU.Optimizer())

MOI.is_empty(::_PDCSBaselineCapture) = true
MOI.empty!(::_PDCSBaselineCapture) = nothing
MOI.supports_incremental_interface(::_PDCSBaselineCapture) = false

function MOI.supports_constraint(
    capture::_PDCSBaselineCapture,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports_constraint(capture.prototype, F, S)
end

function MOI.supports(
    capture::_PDCSBaselineCapture,
    attribute::MOI.AbstractModelAttribute,
)
    return MOI.supports(capture.prototype, attribute)
end

function MOI.default_cache(::_PDCSBaselineCapture, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(PDCS_CPU.OptimizerCache{Integer}())
end

function MOI.optimize!(capture::_PDCSBaselineCapture, source::MOI.ModelLike)
    capture.source = source
    return MOI.Utilities.identity_index_map(source), false
end

function _baseline_cache(path, workers)
    capture = _PDCSBaselineCapture()
    mode = endswith(lowercase(path), ".cbf.bz2") ? :auto : :force
    model = JumpRW.read_cbf(
        path;
        optimizer=() -> capture,
        fast_path=mode,
        workers,
        add_bridges=true,
    )
    JuMP.optimize!(model)
    capture.source === nothing && error("baseline bridge graph did not produce a cache")
    return capture.source
end

function _objective_snapshot(cache)
    sense = MOI.get(cache, MOI.ObjectiveSense())
    objective = zeros(Float64, length(cache.model.variables.lower))
    objective_constant = 0.0
    function_type = MOI.get(cache, MOI.ObjectiveFunctionType())
    if function_type !== nothing
        function_ = MOI.get(cache, MOI.ObjectiveFunction{function_type}())
        objective_constant = MOI.constant(function_)
        for term in function_.terms
            objective[term.variable.value] += term.coefficient
        end
    end
    return (; sense, objective_constant, objective)
end

function _cone_signature(cache)
    signature = Tuple{DataType,Int}[]
    for set_type in _SMALL_SCALE_CONE_TYPES
        index_type = MOI.ConstraintIndex{
            MOI.VectorAffineFunction{Float64},
            set_type,
        }
        for index in MOI.get(cache, MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{Float64},
            set_type,
        }())
            set = MOI.get(cache, MOI.ConstraintSet(), index::index_type)
            push!(signature, (set_type, MOI.dimension(set)))
        end
    end
    return signature
end

function _require_equal(name, left, right)
    left == right || error("structural mismatch in $name")
    return nothing
end

function _require_approx(name, left, right; atol, rtol=0.0)
    if !isapprox(left, right; atol, rtol)
        absolute_error = left isa Number ?
            abs(left - right) : maximum(abs.(left .- right))
        left_scale = left isa Number ? abs(left) : maximum(abs, left)
        right_scale = right isa Number ? abs(right) : maximum(abs, right)
        scale = max(left_scale, right_scale, eps(Float64))
        relative_error = absolute_error / scale
        error(
            "numeric mismatch in $name " *
            "(max_abs=$absolute_error, scale=$scale, max_rel=$relative_error, " *
            "atol=$atol, rtol=$rtol)",
        )
    end
    return nothing
end

function _compare_caches(bulk, baseline)
    bulk_model = bulk.model
    baseline_model = baseline.model
    bulk_matrix = bulk_model.constraints.coefficients
    baseline_matrix = baseline_model.constraints.coefficients

    _require_equal(
        "number of variables",
        length(bulk_model.variables.lower),
        length(baseline_model.variables.lower),
    )
    _require_equal("number of rows", bulk_matrix.m, baseline_matrix.m)
    _require_equal("column pointers", bulk_matrix.colptr, baseline_matrix.colptr)
    _require_equal("row indices", bulk_matrix.rowval, baseline_matrix.rowval)
    _require_approx("matrix coefficients", bulk_matrix.nzval, baseline_matrix.nzval; atol=1e-12)
    _require_approx(
        "affine constants",
        bulk_model.constraints.constants.b,
        baseline_model.constraints.constants.b;
        atol=1e-12,
    )
    _require_equal("variable lower bounds", bulk_model.variables.lower, baseline_model.variables.lower)
    _require_equal("variable upper bounds", bulk_model.variables.upper, baseline_model.variables.upper)
    _require_equal("cone signature", _cone_signature(bulk), _cone_signature(baseline))

    bulk_objective = _objective_snapshot(bulk)
    baseline_objective = _objective_snapshot(baseline)
    _require_equal("objective sense", bulk_objective.sense, baseline_objective.sense)
    _require_approx(
        "objective constant",
        bulk_objective.objective_constant,
        baseline_objective.objective_constant;
        atol=1e-12,
    )
    _require_approx(
        "objective coefficients",
        bulk_objective.objective,
        baseline_objective.objective;
        atol=1e-12,
    )
    return nothing
end

function _small_scale_optimizer()
    optimizer = PDCS_CPU.Optimizer()
    attributes = Pair{String,Any}[
        "abs_tol" => 1e-5,
        "rel_tol" => 1e-5,
        "verbose" => 0,
        "time_limit_secs" => 90.0,
        "max_outer_iter" => 3,
        "max_inner_iter" => 2_000,
        "use_scaling" => true,
        "rescaling_method" => :ruiz_pock_chambolle,
        "use_adaptive_restart" => true,
        "use_adaptive_step_size_weight" => true,
        "use_resolving" => true,
        "use_accelerated" => false,
        "use_aggressive" => true,
        "use_reflection" => true,
        "use_halpern" => false,
        "halpern_mode" => :inline,
        "print_freq" => 2_000,
        "restart_check_freq" => 2_000,
        "check_terminate_freq" => 2_000,
    ]
    for (name, value) in attributes
        MOI.set(optimizer, MOI.RawOptimizerAttribute(name), value)
    end
    return optimizer
end

function _solver_snapshot(optimizer)
    result_count = MOI.get(optimizer, MOI.ResultCount())
    return (
        termination=MOI.get(optimizer, MOI.TerminationStatus()),
        primal_status=MOI.get(optimizer, MOI.PrimalStatus()),
        dual_status=MOI.get(optimizer, MOI.DualStatus()),
        result_count,
        iterations=MOI.get(optimizer, PDCS_CPU.PDHGIterations()),
        objective=result_count > 0 ? MOI.get(optimizer, MOI.ObjectiveValue()) : NaN,
        primal=copy(optimizer.sol.primal),
        dual=copy(optimizer.sol.dual),
    )
end

function _solve_generic(path, workers)
    optimizer = _small_scale_optimizer()
    mode = endswith(lowercase(path), ".cbf.bz2") ? :auto : :force
    model = JumpRW.read_cbf(
        path;
        optimizer=() -> optimizer,
        fast_path=mode,
        workers,
        add_bridges=true,
    )
    JuMP.optimize!(model)
    return _solver_snapshot(optimizer)
end

function _solve_bulk(data)
    optimizer = _small_scale_optimizer()
    model = PDCS_CPU.model_from_conic_data(data; optimizer)
    JuMP.optimize!(model)
    return _solver_snapshot(optimizer)
end

function _compare_solutions(generic, bulk)
    _require_equal("termination status", generic.termination, bulk.termination)
    _require_equal("primal status", generic.primal_status, bulk.primal_status)
    _require_equal("dual status", generic.dual_status, bulk.dual_status)
    _require_equal("result count", generic.result_count, bulk.result_count)
    _require_equal("iteration count", generic.iterations, bulk.iterations)
    _require_equal("primal vector length", length(generic.primal), length(bulk.primal))
    _require_equal("dual vector length", length(generic.dual), length(bulk.dual))

    all(isfinite, generic.primal) || error("generic primal vector contains a non-finite value")
    all(isfinite, bulk.primal) || error("bulk primal vector contains a non-finite value")
    all(isfinite, generic.dual) || error("generic dual vector contains a non-finite value")
    all(isfinite, bulk.dual) || error("bulk dual vector contains a non-finite value")
    _require_approx("primal solution", generic.primal, bulk.primal; atol=5e-4, rtol=5e-4)
    _require_approx("dual solution", generic.dual, bulk.dual; atol=5e-4, rtol=5e-4)

    if isfinite(generic.objective) || isfinite(bulk.objective)
        isfinite(generic.objective) || error("generic objective is not finite")
        isfinite(bulk.objective) || error("bulk objective is not finite")
        _require_approx(
            "objective value",
            generic.objective,
            bulk.objective;
            atol=1e-4,
            rtol=1e-4,
        )
    elseif !isequal(generic.objective, bulk.objective)
        error("non-finite objective values differ")
    end
    return nothing
end

function verify_small_scale_instance(path::AbstractString; workers::Integer)
    data = JumpRW.read_cbf_conic_data(path; workers)
    bulk_cache = PDCS_CPU.conic_cache_from_data(data)
    baseline_cache = _baseline_cache(path, workers)
    _compare_caches(bulk_cache, baseline_cache)

    generic = _solve_generic(path, workers)
    bulk = _solve_bulk(data)
    _compare_solutions(generic, bulk)
    if data.num_variables + data.num_rows > 100_000
        GC.gc(false)
    end
    return (
        generic=(
            termination=generic.termination,
            primal_status=generic.primal_status,
            dual_status=generic.dual_status,
            objective=generic.objective,
        ),
        bulk=(
            termination=bulk.termination,
            primal_status=bulk.primal_status,
            dual_status=bulk.dual_status,
            objective=bulk.objective,
        ),
    )
end
