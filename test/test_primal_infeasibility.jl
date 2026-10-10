using Test
using LinearAlgebra

# Exercise the production convergence functions with small array-backed fixtures.
# Only CUDA.maximum/abs are used by these functions, so their arithmetic can be
# checked on a CPU without loading CUDA or requiring a GPU driver. This does not
# test CUDA kernels or the solver's projection implementations.
function convergence_test_module(backend, diagonal)
    scope = Module(gensym(:PrimalInfeasibility))
    Core.eval(scope, quote
        using LinearAlgebra
        const CUDA = Base
        const rpdhgSolver = NamedTuple
        const primalVector = NamedTuple
        const dualVector = NamedTuple
        const solVecPrimal = NamedTuple
        const solVecDual = NamedTuple
        time_proj = 0.0
        time_proj_dual_slack = 0.0

        Base.@kwdef mutable struct PDHGCLPConvergeInfo
            l_2_abs_primal_res::Float64 = NaN
            l_2_rel_primal_res::Float64 = NaN
            l_inf_abs_primal_res::Float64 = NaN
            l_inf_rel_primal_res::Float64 = NaN
            l_2_abs_dual_res::Float64 = NaN
            l_2_rel_dual_res::Float64 = NaN
            l_inf_abs_dual_res::Float64 = NaN
            l_inf_rel_dual_res::Float64 = NaN
            abs_gap::Float64 = NaN
            rel_gap::Float64 = NaN
            primal_objective::Float64 = NaN
            dual_objective::Float64 = NaN
            status::Symbol = :continue
        end
    end)
    function_name = diagonal ? :converge_info_calculation_diagonal! :
                               :converge_info_calculation
    suffix = diagonal ? "_scaling" : ""
    source = joinpath(
        @__DIR__, "..", "src", "pdcs_$backend",
        "rpdhg_alg_$(backend)_gen$suffix.jl",
    )
    # Include the actual function unchanged, without unrelated solver methods.
    Base.include(scope, source) do expr
        if expr isa Expr && expr.head == :function &&
           expr.args[1].args[1] == function_name
            return expr
        end
        return nothing
    end
    @assert isdefined(scope, function_name)
    return scope, getfield(scope, function_name)
end

function convergence_fixture(backend, diagonal, x, h, c, y)
    # Three constraint rows and two variables distinguish constraint slack from
    # dual slack. The cone and variable bounds are both nonnegative orthants.
    G = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    # xbox must alias x, as it does in the production primalVector.
    function primal_buffer(v = zeros(2))
        buffer = copy(v)
        return (; x = buffer, xbox = buffer)
    end
    dual(v = zeros(3)) = (; y = copy(v))
    slack = (
        primal_sol = primal_buffer(),
        primal_sol_lag = primal_buffer(),
        primal_sol_mean = primal_buffer(),
    )
    dual_sol_temp = (
        dual_sol_mean = dual(), dual_sol_lag = dual(), dual_sol_temp = dual(),
    )
    coeff = (; G, h)
    raw_data = (
        c = copy(c), coeff, coeffTrans = (; G = transpose(G), h),
        bl_finite = zeros(2), bu_finite = zeros(2),
        hNrm1 = norm(h, 1), hNrm2 = norm(h, 2), hNrmInf = norm(h, Inf),
        cNrm1 = norm(c, 1), cNrm2 = norm(c, 2), cNrmInf = norm(c, Inf),
    )
    data = merge(raw_data, (
        d_c = copy(c), raw_data,
        d_bl_finite = raw_data.bl_finite, d_bu_finite = raw_data.bu_finite,
        diagonal_scale = (; Dl_temp = dual(), Dr_temp = primal_buffer()),
        # A diagonal path must normalize recovered residuals with raw norms.
        hNrm1 = diagonal ? 1.0e12 : raw_data.hNrm1,
        hNrm2 = diagonal ? 1.0e12 : raw_data.hNrm2,
        hNrmInf = diagonal ? 1.0e12 : raw_data.hNrmInf,
    ))
    solver = (;
        data,
        primalMV! = (coeff, x, out) -> mul!(out.y, coeff.G, x),
        adjointMV! = (coeff, y, out) -> mul!(
            out, backend == :gpu && diagonal ? transpose(coeff.G) : coeff.G,
            y.y,
        ),
        addCoeffd! = (coeff, out, alpha) -> (out.y .+= alpha .* coeff.h),
        dotCoeffd = (coeff, y) -> dot(coeff.h, y.y),
        sol = (;
            x = (; slack_proj! = (v, _) -> (v.x .= max.(v.x, 0.0))),
            y = (; con_proj! = v -> (v.y .= max.(v.y, 0.0))),
        ),
    )
    return (; solver, primal_sol = primal_buffer(x), dual_sol = dual(y),
            slack, dual_sol_temp)
end

@testset "Primal infeasibility follows the README" begin
    cases = (
        # Hand-calculated residuals and denominators for G defined above.
        (name = "projection dominates", x = [5.0, -1.0], h = [-5.0, -5.0, 5.0],
         abs_inf = 1.0, abs_l2 = 1.0, scale_inf = 11.0, scale_l2 = 1.0 + sqrt(116.0)),
        (name = "Gx dominates", x = [10.0, -1.0], h = [5.0, 1.0, 8.0],
         abs_inf = 2.0, abs_l2 = 2.0, scale_inf = 11.0, scale_l2 = 1.0 + sqrt(182.0)),
        (name = "h dominates", x = [1.0, 2.0], h = [10.0, -1.0, 1.0],
         abs_inf = 9.0, abs_l2 = 9.0, scale_inf = 11.0, scale_l2 = 1.0 + sqrt(102.0)),
        (name = "multiple violations", x = [-1.0, -2.0], h = zeros(3),
         abs_inf = 3.0, abs_l2 = sqrt(14.0), scale_inf = 4.0, scale_l2 = 1.0 + sqrt(14.0)),
        (name = "feasible", x = [1.0, 2.0], h = zeros(3),
         abs_inf = 0.0, abs_l2 = 0.0, scale_inf = 4.0, scale_l2 = 1.0 + sqrt(14.0)),
        (name = "zero", x = zeros(2), h = zeros(3),
         abs_inf = 0.0, abs_l2 = 0.0, scale_inf = 1.0, scale_l2 = 1.0),
    )
    for backend in (:cpu, :gpu), diagonal in (false, true)
        scope, calculate! = convergence_test_module(backend, diagonal)
        @testset "$backend diagonal=$diagonal" begin
            for case in cases
                @testset "$(case.name)" begin
                    # Primal infeasibility must be independent of c and y.
                    for c in ([1.0, 5.0], [1.0e8, 5.0e8]),
                        y in (zeros(3), [7.0, 3.0, 2.0])
                        fixture = convergence_fixture(
                            backend, diagonal, case.x, case.h, c, y,
                        )
                        info = Base.invokelatest(scope.PDHGCLPConvergeInfo)
                        Base.invokelatest(calculate!; fixture..., converge_info = info)
                        @test info.l_inf_abs_primal_res ≈ case.abs_inf
                        @test info.l_inf_rel_primal_res ≈ case.abs_inf / case.scale_inf
                        @test info.l_2_abs_primal_res ≈ case.abs_l2
                        @test info.l_2_rel_primal_res ≈ case.abs_l2 / case.scale_l2
                    end
                end
            end
            if backend == :gpu && diagonal
                # The old dual-slack denominator made this nonfeasible point
                # satisfy every optimality test: the gap and dual residual are
                # zero, but primal infeasibility is 1/11, not about 2e-9.
                Core.eval(scope, quote
                    struct PDHGCLPInfeaInfo end
                    struct PDHGCLPInfo end
                    struct PDHGCLPParameters end
                end)
                Base.include(scope, joinpath(
                    @__DIR__, "..", "src", "pdcs_gpu", "termination.jl",
                ))
                fixture = convergence_fixture(
                    backend, diagonal, [5.0, 0.0], [-5.0, -5.0, 6.0],
                    [0.0, 5.0e8], zeros(3),
                )
                info = Base.invokelatest(scope.PDHGCLPConvergeInfo)
                Base.invokelatest(calculate!; fixture..., converge_info = info)
                @test info.l_inf_rel_dual_res == 0.0
                @test info.rel_gap == 0.0
                Base.invokelatest(
                    scope.optimality_criteria_met;
                    rel_tol = 1.0e-6, abs_tol = 1.0e-6, info,
                )
                @test info.status == :continue
            end
        end
    end
end
