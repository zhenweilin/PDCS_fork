using Test
using SparseArrays
using JuMP
using PDCS: PDCS_CPU

include(joinpath(@__DIR__, "run_penalty_sweep.jl"))

@testset "PDCS bulk Lasso handoff" begin
    A = SparseMatrixCSC{Float32, Int32}(sparse(
        [1, 2, 3, 1, 2, 3],
        [1, 1, 2, 3, 4, 4],
        Float32[2, 1, 3, 1, -1, 2],
        3,
        4,
    ))
    data = LassoData("tiny", A, Float32[1, -1, -1], 3.0, 4, nothing)
    conic = build_lasso_conic_data(data; penalty_ratio = 0.1, workers = 1)
    model = materialize_bulk_model(conic, PDCS_CPU.Optimizer)

    @test JuMP.num_variables(model) == conic.num_variables
    cache = model.ext[:PDCS_optimizer_cache]
    coefficients = cache.model.constraints.coefficients
    @test coefficients.colptr === conic.colptr
    @test coefficients.rowval === conic.rowval
    @test coefficients.nzval === conic.nzval

    set_time_limit_sec(model, 2.0)
    set_optimizer_attribute(model, "verbose", 0)
    optimize!(model)
    @test termination_status(model) in (JuMP.MOI.OPTIMAL, JuMP.MOI.TIME_LIMIT)

    @test set_penalty!(model, conic, 1.5) == 1.5
    optimize!(model)
    @test termination_status(model) in (JuMP.MOI.OPTIMAL, JuMP.MOI.TIME_LIMIT)
end
