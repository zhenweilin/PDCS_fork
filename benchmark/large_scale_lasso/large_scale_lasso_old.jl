using Pkg
Pkg.activate("test_cpu_env")
using Arpack, LinearAlgebra
using JuMP
using Profile
using Random, SparseArrays
using SCS
using ArgParse


function single_cone_case_gen(m, n, density = 1e-3, lambda = 1e-3)
    # sparse matrix with 10% nonzeros
    rng = Random.MersenneTwister(1)
    A = SparseArrays.sprand(rng, m, n, density)
    println("m: ", m, " n: ", n, " density: ", density)
    # # set u diagonal to 1.0
    # count = 0
    # while count < ceil(m/n) - 1
    #     for i in 1:n
    #         A[count * n + i, i] = abs(rand(rng))
    #     end
    #     count += 1
    # end
    
    println(nnz(A))

    # x_feat = sprandn(rng, n, 0.5) ./ sqrt(n)
    x_feat = randn(rng, n)
    x_feat ./= sqrt(n)
    zero_indices = randperm(rng, n)[1:div(n,2)]
    x_feat[zero_indices] .= 0.0

    b = (A * x_feat) .+ 1e-6

    c_len = 1 + 1 + m + 2 * n
    c = rand(rng, c_len)
    c[1] = 0.0
    c[2] = 2.0
    c[2+1:2+m] .= 0.0
    lambda = norm(A' * b, Inf)
    println("lambda: ", lambda)
    c[2+m+1:end] .= lambda
    constraint_matrix = spzeros((1 + m, 1 + 1 + m + 2*n))
    constraint_matrix[1, 1] = 1.0
    constraint_matrix[2:end, 2+1:2+m] = I(m)
    constraint_matrix[2:end, 2+m+1:2+m+n] = A
    constraint_matrix[2:end, 2+m+n+1:end] = -A
    new_b = spzeros(1 + m)
    new_b[1] = 1.0
    new_b[2:end] .= b

    # model = Model(RPDHG_CLP_CPU.Optimizer)
    model = Model(SCS.Optimizer)
    @variable(model, x[1:c_len])
    @objective(model, Min, c' * x)
    @constraint(model, x[2+m+1:end] .>= 0.0)
    @constraint(model, constraint_matrix * x .== new_b)
    # @constraint(model, x[1:3+m] in MOI.RotatedSecondOrderCone(3+m))
    @variable(model, t)
    @variable(model, u)
    @constraint(model, t == (x[1] + x[2]) / sqrt(2))
    @constraint(model, u == (x[1] - x[2]) / sqrt(2))
    @constraint(model, [t; u; x[3:3+m]] in SecondOrderCone())
    # set_optimizer_attribute(model, "verbose", 2)
    # write_to_file(model, "./data/lasso_data/lasso_m_$(m)_n_$(n)_nnz_$(nnz(A))_lambda_$(lambda).cbf.gz")
    println("have written into data file")
    # optimize!(model)
    # return termination_status(model)
end

function main()
    s = ArgParse.ArgParseSettings()
    # ArgParse.@add_arg_table s begin
    #     "--m"
    #     help = "Number of constraints"
    #     default = 300
    #     arg_type = Int64

    #     "--n"
    #     help = "Number of variables"
    #     default = 1000
    #     arg_type = Int64

    #     "--density"
    #     help = "Density"
    #     default = 1e-3
    #     arg_type = Float64
    # end
    # args = ArgParse.parse_args(s)
    # @assert args["m"] > 0
    # @assert args["n"] > 0
    # @assert args["density"] > 0
    # @assert args["density"] <= 1
    # @assert args["m"] < args["n"]
    for m in 750000
        for n in 7500000
            density = 1e-4
            # for lambda in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
            single_cone_case_gen(m, n, density)
            # end
        end
    end
    for m in 100000:300000:700000
        for n in 1000000:3000000:7000000
            density = 1e-4
            # for lambda in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
            single_cone_case_gen(m, n, density)
            # end
        end
    end
end

main()