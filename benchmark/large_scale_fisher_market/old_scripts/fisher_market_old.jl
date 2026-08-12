# 导入必要的库
using Pkg
Pkg.activate("test_cpu_env")
include("../src/rpdhg_clp_cpu/RPDHG_CLP_CPU.jl")
using .RPDHG_CLP_CPU
using LinearAlgebra
using SparseArrays
using Printf
using Random
using MosekTools
using JuMP
using JLD2
using SCS
using ArgParse
using SparseArrays
using COPT
using Base.Threads


function fisher_market_old(m, n, density)
    rng = Random.MersenneTwister(1)
    w = rand(rng, m)
    u = SparseArrays.sprand(rng, m, n, density)
    u .= abs.(u)
    btilde = 0.25*m*ones(n)
    b = zeros(m + n)
    b[1:n] .= btilde
    repeat = 0
    while repeat < ceil(m/n) - 1
        for i in 1:n
            u[repeat * n + i, i] = abs(rand(rng))
        end
        repeat += 1
    end

    println(nnz(u))
    m, n = size(u)
    model = Model(RPDHG_CLP_CPU.Optimizer)
    total = m*n + 2m

    # 定义总变量
    @variable(model, x[i=1:total], lower_bound = i <= m*n ? 0 : -Inf)
    c = zeros(m*n + 2*m)
    @threads for i in 1:m
        c[m*n + 2*i - 1] = -w[i]
    end
    @objective(model, Min, c' * x)
    rows = vcat([1:n for _ in 1:m]...)  # Flatten all ranges into a single vector
    cols = vcat([(i - 1) * n + 1:(i - 1) * n + n for i in 1:m]...)  # Flatten all ranges
    values = ones(n * m)              # Non-zero values (all ones)
    
    # Construct the sparse matrix
    I_hcat_sparse = sparse(rows, cols, values, n, n * m)
    A = zeros(m + n, m*n + m + m)
    A[1:n, 1:m*n] = I_hcat_sparse
    @threads for i in 1:m
        A[i + n, (i - 1) * n + 1: i * n] .= vec(u[i, :]);
    end
    @threads for i in 1:m
        A[i + n, (m * n + 2 * i) ] = -1.0;
    end
    @constraint(model, A * x .== b)
    @constraint(model, con[i=1:m], [x[m*n + 2*i - 1], 1.0, x[m*n + 2*i]] in MOI.ExponentialCone())
    # optimize!(model)
    # println(termination_status(model))
    write_to_file(model, "./code/synthetic/fisher_market_data/fisher_market_m_$(m)_n_$(n)_density_$(density).cbf.gz")
    println("write to file, fisher_market_m_$(m)_n_$(n)_density_$(density).cbf.gz")
end


function fisher_market(m, n, count)
    time_start = time()
    rng = Random.MersenneTwister(1)
    w = rand(rng, m)   # 随机生成 m 个权重
    u = rand(rng, m * n)
    u .= abs.(u)
    if m < 1000
        u[rand(rng, m * n) .< 0.0] .= 0.0
    elseif m < 100000
        u[rand(rng, m * n) .< 0.5] .= 0.0
    else
        u[rand(rng, m * n) .< 0.8] .= 0.0
    end
    u = reshape(u, m, n)
    sparse_u = sparse(u)
    nnz_u = nnz(sparse_u)
    println(nnz_u)
    b = 0.25*m*ones(n)

    # 获取维度
    m, n = size(u)
    JLD2.save("./fisher_market_data/$(count)_fisher_market_m_$(m)_n_$(n)_nnz_$(nnz_u).jld2", "u", sparse_u, "w", w, "b", b)
end

function main()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--m"
        help = "Number of customers"
        default = 300
        arg_type = Int64

        "--n"
        help = "Number of goods"
        default = 100
        arg_type = Int64
    end
    args = ArgParse.parse_args(s)
    @assert args["m"] > 0 && args["n"] > 0 && args["m"] > args["n"]
    count = 0
    fisher_market(args["m"], args["n"], count)
    count += 1
    for i in 1:5:10
        fisher_market(i * 100, 5, count)
        count += 1
    end
    for i in 1:5:10
        fisher_market(i * 1000, 50, count)
        count += 1
    end
    for i in 1:5:10
        fisher_market(i * 10000, 50, count)
        count += 1
    end
    for i in 1:1:5
        fisher_market(i * 100000, 500, count)
        count += 1
    end
    # for i in 1:2
    #     fisher_market(i * 1000000, 500, count)
    #     count += 1
    # end
end
main()

