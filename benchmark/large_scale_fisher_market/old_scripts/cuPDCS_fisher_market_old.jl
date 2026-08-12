using Pkg
Pkg.activate("test_cpu_env")
include("../src/rpdhg_clp_gpu/RPDHG_CLP_GPU.jl")
using .RPDHG_CLP_GPU
using JuMP
using SCS
using CodecZlib
import MathOptInterface as MOI
import MathOptInterface.FileFormats: CBF
using CSV
using DataFrames
using FilePathsBase
import ArgParse
using CUDA
using JLD2
using SparseArrays
function write_to_csv(file_path::String, file_name::String, objective_value::Float64, solve_time::Float64)
    data = DataFrame(filename = [file_name], objective_value = [objective_value], solve_time = [solve_time], solver = ["PDHG_rescaling_restart_stepsize_resolving_aggressive"])
    CSV.write(file_path, data, append=true)
end

function read_and_solve_cbf(file_path::String, output_csv::String, logfile_name::String, kkt_restart_freq::Int64, duality_gap_restart_freq::Int64, use_kkt_restart::Bool, use_duality_gap_restart::Bool)
    # create a empty MathOptInterface
    println("file_path: ", file_path)
    # 从 CBF 文件中加载模型
    data = JLD2.load(file_path)
    u = data["u"]
    w = data["w"]
    btilde = data["b"]
    m, n = size(u)
    b = zeros(m + n + 3m)
    b[1:n] .= btilde
    for i in 1:m
        b[n + m + 3(i - 1) + 2] = -1.0
    end
    u = vec(u')
    sparse_u = sparse(u)
    nnz_u = nnz(sparse_u)
    println(nnz_u)
    row_indices = zeros(Int, nnz_u)
    for i = 1:nnz_u
        row_indices[i] = (sparse_u.nzind[i] - 1) ÷ n + 1 + n
    end

    c = zeros(m*n + 2*m)
    @views c[m*n+1:2:m*n+2*m-1] .= -w
    rows = repeat(1:n, m)
    cols = repeat((0:(m-1)) * n, inner=n) .+ repeat(1:n, m)
    values = ones(n * m)
    row2 = row_indices
    col2 = sparse_u.nzind
    val2 = sparse_u.nzval
    row = vcat([rows, row2]...)
    col = vcat([cols, col2]...)
    val = vcat([values, val2]...)
    row3 = (n+1) : (n+m)
    col3 = m * n .+ 2 .* (1:m)  # 直接计算每个值
    val3 = fill(-1.0, m)        # 创建一个长度为 m 的数组，值全为 -1.0

    row = vcat([row, row3]...)
    col = vcat([col, col3]...)
    val = vcat([val, val3]...)

    row4 = vcat([[3*i + 1, 3i+3] for i in 0:(m-1)]...) .+ (m + n)
    col4 = Vector{Integer}(1:2m)
    col4 = col4 .+ (m * n)
    val4 = fill(1.0, 2m)
    row = vcat([row, row4]...)
    col = vcat([col, col4]...)
    val = vcat([val, val4]...)
    A = sparse(row, col, val, m + n + 3m, m*n + m + m)
    mA, nA = size(A)
    bl = fill(0.0, nA)
    bl[m*n+1:m*n+2*m] .= -Inf
    bu = fill(Inf, nA)
    # expG = Vector{Integer}(ones(m) * 3)

    # 求解模型并记录内部求解时间
    start_time = time()
    time_limit = 3600.0 * 6
    sol_res = RPDHG_CLP_GPU.rpdhg_gpu_solve(
        n = nA,
        m = mA,
        nb = nA,
        c_cpu = c,
        G_cpu = A,
        h_cpu = b,
        mGzero = m + n,
        mGnonnegative = 0,
        socG = Vector{Integer}([]),
        rsocG = Vector{Integer}([]),
        expG = m,
        dual_expG = 0,
        bl_cpu = bl,
        bu_cpu = bu,
        soc_x = Vector{Integer}([]),
        rsoc_x = Vector{Integer}([]),
        exp_x = 0,
        dual_exp_x = 0,
        print_freq = 2000,
        use_preconditioner = true,
        rescaling_method = :ruiz_pock_chambolle,
        method = :average,
        time_limit = time_limit,
        use_adaptive_restart = true,
        use_adaptive_step_size_weight = true,
        use_resolving = true,
        use_accelerated = false,
        use_aggressive = true,
        verbose = 2,
        rel_tol = 1e-6,
        abs_tol = 1e-6,
        logfile_name = logfile_name,
        kkt_restart_freq = kkt_restart_freq,
        duality_gap_restart_freq = duality_gap_restart_freq,
        use_kkt_restart = use_kkt_restart,
        use_duality_gap_restart = use_duality_gap_restart,
    )
    # 获取求解时间
    solving_time = time() - start_time
    println("solving time: ", solving_time, " s")
    GC.gc()
    return solving_time
end

# 主函数入口，接受外部参数
function main()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--fileDir"
        help = "Problem file path"
        default = "./data"
        arg_type = String

        "--outputDir"
        help = "Output CSV file path"
        default = "./output"
        arg_type = String

        "--outputName"
        help = "Output CSV file path"
        default = "res.csv"
        arg_type = String

        "--kkt_restart_freq"
        help = "use kkt restart freq"
        default = 10^10
        arg_type = Int64

        "--duality_gap_restart_freq"
        help = "use duality gap restart freq"
        default = 10^10
        arg_type = Int64

        "--use_kkt_restart"
        help = "use kkt restart"
        default = true
        arg_type = Bool

        "--use_duality_gap_restart"
        help = "use duality gap restart"
        default = true
        arg_type = Bool
    end
    args = ArgParse.parse_args(s)
    # join the file path using the system's path separator
    output_csv = joinpath(args["outputDir"], args["outputName"])
    if !isdir(args["outputDir"])
        mkpath(args["outputDir"])
    end
    if !isfile(output_csv)
        # create an empty file
        touch(output_csv)
        # write header
        header = DataFrame(filename = String[], objective_value = Float64[], solve_time = Float64[], solver = String[])
        CSV.write(output_csv, header)
    end

    for data_file in readdir(args["fileDir"])
        if endswith(data_file, ".cbf")|| endswith(data_file, ".jld2")
            println("case begin summary ------------------------------------------------------------------------------------------------------")
            println("data file: ", data_file)
            file_path = joinpath(args["fileDir"], data_file)
            logfile_name = joinpath(args["outputDir"], basename(data_file) * ".log")
            println("file_path: ", file_path)
            println("output_csv: ", output_csv)
            println("logfile_name: ", logfile_name)
            solve_time = read_and_solve_cbf(file_path, output_csv, logfile_name, args["kkt_restart_freq"], args["duality_gap_restart_freq"], args["use_kkt_restart"], args["use_duality_gap_restart"])
            println("内部记录的求解时间: ", solve_time, " 秒")
            CUDA.memory_status()
            GC.gc()
            CUDA.reclaim()
            CUDA.memory_status()
            println("case end summary ------------------------------------------------------------------------------------------------------")
        end
    end
end

main()