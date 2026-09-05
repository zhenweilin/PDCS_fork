
using Pkg
Pkg.activate("test_cpu_env")
using JuMP
using SCS
using MosekTools
using Mosek
using CodecZlib
import MathOptInterface as MOI
import MathOptInterface.FileFormats: CBF
using CSV
using DataFrames
using FilePathsBase
import ArgParse
using JLD2
using SparseArrays

function write_to_csv(file_path::String, file_name::String, objective_value::Float64, solve_time::Float64)
    data = DataFrame(filename = [file_name], objective_value = [objective_value], solve_time = [solve_time], solver = ["PDHG_rescaling_restart_stepsize_resolving_aggressive"])
    CSV.write(file_path, data, append=true)
end

function read_and_solve_cbf(file_path::String, output_csv::String, logfile_name::String, kkt_restart_freq::Int64, duality_gap_restart_freq::Int64, use_kkt_restart::Bool, use_duality_gap_restart::Bool)
    println("file_path: ", file_path)
    # 从 CBF 文件中加载模型
    data = JLD2.load(file_path)
    u = data["u"]
    w = data["w"]
    m,n = size(u)
    btilde = data["b"]

    b = zeros(m + n + 3m + m * n)
    b[1:n] .= btilde
    for i in 1:m
        b[n + m + m * n + 3(i - 1) + 2 ] = -1.0
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
    row_eq = copy(row)
    col_eq = copy(col)
    val_eq = copy(val)
    # before are equality constraints

    # insert a identity matrix
    rowI = (m + n + 1) : (m + n + m * n)
    colI = (1) : (m * n)
    valI = ones(m * n)
    row = vcat([row, rowI]...)
    col = vcat([col, colI]...)
    val = vcat([val, valI]...)

    row4 = vcat([[3*i + 1, 3i+3] for i in 0:(m-1)]...) .+ (m + n)
    row4 = row4 .+ (m * n)
    col4 = Vector{Integer}(1:2m)
    col4 = col4 .+ (m * n)
    val4 = fill(1.0, 2m)
    row = vcat([row, row4]...)
    col = vcat([col, col4]...)
    val = vcat([val, val4]...)
    A = sparse(row, col, val, m + n + 3m + m * n, m*n + m + m)
    mA, nA = size(A)

    row_exp = row4 .- ((m * n) + m + n)
    col_exp = col4
    val_exp = val4
    for i in 1:m
        base_idx = 2*(i-1)
        row_exp[base_idx + 1], row_exp[base_idx + 2] = row_exp[base_idx + 2], row_exp[base_idx + 1]
        base_idx = 3*(i-1)
        base_idx += m + n + m * n
        b[base_idx + 1], b[base_idx + 3] = b[base_idx + 3], b[base_idx + 1]
    end
    maketask() do task
        putstreamfunc(task,MSK_STREAM_LOG,msg -> print(msg))
        # linkfiletostream(task,MSK_STREAM_LOG,"mosek.log",2);
        putintparam(task, MSK_IPAR_PRESOLVE_USE, 0)
        putdouparam(task, MSK_DPAR_OPTIMIZER_MAX_TIME, 3600.0 * 2);
        putdouparam(task, MSK_DPAR_INTPNT_TOL_PFEAS, 1e-3)
        putdouparam(task, MSK_DPAR_INTPNT_TOL_DFEAS, 1e-3)
        putdouparam(task, MSK_DPAR_INTPNT_TOL_REL_GAP, 1e-3)
        numvar = nA
        numcon = m + n
        @assert numcon == mA - m * n - 3m
        appendcons(task,numcon)
        appendvars(task,numvar)
        
        # 设置目标函数和变量边界
        putobjsense(task,MSK_OBJECTIVE_SENSE_MINIMIZE)
        putcslice(task,1, numvar+1, c)
        putvarboundsliceconst(task, 1, m * n + 1, MSK_BK_LO, 0.0, Inf)
        putvarboundsliceconst(task, m * n + 1, numvar + 1, MSK_BK_FR, -Inf, Inf) # give bound even it is free
        putaijlist(task, row_eq, col_eq, val_eq)
        for i = 1:(m + n)
            putconbound(task, i, MSK_BK_FX, b[i], b[i])
        end
        
        
        appendafes(task, 3*m)
        putafefentrylist(task, row_exp, col_exp, val_exp)
        for i = 1:m
            base_idx = 3*(i-1)
            putafeg(task, base_idx + 1, -b[m+n+m*n+3*(i-1)+1])
            putafeg(task, base_idx + 2, -b[m+n+m*n+3*(i-1)+2])
            putafeg(task, base_idx + 3, -b[m+n+m*n+3*(i-1)+3])
        end

        for i = 1:m
            expdomain = appendprimalexpconedomain(task)
            base_idx = 3*(i-1)
            appendacc(task,
                      expdomain,               # Domain
                      [base_idx + 1, base_idx + 2, base_idx + 3],               # Rows from F
                      nothing)                 # Unused
        end
        
        # 优化求解
        optimize(task)
        solutionsummary(task,MSK_STREAM_MSG)
        prosta = getprosta(task,MSK_SOL_ITR)
        solsta = getsolsta(task,MSK_SOL_ITR)
    
        # Output a solution
        xx = getxx(task,MSK_SOL_ITR)
    
        if solsta == MSK_SOL_STA_OPTIMAL
            # println("Optimal solution: $xx")
        elseif solsta == MSK_SOL_STA_DUAL_INFEAS_CER
            println("Primal or dual infeasibility.")
        elseif solsta == MSK_SOL_STA_PRIM_INFEAS_CER
            println("Primal or dual infeasibility.")
        elseif solsta == MSK_SOL_STA_UNKNOWN
            println("Unknown solution status")
        else
            println("Other solution status")
        end
    end
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
            try
                println("case begin summary ------------------------------------------------------------------------------------------------------")
                println("data file: ", data_file)
                file_path = joinpath(args["fileDir"], data_file)
                logfile_name = joinpath(args["outputDir"], basename(data_file) * ".log")
                println("file_path: ", file_path)
                println("output_csv: ", output_csv)
                println("logfile_name: ", logfile_name)
                solve_time = read_and_solve_cbf(file_path, output_csv, logfile_name, args["kkt_restart_freq"], args["duality_gap_restart_freq"], args["use_kkt_restart"], args["use_duality_gap_restart"])
            catch
                println("error in solving the problem")
            end
            println("内部记录的求解时间: ", solve_time, " 秒")
            println("case end summary ------------------------------------------------------------------------------------------------------")
        end
    end
end

main()