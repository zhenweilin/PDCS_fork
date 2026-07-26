using Test

mktempdir() do root
    raw=joinpath(root,"raw")
    ledger_root=joinpath(root,"ledger")
    mkpath(raw)
    mkpath(joinpath(ledger_root,"iteration","seed2026"))
    gpu=joinpath(raw,"iteration_grouped_threadWise_seed2026.gpu.csv")
    open(gpu,"w") do io
        for i in 1:40
            # fields: timestamp,index,uuid,gpu%,memory%,memory.used,power,
            # clocks.sm,clocks.mem,temperature
            println(io,"2026/07/26 12:00:$(lpad(i,2,'0')), 0, GPU-test, $(i), 10, 1000, 200, 1500, 1200, 60")
        end
    end
    ledger=joinpath(ledger_root,"iteration","seed2026",
      "duration_iteration_grouped_threadWise_seed2026_ledger.csv")
    write(ledger,"experiment,layout,strategy,seed\niteration,grouped,threadWise,2026\n")
    output=joinpath(root,"summary.csv")
    script=joinpath(@__DIR__,"..","..","benchmark",
      "summarize_soc_divergence_utilization.jl")
    command=`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..","..")) $script --root $raw --ledger-root $ledger_root --output $output`
    run(setenv(command,"JULIA_DEPOT_PATH"=>get(ENV,"JULIA_DEPOT_PATH","")))
    lines=readlines(output)
    @test length(lines)==2
    fields=split(lines[2],',')
    @test parse(Int,fields[6])==30
    @test parse(Float64,fields[7])≈20.5
    @test fields[end]==ledger
end
