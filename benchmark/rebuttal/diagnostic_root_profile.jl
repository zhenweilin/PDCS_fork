module DiagnosticRootProfile

using CUDA

export RootProfileRecord, profile_project!

struct RootProfileRecord
    branch_code::Int32
    interval_expansion_iterations::Int32
    newton_attempts::Int32
    newton_accepts::Int32
    bisection_iterations::Int32
    oracle_evaluations::Int32
    warm_start_attempted::Int32
    warm_start_accepted::Int32
    max_iter_reached::Int32
    reserved::Int32
    final_residual::Float64
end
sizeof(RootProfileRecord)==48 || error("RootProfileRecord ABI mismatch")

const MODULES=Dict{Symbol,CuModule}()
const FUNCTIONS=Dict{Tuple{Symbol,String},CuFunction}()
const THREADS=256

function functions(strategy)
    stem = strategy===:threadWise ? "massive_block_proj_profile" :
           strategy===:warpWise ? "sufficient_block_proj_profile" :
           strategy===:blockWise ? "moderate_block_proj_profile" :
           error("diagnostic strategy must be threadWise, warpWise, or blockWise")
    kernel=replace(stem,"_profile"=>"")
    mod=get!(MODULES,strategy) do
        path=normpath(joinpath(@__DIR__,"..","..","src","pdcs_gpu","cuda",stem*".ptx"))
        isfile(path) || error("profile PTX missing; run make rebuild-profile")
        CuModule(read(path))
    end
    init=get!(FUNCTIONS,(strategy,"init")) do
        CuFunction(mod,"pdcs_root_profile_initialize")
    end
    project=get!(FUNCTIONS,(strategy,"project")) do
        CuFunction(mod,kernel)
    end
    init,project
end

function profile_project!(strategy, vec, bl, bu, scale, scale2, scale_x, temp,
                          warm, starts, sizes, count::Int64, types;
                          abs_tol=1e-12, rel_tol=1e-12)
    init,kernel=functions(strategy)
    records=CuArray{RootProfileRecord}(undef,count)
    CUDA.cudacall(init,(CuPtr{RootProfileRecord},Int64),records,count;
                  threads=THREADS,blocks=cld(count,THREADS))
    blocks = strategy===:threadWise ? cld(count+THREADS,THREADS) :
             strategy===:warpWise ? cld((count+1)*32,THREADS) : count+1
    signature=(CuPtr{Float64},CuPtr{Float64},CuPtr{Float64},CuPtr{Float64},
      CuPtr{Float64},CuPtr{Float64},CuPtr{Float64},CuPtr{Float64},
      CuPtr{Int64},CuPtr{Int64},Int64,CuPtr{Int64},Float64,Float64)
    CUDA.cudacall(kernel,signature,vec,bl,bu,scale,scale2,scale_x,temp,warm,
                  starts,sizes,count,types,Float64(abs_tol),Float64(rel_tol);
                  threads=THREADS,blocks)
    CUDA.synchronize()
    Array(records)
end

end
