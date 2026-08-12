using Pkg
Pkg.activate("pdcs_env")
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.add("CUDA")
Pkg.resolve()
using CUDA
using PDCS: PDCS_CPU, PDCS_GPU
