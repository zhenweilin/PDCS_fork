#!/usr/bin/env julia

using Pkg

# Package installation does not require running PDCS's optional GPU warm-up
# solve. Skipping it keeps installation independent of GPU availability and
# avoids calling native grid-wise symbols while Julia is generating cache
# files (before module runtime initialization).
ENV["PDCS_SKIP_GPU_PRECOMPILE"] = "1"

# Install the external comparison solvers in the active project.
#
# SCS and CPU Clarabel are registered packages.  The CuClarabel branch exports
# the same `Clarabel` module and must not be installed into this project.
# Use `benchmark/install_cuclarabel_gpu.jl` for its isolated GPU environment.
# Installation is separate from benchmark execution so downloads and
# precompilation never contaminate solver timing.
Pkg.add("SCS")
Pkg.add("Clarabel")
Pkg.instantiate()
Pkg.precompile()
println("LASSO_SOLVER_INSTALL_COMPLETE")
