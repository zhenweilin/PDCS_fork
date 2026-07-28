#!/usr/bin/env julia

using Pkg

"""Install the external comparison solvers in the active project.

SCS is registered.  CuClarabel is intentionally installed from its official
repository because it may not be available in every General registry snapshot.
The script is separate from benchmark execution so package download and
precompilation never contaminate solver timing.
"""
Pkg.add("SCS")
Pkg.add(url="https://github.com/oxfordcontrol/CuClarabel.jl.git")
Pkg.instantiate()
Pkg.precompile()
println("LASSO_SOLVER_INSTALL_COMPLETE")
