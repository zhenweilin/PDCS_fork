#!/usr/bin/env julia

"""
Install the GPU `CuClarabel` branch in an isolated Julia project.

The branch exports the module `Clarabel`, just like the registered CPU
package, so installing both in the main PDCS project would replace one with
the other.  This installer instead uses `benchmark/cuclarabel_env`.

At the pinned upstream commit, PythonCall and OpenSSL_jll are direct
dependencies solely for the optional Python extension.  OpenSSL_jll ~3.0 is
not resolvable on Julia 1.12.  The Lasso/JuMP GPU solver does not load that
extension, so this installer removes only those two optional metadata entries
from its private source checkout.  No numerical CuClarabel source is changed.
"""

using Pkg
using TOML

const ENV_DIR = joinpath(@__DIR__, "cuclarabel_env")
const SOURCE_DIR = joinpath(ENV_DIR, ".cuclarabel-source")
const UPSTREAM_URL = "https://github.com/oxfordcontrol/Clarabel.jl.git"
const UPSTREAM_COMMIT = "ffa325c89fa90b7e86b745fa61b1dca64daf3a06"

mkpath(ENV_DIR)
if !isdir(SOURCE_DIR)
    # A blob-filtered checkout avoids storing the full upstream Git history;
    # only files needed by the pinned numerical source are materialized.
    run(`git clone --filter=blob:none --single-branch --branch CuClarabel --no-checkout $UPSTREAM_URL $SOURCE_DIR`)
    run(`git -C $SOURCE_DIR checkout --detach $UPSTREAM_COMMIT`)
else
    isdir(joinpath(SOURCE_DIR, ".git")) ||
        error("existing CuClarabel source is not a Git checkout: $SOURCE_DIR")
    actual = readchomp(`git -C $SOURCE_DIR rev-parse HEAD`)
    actual == UPSTREAM_COMMIT || error(
        "refusing to modify existing CuClarabel checkout at commit $actual; " *
        "expected $UPSTREAM_COMMIT",
    )
end

project_path = joinpath(SOURCE_DIR, "Project.toml")
project = TOML.parsefile(project_path)
for package in ("PythonCall", "OpenSSL_jll")
    pop!(project["deps"], package, nothing)
    pop!(project["compat"], package, nothing)
end
if get(get(project, "extensions", Dict{String,Any}()), "PythonExt", nothing) ==
   "PythonCall"
    pop!(project["extensions"], "PythonExt")
end
open(project_path, "w") do io
    TOML.print(io, project; sorted=true)
end

Pkg.activate(ENV_DIR)
Pkg.develop(Pkg.PackageSpec(path=SOURCE_DIR))
Pkg.add([
    Pkg.PackageSpec(name="JuMP", version="1"),
    Pkg.PackageSpec(name="CUDA", version="5"),
    Pkg.PackageSpec(name="CUDSS", version="0.6"),
    Pkg.PackageSpec(name="MathOptInterface", version="1"),
])
Pkg.instantiate()
Pkg.precompile()

println(
    "CUCLARABEL_INSTALL_COMPLETE project=$ENV_DIR source=$SOURCE_DIR " *
    "upstream_commit=$UPSTREAM_COMMIT",
)
