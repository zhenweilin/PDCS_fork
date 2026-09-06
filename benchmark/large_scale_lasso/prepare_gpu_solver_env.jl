#!/usr/bin/env julia

using Pkg
using TOML

const BENCHMARK_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(BENCHMARK_DIR, "..", ".."))
const ENV_DIR = joinpath(BENCHMARK_DIR, ".gpu_solver_env")
const SCS_ENV_DIR = joinpath(BENCHMARK_DIR, ".gpu_scs_env")
const SOURCE_DIR = joinpath(ENV_DIR, "cuclarabel-source")
const CUCLARABEL_URL = "https://github.com/oxfordcontrol/Clarabel.jl.git"
const CUCLARABEL_COMMIT = "ffa325c89fa90b7e86b745fa61b1dca64daf3a06"

ENV["PDCS_SKIP_GPU_PRECOMPILE"] = "1"
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

mkpath(ENV_DIR)
if !isdir(SOURCE_DIR)
    run(`git clone --filter=blob:none --single-branch --branch CuClarabel --no-checkout $CUCLARABEL_URL $SOURCE_DIR`)
    run(`git -C $SOURCE_DIR checkout --detach $CUCLARABEL_COMMIT`)
else
    isdir(joinpath(SOURCE_DIR, ".git")) ||
        error("existing CuClarabel source is not a Git checkout: $SOURCE_DIR")
    actual = readchomp(`git -C $SOURCE_DIR rev-parse HEAD`)
    actual == CUCLARABEL_COMMIT || error(
        "CuClarabel source is at $actual; expected $CUCLARABEL_COMMIT",
    )
end

# These dependencies belong only to CuClarabel's optional Python extension.
# Removing them matches the repository's previously validated GPU setup and
# does not alter any numerical solver source.
source_project_path = joinpath(SOURCE_DIR, "Project.toml")
source_project = TOML.parsefile(source_project_path)
for package in ("PythonCall", "OpenSSL_jll")
    pop!(source_project["deps"], package, nothing)
    pop!(source_project["compat"], package, nothing)
end
extensions = get(source_project, "extensions", Dict{String,Any}())
get(extensions, "PythonExt", nothing) == "PythonCall" &&
    pop!(extensions, "PythonExt")
open(source_project_path, "w") do stream
    TOML.print(stream, source_project; sorted = true)
end

Pkg.activate(ENV_DIR)
Pkg.develop(Pkg.PackageSpec(path = SOURCE_DIR))
Pkg.add([
    Pkg.PackageSpec(name = "CUDA", version = "5"),
    Pkg.PackageSpec(name = "CUDSS", version = "0.6"),
    Pkg.PackageSpec(name = "DataStructures"),
    Pkg.PackageSpec(name = "JuMP", version = "1"),
    Pkg.PackageSpec(name = "Match"),
    Pkg.PackageSpec(name = "MathOptInterface", version = "1"),
    Pkg.PackageSpec(name = "Polynomials"),
    Pkg.PackageSpec(name = "PythonCall"),
    Pkg.PackageSpec(name = "SCS", version = "2.6.4"),
    Pkg.PackageSpec(name = "SCS_GPU_jll", version = "300.200.1100"),
    Pkg.PackageSpec(name = "SnoopPrecompile"),
])
Pkg.instantiate()

preferences_path = joinpath(ENV_DIR, "LocalPreferences.toml")
preferences = isfile(preferences_path) ?
    TOML.parsefile(preferences_path) : Dict{String,Any}()
preferences["CUDA_Runtime_jll"] = Dict(
    "local" => "true",
    "version" => "12.6",
)
open(preferences_path, "w") do stream
    TOML.print(stream, preferences; sorted = true)
end

Pkg.precompile()

# SCS_GPU_jll 300.200.1100 publishes CUDA 11.4--11.8 artifacts only. Keep it
# in a separate project so cuClarabel can continue using its required CUDA 12
# cuDSS stack while cuSCS selects the CUDA 11.8 binary on the same H100 driver.
Pkg.activate(SCS_ENV_DIR)
Pkg.add([
    Pkg.PackageSpec(name = "CUDA_Runtime_jll"),
    Pkg.PackageSpec(name = "JuMP", version = "1"),
    Pkg.PackageSpec(name = "MathOptInterface", version = "1"),
    Pkg.PackageSpec(name = "SCS", version = "2.6.4"),
    Pkg.PackageSpec(name = "SCS_GPU_jll", version = "300.200.1100"),
])
scs_preferences_path = joinpath(SCS_ENV_DIR, "LocalPreferences.toml")
scs_preferences = isfile(scs_preferences_path) ?
    TOML.parsefile(scs_preferences_path) : Dict{String,Any}()
scs_preferences["CUDA_Runtime_jll"] = Dict(
    "local" => "false",
    "version" => "11.8",
)
open(scs_preferences_path, "w") do stream
    TOML.print(stream, scs_preferences; sorted = true)
end

# CUDA platform selection is a compile-time preference, so instantiate and
# precompile from a fresh Julia process after writing LocalPreferences.toml.
run(`$(Base.julia_cmd()) --startup-file=no --project=$SCS_ENV_DIR -e \
    'using Pkg; Pkg.instantiate(); Pkg.precompile()'`)
println(
    "LASSO_GPU_ENV_READY project=$ENV_DIR scs_project=$SCS_ENV_DIR " *
    "cuclarabel_commit=$CUCLARABEL_COMMIT",
)
