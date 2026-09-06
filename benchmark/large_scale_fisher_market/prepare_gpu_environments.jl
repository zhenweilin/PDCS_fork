#!/usr/bin/env julia

using Pkg
using TOML

const BENCHMARK_DIR = @__DIR__
const ENV_ROOT = joinpath(BENCHMARK_DIR, ".envs")
const CUPDCS_ENV = joinpath(ENV_ROOT, "cupdcs")
const SCS_ENV = joinpath(ENV_ROOT, "scs_gpu")
const CUCLARABEL_ENV = joinpath(ENV_ROOT, "cuclarabel")
const CUCLARABEL_SOURCE = joinpath(CUCLARABEL_ENV, "cuclarabel-source")
const CUCLARABEL_URL = "https://github.com/oxfordcontrol/Clarabel.jl.git"
const CUCLARABEL_COMMIT = "ffa325c89fa90b7e86b745fa61b1dca64daf3a06"

VERSION == v"1.10.4" || error("expected Julia 1.10.4, got $VERSION")
ENV["PDCS_SKIP_GPU_PRECOMPILE"] = "1"
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
mkpath(ENV_ROOT)

function write_cuda_preference(environment; local_runtime, version)
    path = joinpath(environment, "LocalPreferences.toml")
    preferences = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
    preferences["CUDA_Runtime_jll"] = Dict(
        "local" => local_runtime,
        "version" => version,
    )
    open(path, "w") do stream
        TOML.print(stream, preferences; sorted = true)
    end
end

function ensure_packages(environment, packages)
    Pkg.activate(environment)
    project_path = joinpath(environment, "Project.toml")
    direct_dependencies = isfile(project_path) ?
        get(TOML.parsefile(project_path), "deps", Dict{String,Any}()) :
        Dict{String,Any}()
    missing = [
        package for package in packages
        if !haskey(direct_dependencies, something(package.name, ""))
    ]
    isempty(missing) || Pkg.add(missing)
    Pkg.instantiate()
end

ensure_packages(CUPDCS_ENV, [
    Pkg.PackageSpec(name = "CUDA", version = "5"),
    Pkg.PackageSpec(name = "DataStructures"),
    Pkg.PackageSpec(name = "JuMP", version = "1"),
    Pkg.PackageSpec(name = "Match"),
    Pkg.PackageSpec(name = "MathOptInterface", version = "1"),
    Pkg.PackageSpec(name = "Polynomials"),
    Pkg.PackageSpec(name = "PythonCall"),
    Pkg.PackageSpec(name = "SnoopPrecompile"),
])
write_cuda_preference(CUPDCS_ENV; local_runtime = "true", version = "12.6")

ensure_packages(SCS_ENV, [
    Pkg.PackageSpec(name = "CUDA_Runtime_jll"),
    Pkg.PackageSpec(name = "SCS", version = "2.6.4"),
    Pkg.PackageSpec(name = "SCS_GPU_jll", version = "300.200.1100"),
])
write_cuda_preference(SCS_ENV; local_runtime = "false", version = "11.8")

mkpath(CUCLARABEL_ENV)
if !isdir(CUCLARABEL_SOURCE)
    run(`git clone --filter=blob:none --single-branch --branch CuClarabel --no-checkout $CUCLARABEL_URL $CUCLARABEL_SOURCE`)
    run(`git -C $CUCLARABEL_SOURCE checkout --detach $CUCLARABEL_COMMIT`)
else
    isdir(joinpath(CUCLARABEL_SOURCE, ".git")) ||
        error("existing CuClarabel source is not a Git checkout")
    actual = readchomp(`git -C $CUCLARABEL_SOURCE rev-parse HEAD`)
    actual == CUCLARABEL_COMMIT ||
        error("CuClarabel is at $actual; expected $CUCLARABEL_COMMIT")
end

# PythonCall belongs only to this fork's optional Python extension.  Removing
# it from the local checkout avoids Conda setup in noninteractive Slurm jobs;
# no numerical solver source is changed.
source_project_path = joinpath(CUCLARABEL_SOURCE, "Project.toml")
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

Pkg.activate(CUCLARABEL_ENV)
if !isfile(joinpath(CUCLARABEL_ENV, "Manifest.toml"))
    cd(CUCLARABEL_ENV) do
        Pkg.develop(Pkg.PackageSpec(path = "cuclarabel-source"))
        Pkg.add([
            Pkg.PackageSpec(name = "CUDA", version = "5"),
            Pkg.PackageSpec(name = "CUDSS", version = "0.6"),
        ])
    end
end
Pkg.instantiate()
write_cuda_preference(
    CUCLARABEL_ENV;
    local_runtime = "true",
    version = "12.6",
)

# Preferences affect CUDA platform selection during precompilation.  Run each
# environment in a fresh process after writing them.
for environment in (CUPDCS_ENV, SCS_ENV, CUCLARABEL_ENV)
    run(`$(Base.julia_cmd()) --startup-file=no --project=$environment -e \
        'using Pkg; Pkg.instantiate(); Pkg.precompile()'`)
end

println(
    "FISHER_GPU_ENVS_READY julia=$VERSION " *
    "cupdcs=$CUPDCS_ENV scs=$SCS_ENV cuclarabel=$CUCLARABEL_ENV " *
    "cuclarabel_commit=$CUCLARABEL_COMMIT",
)
