#!/usr/bin/env julia

include(joinpath(@__DIR__, "solve_case.jl"))
exit(solve_case_main(vcat(["--solver", "cuclarabel"], ARGS)))
