module ValidateProjection

using LinearAlgebra
using PDCS: PDCS_CPU

export project_soc!, validate_soc, validate_exp_diagonal, finite_error

function project_soc!(z)
    t = z[1]; u = @view z[2:end]; n = norm(u)
    if n <= t
        return z
    elseif n <= -t
        fill!(z, 0)
    else
        a = (n+t)/(2n)
        z[1] = (n+t)/2
        u .*= a
    end
    z
end

finite_error(actual, expected) =
    all(isfinite, actual) && all(isfinite, expected) ?
    maximum(abs.(actual .- expected); init=0.0) : Inf

function validate_soc(input, output, count, dimension; tolerance=5e-8)
    error = 0.0; feasible = true
    for cone in 1:count
        r = (cone-1)*dimension+1:cone*dimension
        expected = project_soc!(copy(@view input[r]))
        actual = @view output[r]
        error = max(error, finite_error(actual, expected))
        feasible &= norm(@view(actual[2:end])) <= actual[1] + tolerance
    end
    (; max_error=error, finite=all(isfinite, output), feasible,
       status=error <= tolerance && feasible ? "PASS" : "FAIL")
end

function validate_exp_diagonal(input, diagonal, output, count; tolerance=5e-8)
    error = 0.0
    for cone in 1:count
        r = 3cone-2:3cone
        expected = copy(@view input[r])
        PDCS_CPU.exponent_proj_diagonal!(expected, @view diagonal[r])
        error = max(error, finite_error(@view(output[r]), expected))
    end
    (; max_error=error, finite=all(isfinite, output),
       status=error <= tolerance ? "PASS" : "FAIL")
end

end
