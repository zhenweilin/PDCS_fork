#!/usr/bin/env julia

using Printf
using Random
using Statistics
using LinearAlgebra

const EPS64 = eps(Float64)
const LOGIT_HALLEY_THRESHOLD =
    parse(Float64, get(ENV, "PDCS_LOGIT_HALLEY_THRESHOLD", "0.0"))
const Z_NEWTON_STEPS = parse(Int, get(ENV, "PDCS_Z_NEWTON_STEPS", "16"))
const WARM_MODE_OVERRIDE = parse(Int, get(ENV, "PDCS_WARM_MODE", "0"))

function oracle(a, c, t, u, increasing)
    value = 0.0
    derivative = 0.0
    second = 0.0
    @inbounds for j in eachindex(a, c)
        a2 = a[j]^2
        cj = c[j]
        q = 1.0 + cj
        if increasing
            denominator = q - u
            denominator2 = denominator^2
            value += a2 * u^2 / denominator2
            derivative += 2.0 * a2 * u * q / (denominator2 * denominator)
            second += 2.0 * a2 * q * (q + 2.0 * u) / denominator2^2
        else
            one_minus_u = 1.0 - u
            denominator = 1.0 + cj * u
            denominator2 = denominator^2
            value += a2 * one_minus_u^2 / denominator2
            derivative -= 2.0 * a2 * one_minus_u * q /
                          (denominator2 * denominator)
            second += 2.0 * a2 * q *
                      (1.0 + 3.0 * cj - 2.0 * cj * u) / denominator2^2
        end
    end
    return value - t^2, derivative, second
end

function value_at(a, c, t, u, increasing)
    return first(oracle(a, c, t, u, increasing))
end

function endpoint_values(a, c, t, increasing)
    endpoint = increasing ? sum(abs2, a ./ c) - t^2 : sum(abs2, a) - t^2
    return increasing ? (-t^2, endpoint) : (endpoint, -t^2)
end

function reference_root(a, c, t, increasing)
    left = 0.0
    right = 1.0
    for _ in 1:256
        candidate = 0.5 * (left + right)
        candidate == left && return candidate
        candidate == right && return candidate
        f = value_at(a, c, t, candidate, increasing)
        if (increasing && f > 0.0) || (!increasing && f < 0.0)
            right = candidate
        else
            left = candidate
        end
    end
    return 0.5 * (left + right)
end

function projected_point(a, c, t, s, negative_branch)
    x = a ./ sqrt.(c)
    if negative_branch
        first = t * (1.0 - s) / s
        tail = x .* (1.0 - s) ./ (c .+ 1.0 .- s)
    else
        first = t / s
        tail = x ./ (1.0 .+ c .- c .* s)
    end
    return first, tail
end

function update_bracket(increasing, candidate, candidate_f,
                        left, right, left_f, right_f)
    update_right = (increasing && candidate_f > 0.0) ||
                   (!increasing && candidate_f < 0.0)
    return update_right ? (left, candidate, left_f, candidate_f) :
                          (candidate, right, candidate_f, right_f)
end

function valid_candidate(candidate, left, right, guard)
    return isfinite(candidate) && candidate > left + guard &&
           candidate < right - guard
end

function robust_candidate(u, f, h, h2, left, right, left_f, right_f)
    width = right - left
    width > 0.0 || return nothing
    guard = max(64.0 * EPS64, 1e-8 * width)

    # Halley in z=logit(u) preserves multiplicative resolution near both
    # endpoints.  The transformed derivatives cost no vector work because
    # F, F', and F'' are already available.
    clamped_u = clamp(u, 64.0 * EPS64, 1.0 - 64.0 * EPS64)
    coordinate_scale = clamped_u * (1.0 - clamped_u)
    derivative_logit = h * coordinate_scale
    second_logit = h2 * coordinate_scale^2 +
                   h * coordinate_scale * (1.0 - 2.0 * clamped_u)
    denominator_logit = 2.0 * derivative_logit^2 - f * second_logit
    logit_halley = nothing
    if isfinite(denominator_logit) && abs(denominator_logit) > 1e-300
        z = log(clamped_u / (1.0 - clamped_u))
        z_candidate = z - 2.0 * f * derivative_logit / denominator_logit
        candidate = z_candidate >= 36.0 ? 1.0 - 64.0 * EPS64 :
                    z_candidate <= -36.0 ? 64.0 * EPS64 :
                    inv(1.0 + exp(-z_candidate))
        if valid_candidate(candidate, left, right, guard)
            logit_halley = candidate
        end
    end

    denominator = 2.0 * h^2 - f * h2
    direct_halley = nothing
    if isfinite(denominator) && abs(denominator) > 1e-300
        candidate = u - 2.0 * f * h / denominator
        if valid_candidate(candidate, left, right, guard)
            direct_halley = candidate
        end
    end
    prefer_logit = min(clamped_u, 1.0 - clamped_u) <
                   LOGIT_HALLEY_THRESHOLD
    prefer_logit && !isnothing(logit_halley) && return logit_halley
    !isnothing(direct_halley) && return direct_halley
    !isnothing(logit_halley) && return logit_halley

    if isfinite(h) && abs(h) > 1e-18
        candidate = u - f / h
        valid_candidate(candidate, left, right, guard) && return candidate

        if isfinite(derivative_logit) && abs(derivative_logit) > 1e-18
            z_candidate = log(clamped_u / (1.0 - clamped_u)) -
                          f / derivative_logit
            candidate = z_candidate >= 36.0 ? 1.0 - 64.0 * EPS64 :
                        z_candidate <= -36.0 ? 64.0 * EPS64 :
                        inv(1.0 + exp(-z_candidate))
            valid_candidate(candidate, left, right, guard) && return candidate
        end
    end

    denominator = right_f - left_f
    if isfinite(denominator) && abs(denominator) > 1e-300
        candidate = left - left_f * width / denominator
        valid_candidate(candidate, left, right, guard) && return candidate
    end

    candidate = 0.5 * (left + right)
    return valid_candidate(candidate, left, right, 64.0 * EPS64) ?
           candidate : nothing
end

function coordinate_tolerance(u, rel_tol)
    return rel_tol * max(abs(u), 64.0 * EPS64)
end

function bracket_converged(left, right, rel_tol)
    midpoint = 0.5 * (left + right)
    return right - left <= coordinate_tolerance(midpoint, rel_tol)
end

function root_converged(f, h, u, left, right, t, abs_tol, rel_tol)
    bracket_converged(left, right, rel_tol) && return true
    residual_ok = abs(f) <= abs_tol * (1.0 + t^2)
    correction_ok = isfinite(h) && abs(h) > 1e-300 &&
                    abs(f / h) <= coordinate_tolerance(u, rel_tol)
    return residual_ok && correction_ok
end

function solve_root(a, c, t, increasing, warm_u;
                    robust=false, illinois=true,
                    abs_tol=1e-12, rel_tol=1e-12, max_newton=8)
    left = 0.0
    right = 1.0
    left_f, right_f = endpoint_values(a, c, t, increasing)
    valid_warm = isfinite(warm_u) && 0.0 < warm_u < 1.0
    u = valid_warm ? warm_u : 0.5
    f, h, h2 = oracle(a, c, t, u, increasing)
    evaluations = 1
    if valid_warm && root_converged(
            f, h, u, left, right, t, abs_tol, rel_tol)
        return (; u, evaluations, fallback=false)
    end
    left, right, left_f, right_f = update_bracket(
        increasing, u, f, left, right, left_f, right_f)

    for _ in 1:max_newton
        root_converged(f, h, u, left, right, t, abs_tol, rel_tol) &&
            return (; u, evaluations, fallback=false)
        candidate = if robust
            robust_candidate(u, f, h, h2, left, right, left_f, right_f)
        else
            guard = max(64.0 * EPS64, 1e-8 * (right - left))
            trial = isfinite(h) && abs(h) > 1e-18 ? u - f / h : NaN
            valid_candidate(trial, left, right, guard) ? trial : nothing
        end
        isnothing(candidate) && break

        candidate_f, candidate_h, candidate_h2 =
            oracle(a, c, t, candidate, increasing)
        evaluations += 1
        left, right, left_f, right_f = update_bracket(
            increasing, candidate, candidate_f,
            left, right, left_f, right_f)
        isfinite(candidate_f) || break
        if robust && root_converged(candidate_f, candidate_h, candidate,
                                    left, right, t, abs_tol, rel_tol)
            return (; u=candidate, evaluations, fallback=false)
        end
        !robust && abs(candidate_f) >= abs(f) && break
        u, f, h, h2 = candidate, candidate_f, candidate_h, candidate_h2
        root_converged(f, h, u, left, right, t, abs_tol, rel_tol) &&
            return (; u, evaluations, fallback=false)
    end

    last_updated_side = 0
    for _ in 1:256
        bracket_converged(left, right, rel_tol) && break
        if illinois
            denominator = right_f - left_f
            candidate = left - left_f * (right - left) / denominator
            guard = max(64.0 * EPS64, 1e-6 * (right - left))
            if !valid_candidate(candidate, left, right, guard)
                candidate = 0.5 * (left + right)
                last_updated_side = 0
            end
        else
            candidate = 0.5 * (left + right)
        end
        candidate_f = value_at(a, c, t, candidate, increasing)
        evaluations += 1
        !isfinite(candidate_f) && break
        update_right = (increasing && candidate_f > 0.0) ||
                       (!increasing && candidate_f < 0.0)
        if update_right
            right, right_f = candidate, candidate_f
            if illinois && last_updated_side == 1
                left_f *= 0.5
            end
            last_updated_side = 1
        else
            left, left_f = candidate, candidate_f
            if illinois && last_updated_side == -1
                right_f *= 0.5
            end
            last_updated_side = -1
        end
    end
    return (; u=0.5 * (left + right), evaluations, fallback=true)
end

function logistic_pair(z)
    if z >= 0.0
        exponential = exp(-z)
        denominator = 1.0 + exponential
        return inv(denominator), exponential / denominator
    else
        exponential = exp(z)
        denominator = 1.0 + exponential
        return exponential / denominator, inv(denominator)
    end
end

function oracle_z(a, c, t, z, negative_branch)
    s, complement = logistic_pair(z)
    value = 0.0
    derivative = 0.0
    second = 0.0
    @inbounds for j in eachindex(a, c)
        a2 = a[j]^2
        cj = c[j]
        q = 1.0 + cj
        alpha = negative_branch ? cj : 1.0
        beta = negative_branch ? 1.0 : cj
        denominator = alpha + beta * complement
        denominator2 = denominator^2
        term = a2 * s^2 / denominator2
        term_derivative = 2.0 * a2 * s^2 * complement * q /
                          (denominator2 * denominator)
        value += term
        derivative += term_derivative
        second += term_derivative *
                  (2.0 * complement - s +
                   3.0 * beta * s * complement / denominator)
    end
    return value - t^2, derivative, second
end

function z_endpoint_values(a, c, t, negative_branch)
    endpoint = negative_branch ? sum(abs2, a ./ c) - t^2 :
                                 sum(abs2, a) - t^2
    return -t^2, endpoint
end

function z_converged(f, h, z, left, right, t, abs_tol, rel_tol)
    right - left <= rel_tol && return true
    residual_ok = abs(f) <= abs_tol * (1.0 + t^2)
    correction_ok = isfinite(h) && abs(h) > 1e-300 &&
                    abs(f / h) <= rel_tol
    return residual_ok && correction_ok
end

function z_candidate(z, f, h, h2, left, right, left_f, right_f,
                     robust)
    width = right - left
    guard = max(64.0 * EPS64, 1e-8 * width)
    if robust
        denominator = 2.0 * h^2 - f * h2
        if isfinite(denominator) && abs(denominator) > 1e-300
            candidate = z - 2.0 * f * h / denominator
            valid_candidate(candidate, left, right, guard) && return candidate
        end
    end
    if isfinite(h) && abs(h) > 1e-300
        candidate = z - f / h
        valid_candidate(candidate, left, right, guard) && return candidate
    end
    if robust
        # Exponent-space expansion: z is already a logarithmic scale, so
        # doubling |z| locates a multiplicative root scale in O(log log).
        direction = f < 0.0 ? 1.0 : -1.0
        expansion_step = max(1.0, abs(z))
        candidate = z + direction * expansion_step
        valid_candidate(candidate, left, right, guard) && return candidate

        denominator = right_f - left_f
        if isfinite(denominator) && abs(denominator) > 1e-300
            candidate = left - left_f * width / denominator
            valid_candidate(candidate, left, right, guard) && return candidate
        end
        candidate = 0.5 * (left + right)
        valid_candidate(candidate, left, right, 64.0 * EPS64) &&
            return candidate
    end
    return nothing
end

function solve_z(a, c, t, negative_branch, warm_z;
                 robust=false, illinois=true,
                 abs_tol=1e-12, rel_tol=1e-12,
                 max_newton=Z_NEWTON_STEPS)
    left = -700.0
    right = 700.0
    left_f, right_f = z_endpoint_values(a, c, t, negative_branch)
    z = isfinite(warm_z) && left < warm_z < right ? warm_z : 0.0
    f, h, h2 = oracle_z(a, c, t, z, negative_branch)
    evaluations = 1
    if isfinite(warm_z) && z_converged(
            f, h, z, left, right, t, abs_tol, rel_tol)
        return (; z, evaluations, fallback=false)
    end
    if f > 0.0
        right, right_f = z, f
    else
        left, left_f = z, f
    end

    for _ in 1:max_newton
        z_converged(f, h, z, left, right, t, abs_tol, rel_tol) &&
            return (; z, evaluations, fallback=false)
        candidate = z_candidate(
            z, f, h, h2, left, right, left_f, right_f, robust)
        isnothing(candidate) && break
        candidate_f, candidate_h, candidate_h2 =
            oracle_z(a, c, t, candidate, negative_branch)
        evaluations += 1
        if candidate_f > 0.0
            right, right_f = candidate, candidate_f
        else
            left, left_f = candidate, candidate_f
        end
        isfinite(candidate_f) || break
        !robust && abs(candidate_f) >= abs(f) && break
        z, f, h, h2 = candidate, candidate_f, candidate_h, candidate_h2
        z_converged(f, h, z, left, right, t, abs_tol, rel_tol) &&
            return (; z, evaluations, fallback=false)
    end
    z_converged(f, h, z, left, right, t, abs_tol, rel_tol) &&
        return (; z, evaluations, fallback=false)

    last_updated_side = 0
    for _ in 1:256
        right - left <= rel_tol && break
        if illinois
            denominator = right_f - left_f
            candidate = left - left_f * (right - left) / denominator
            guard = max(64.0 * EPS64, 1e-6 * (right - left))
            if !valid_candidate(candidate, left, right, guard)
                candidate = 0.5 * (left + right)
                last_updated_side = 0
            end
        else
            candidate = 0.5 * (left + right)
        end
        candidate_f = first(oracle_z(a, c, t, candidate, negative_branch))
        evaluations += 1
        isfinite(candidate_f) || break
        if candidate_f > 0.0
            right, right_f = candidate, candidate_f
            if illinois && last_updated_side == 1
                left_f *= 0.5
            end
            last_updated_side = 1
        else
            left, left_f = candidate, candidate_f
            if illinois && last_updated_side == -1
                right_f *= 0.5
            end
            last_updated_side = -1
        end
    end
    return (; z=0.5 * (left + right), evaluations, fallback=true)
end

function reference_z(a, c, t, negative_branch)
    left = -700.0
    right = 700.0
    for _ in 1:256
        candidate = 0.5 * (left + right)
        candidate == left && return candidate
        candidate == right && return candidate
        f = first(oracle_z(a, c, t, candidate, negative_branch))
        if f > 0.0
            right = candidate
        else
            left = candidate
        end
    end
    return 0.5 * (left + right)
end

function projected_point_z(a, c, t, z, negative_branch)
    s, complement = logistic_pair(z)
    x = a ./ sqrt.(c)
    if negative_branch
        first = t * complement / s
        tail = x .* complement ./ (c .+ complement)
    else
        first = t / s
        tail = x ./ (1.0 .+ c .* complement)
    end
    return first, tail
end

function warm_start_z(rng, root_z, mode)
    if mode == 1
        return NaN
    elseif mode == 2
        return root_z
    elseif mode == 3
        return root_z + 0.02 * randn(rng)
    elseif mode == 4
        return root_z + 0.5 * randn(rng)
    elseif mode == 5
        return root_z + 4.0 * randn(rng)
    else
        return -root_z
    end
end

function warm_start(rng, root, mode)
    z = log(root / (1.0 - root))
    if mode == 1
        return NaN
    elseif mode == 2
        return root
    elseif mode == 3
        return inv(1.0 + exp(-(z + 0.02 * randn(rng))))
    elseif mode == 4
        return inv(1.0 + exp(-(z + 0.5 * randn(rng))))
    elseif mode == 5
        return inv(1.0 + exp(-(z + 4.0 * randn(rng))))
    else
        return 1.0 - root
    end
end

function quantile_sorted(values, p)
    ordered = sort(values)
    return ordered[clamp(ceil(Int, p * length(ordered)), 1, length(ordered))]
end

function main()
    sample_count = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
    seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20260819
    rng = MersenneTwister(seed)
    dimensions = (1, 2, 3, 8, 32, 201, 1024)
    baseline_evaluations = Int[]
    robust_evaluations = Int[]
    baseline_fallbacks = 0
    robust_fallbacks = 0
    failures = 0
    max_root_error = 0.0
    max_scaled_residual = 0.0
    max_projection_error = 0.0

    for sample in 1:sample_count
        dimension = rand(rng, dimensions)
        log_c = rand(rng, dimension) .* 24.0 .- 12.0
        c = 10.0 .^ log_c
        a = randn(rng, dimension) .* (10.0 .^ (rand(rng, dimension) .* 8.0 .- 4.0))
        negative_branch = isodd(sample)
        root_z = rand(rng) * 60.0 - 30.0
        root_s, root_complement = logistic_pair(root_z)
        ratios = negative_branch ?
            root_s ./ (c .+ root_complement) :
            root_s ./ (1.0 .+ c .* root_complement)
        t = sqrt(sum(abs2, a .* ratios))
        warm_mode = WARM_MODE_OVERRIDE == 0 ? mod1(sample, 6) :
                                             WARM_MODE_OVERRIDE
        warm = warm_start_z(rng, root_z, warm_mode)
        illinois = isodd(div(sample - 1, 6))

        baseline = solve_z(a, c, t, negative_branch, warm;
                           robust=false, illinois=illinois)
        candidate = solve_z(a, c, t, negative_branch, warm;
                            robust=true, illinois=illinois)
        push!(baseline_evaluations, baseline.evaluations)
        push!(robust_evaluations, candidate.evaluations)
        baseline_fallbacks += baseline.fallback
        robust_fallbacks += candidate.fallback

        f = first(oracle_z(a, c, t, candidate.z, negative_branch))
        scaled_residual = abs(f) / (1.0 + t^2)
        root_error = abs(candidate.z - root_z)
        reference_coordinate = reference_z(a, c, t, negative_branch)
        candidate_first, candidate_tail =
            projected_point_z(a, c, t, candidate.z, negative_branch)
        reference_first, reference_tail =
            projected_point_z(a, c, t, reference_coordinate, negative_branch)
        output_scale = max(1.0, abs(reference_first),
                           maximum(abs, reference_tail; init=0.0))
        projection_error = max(abs(candidate_first - reference_first),
                               maximum(abs, candidate_tail .- reference_tail;
                                       init=0.0)) / output_scale
        max_root_error = max(max_root_error, root_error)
        max_scaled_residual = max(max_scaled_residual, scaled_residual)
        max_projection_error = max(max_projection_error, projection_error)
        if !(isfinite(candidate.z) && -700.0 < candidate.z < 700.0 &&
             projection_error <= 5e-8)
            failures += 1
            failures <= 5 && @warn "candidate validation failure" sample negative_branch root_z reference_coordinate candidate warm scaled_residual projection_error
        end
    end

    ratio = mean(baseline_evaluations) / mean(robust_evaluations)
    wins = count(pair -> pair[1] < pair[2],
                 zip(robust_evaluations, baseline_evaluations))
    ties = count(pair -> pair[1] == pair[2],
                 zip(robust_evaluations, baseline_evaluations))
    @printf("samples=%d seed=%d failures=%d\n", sample_count, seed, failures)
    @printf("baseline mean=%.4f median=%d p95=%d fallback=%.3f%%\n",
            mean(baseline_evaluations), quantile_sorted(baseline_evaluations, 0.5),
            quantile_sorted(baseline_evaluations, 0.95),
            100 * baseline_fallbacks / sample_count)
    @printf("robust   mean=%.4f median=%d p95=%d fallback=%.3f%%\n",
            mean(robust_evaluations), quantile_sorted(robust_evaluations, 0.5),
            quantile_sorted(robust_evaluations, 0.95),
            100 * robust_fallbacks / sample_count)
    @printf("oracle reduction speedup=%.4fx wins=%d ties=%d losses=%d\n",
            ratio, wins, ties, sample_count - wins - ties)
    @printf("max logit-root error=%.3e max scaled residual=%.3e max projection error=%.3e\n",
            max_root_error, max_scaled_residual, max_projection_error)
    failures == 0 || error("bounded SOC root validation failed")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
