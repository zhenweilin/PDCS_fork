#pragma once

// Candidate selection for the bounded SOC root u in (0, 1).  The caller owns
// the strict bracket and evaluates exactly one selected point.  This helper is
// deliberately scalar: Halley, ordinary Newton, logit-Newton, Illinois, and
// midpoint candidates are tried without another vector traversal.

__host__ __device__ __forceinline__ bool pdcs_bounded_soc_valid_candidate(
    double candidate, double left, double right, double guard) {
  return isfinite(candidate) && candidate > left + guard &&
         candidate < right - guard;
}

__host__ __device__ __forceinline__ bool pdcs_bounded_soc_candidate(
    double u, double f, double h, double h2,
    double left, double right, double left_f, double right_f,
    double *candidate) {
  const double width = right - left;
  if (!(width > 0.0) || !isfinite(f)) return false;
  const double machine_guard = 64.0 * 2.220446049250313e-16;
  const double guard = fmax(machine_guard, 1e-8 * width);

  // Halley's method is cubic near a simple root.  F, F', and F'' are produced
  // by one fused vector traversal/reduction in the enabled implementation.
  if (isfinite(h) && isfinite(h2)) {
    const double denominator = 2.0 * h * h - f * h2;
    if (isfinite(denominator) && fabs(denominator) > 1e-300) {
      const double halley = u - 2.0 * f * h / denominator;
      if (pdcs_bounded_soc_valid_candidate(halley, left, right, guard)) {
        *candidate = halley;
        return true;
      }
    }
  }

  // Ordinary Newton remains the best inexpensive alternative away from the
  // transformed-coordinate endpoints.
  if (isfinite(h) && fabs(h) > 1e-18) {
    const double newton = u - f / h;
    if (pdcs_bounded_soc_valid_candidate(newton, left, right, guard)) {
      *candidate = newton;
      return true;
    }

    // In logit coordinates, additive steps correspond to multiplicative
    // refinement of u/(1-u), which resolves roots close to either endpoint.
    const double clamped_u = fmin(fmax(u, machine_guard),
                                  1.0 - machine_guard);
    const double derivative_logit = h * clamped_u * (1.0 - clamped_u);
    if (isfinite(derivative_logit) && fabs(derivative_logit) > 1e-18) {
      const double z = log(clamped_u / (1.0 - clamped_u));
      const double z_candidate = z - f / derivative_logit;
      double logit;
      if (z_candidate >= 36.0) {
        logit = 1.0 - machine_guard;
      } else if (z_candidate <= -36.0) {
        logit = machine_guard;
      } else {
        logit = 1.0 / (1.0 + exp(-z_candidate));
      }
      if (pdcs_bounded_soc_valid_candidate(logit, left, right, guard)) {
        *candidate = logit;
        return true;
      }
    }
  }

  // If all derivative steps are rejected, use the current strict bracket.
  // Illinois interpolation usually improves substantially on a midpoint but
  // the midpoint remains the unconditional finite fallback.
  const double denominator = right_f - left_f;
  if (isfinite(left_f) && isfinite(right_f) && isfinite(denominator) &&
      fabs(denominator) > 1e-300) {
    const double illinois = left - left_f * width / denominator;
    if (pdcs_bounded_soc_valid_candidate(illinois, left, right, guard)) {
      *candidate = illinois;
      return true;
    }
  }
  const double midpoint = 0.5 * (left + right);
  if (pdcs_bounded_soc_valid_candidate(midpoint, left, right,
                                       machine_guard)) {
    *candidate = midpoint;
    return true;
  }
  return false;
}
