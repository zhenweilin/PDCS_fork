#pragma once

#ifndef PDCS_EXP_COORDINATE_MODE
// 0: rho; 1: asinh(rho/s); 2: adaptive asinh for large signed brackets.
#define PDCS_EXP_COORDINATE_MODE 0
#endif
#ifndef PDCS_EXP_COORDINATE_SCALE
#define PDCS_EXP_COORDINATE_SCALE 1.0
#endif
#ifndef PDCS_EXP_LOG_THRESHOLD
#define PDCS_EXP_LOG_THRESHOLD 16.0
#endif

__host__ __device__ __forceinline__ bool pdcs_exp_bracket_needs_log_coordinate(
    double low, double high) {
  return fmax(fabs(low), fabs(high)) >=
         PDCS_EXP_LOG_THRESHOLD * PDCS_EXP_COORDINATE_SCALE;
}

__host__ __device__ __forceinline__ double pdcs_exp_root_to_coordinate(
    double rho) {
#if PDCS_EXP_COORDINATE_MODE == 0
  return rho;
#else
  return asinh(rho / PDCS_EXP_COORDINATE_SCALE);
#endif
}

__host__ __device__ __forceinline__ double pdcs_exp_coordinate_to_root(
    double u) {
#if PDCS_EXP_COORDINATE_MODE == 0
  return u;
#else
  return PDCS_EXP_COORDINATE_SCALE * sinh(u);
#endif
}

__host__ __device__ __forceinline__ double pdcs_exp_root_coordinate_derivative(
    double rho) {
#if PDCS_EXP_COORDINATE_MODE == 0
  return 1.0;
#else
  return hypot(PDCS_EXP_COORDINATE_SCALE, rho);
#endif
}

__host__ __device__ __forceinline__ double pdcs_exp_bisection_midpoint(
    double low, double high) {
#if PDCS_EXP_COORDINATE_MODE == 0
  return 0.5 * (low + high);
#else
#if PDCS_EXP_COORDINATE_MODE == 2
  if (!pdcs_exp_bracket_needs_log_coordinate(low, high)) {
    return 0.5 * (low + high);
  }
#endif
  double midpoint = pdcs_exp_coordinate_to_root(
      0.5 * (pdcs_exp_root_to_coordinate(low) +
             pdcs_exp_root_to_coordinate(high)));
  if (!isfinite(midpoint) || midpoint <= low || midpoint >= high) {
    midpoint = 0.5 * (low + high);
  }
  return midpoint;
#endif
}

__host__ __device__ __forceinline__ double pdcs_exp_newton_candidate(
    double rho, double f, double df, double low, double high) {
#if PDCS_EXP_COORDINATE_MODE == 0
  return rho - f / df;
#else
  bool use_log_coordinate = true;
#if PDCS_EXP_COORDINATE_MODE == 2
  use_log_coordinate = pdcs_exp_bracket_needs_log_coordinate(low, high);
#endif
  if (!use_log_coordinate) return rho - f / df;
  double u = pdcs_exp_root_to_coordinate(rho);
  double derivative_u = df * pdcs_exp_root_coordinate_derivative(rho);
  if (!isfinite(derivative_u) || fabs(derivative_u) <= 1e-18) {
    return HUGE_VAL;
  }
  return pdcs_exp_coordinate_to_root(u - f / derivative_u);
#endif
}
