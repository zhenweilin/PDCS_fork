#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>


#define positive_zero 1e-20
#define negative_zero -1e-20
// #define proj_rel_tol 1e-14
// #define proj_abs_tol 1e-16
#define minVal 1e-3
#define minVal_inv 1e+3
#ifndef PDCS_ENABLE_SAFEGUARDED_NEWTON
#define PDCS_ENABLE_SAFEGUARDED_NEWTON 1
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_ORACLE
#define PDCS_ENABLE_FUSED_SOC_ORACLE 1
#endif
#ifndef PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
#define PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS 1
#endif
#ifndef PDCS_ENABLE_GRID_SOC_FASTPATH
// Grid-wise diagonal SOC projection is host-orchestrated.  Reuse its existing
// cone workspace and make the two cheap cone decisions on the host instead of
// allocating three device scalars on every projection call.
#define PDCS_ENABLE_GRID_SOC_FASTPATH 1
#endif
#include "soc_root_coordinate.cuh"

#include "exp_proj_kernel.cu"

// n is the length of the vector, including the first element
// len is the length of the vector, not including the first element or the top two elements

__global__ void box_proj(double *sol, const double *bl, const double *bu, long *n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n) {
      sol[idx] = min(max(sol[idx], bl[idx]), bu[idx]);
    }
}

__global__ void positive_proj(double *sol, long *n){
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *n){
        sol[idx] = max(sol[idx], 0.0);
    }
}

// `exponent_proj_diagonal_kernel` consumes the inverse diagonal. The PTX
// block/warp/thread-wise kernels perform this conversion before calling their
// device function; grid-wise must use the same convention.
__global__ void invert_exp_diagonal(const double* D, double* D_inv){
    int index = threadIdx.x;
    if (index < 3){
        D_inv[index] = 1.0 / D[index];
    }
}



__global__ void soc_proj_scale_kernel(double* sol, double* temp, long* n){
  double t = sol[0];
  double *norm = temp;
  if (*norm + t <= 0.0)
  {
    *norm = 0.0;
  }
  else if (*norm <= t)
  {
    *norm = 1.0;
  }
  else
  {
    sol[0] = *norm;
    *norm = (1.0 + t / *norm) / 2.0;
  }
}

extern void soc_proj(cublasHandle_t handle, double* __restrict__ sol, long* __restrict__ n_cpu, long* __restrict__ n_gpu, long* __restrict__ len_cpu, double* __restrict__ temp, int ThreadPerBlock, int nBlock)
{

  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  // temp for storing the norm of the vector
  cublasDnrm2(handle, *len_cpu, sol + 1, 1, temp);
  soc_proj_scale_kernel<<<1, 1>>>(sol, temp, n_gpu);
  cublasDscal(handle, *n_cpu, temp, sol, 1);


  // create a new handle
  // cublasHandle_t handle_temp;
  // cublasCreate(&handle_temp);

  // cublasSetPointerMode(handle_temp, CUBLAS_POINTER_MODE_DEVICE);
  // // temp for storing the norm of the vector
  // cublasDnrm2(handle_temp, *len_cpu, sol + 1, 1, temp);
  // soc_proj_scale_kernel<<<1, 1>>>(sol, temp, n_gpu);
  // cublasDscal(handle_temp, *n_cpu, temp, sol, 1);
  // cublasDestroy(handle_temp);


}





__global__ void vvrscl(long* __restrict__ len, double* __restrict__ x, double* __restrict__ y, double* __restrict__ z) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len) {
    z[j] = x[j] / y[j];
  }
}

__global__ void soc_cone_dual(double* __restrict__ sol_gpu, long* __restrict__ n_gpu, double* __restrict__ temp_gpu, bool* __restrict__ d_return_flag) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j == 0){
    if (temp_gpu[0] <= -sol_gpu[0] && sol_gpu[0] <= 0){
      *d_return_flag = true;
    }
  }
  __syncthreads();
  if (*d_return_flag){
    for (long j = 0; j < *n_gpu; ++j){
      sol_gpu[j] = 0.0;
    }
  }
}

__global__ void vvscal(long* __restrict__ len, double* __restrict__ x, double* __restrict__ y, double* __restrict__ z) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len) {
    z[j] = x[j] * y[j];
  }
}

__global__ void soc_cone_heuristic(double* __restrict__ sol_gpu, double* __restrict__ temp_gpu, bool* __restrict__ d_return_flag) {
  if (temp_gpu[0] <= sol_gpu[0])
  {
    *d_return_flag = true;
  }
  if (*d_return_flag)
  {
    sol_gpu[0] = max(sol_gpu[0], 0.0);
  }
}

// __global__ void determine_case(int* __restrict__ case_flag, double *__restrict__ sol) {
//   long j = threadIdx.x + blockIdx.x * blockDim.x;
//   if (j == 0){
//     if (sol[0] > proj_rel_tol){
//       *case_flag = 0;
//     }
//     else if (sol[0] < -proj_rel_tol){
//       *case_flag = 1;
//     }
//     else {
//       *case_flag = 2;
//     }
//   }
// }

__global__ void initialize_case0(double* __restrict__ xiRight_gpu, double* __restrict__ xiLeft_gpu, double* __restrict__ oracleVal_gpu) {
    *xiRight_gpu = 0.5;
    *xiLeft_gpu = 0.0;
    *oracleVal_gpu = 1.0;
}

__global__ void initialize_case1(double* __restrict__ xiRight_gpu, double* __restrict__ xiLeft_gpu, double* __restrict__ oracleVal_gpu) {
    *xiRight_gpu = 1.0;
    *xiLeft_gpu = 0.5;
    *oracleVal_gpu = 1.0;
}

__global__ void check_t_range_case0(double* __restrict__ t_warm_start_gpu, double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_auxiliary_flag) {
    xiLeft_gpu[0] = 0.0;
    xiRight_gpu[0] = 0.5;
    oracleVal_gpu[0] = 1.0;
    if (t_warm_start_gpu[0] > xiLeft_gpu[0] && t_warm_start_gpu[0] < xiRight_gpu[0]){
      *d_auxiliary_flag = true;
    }
}

__global__ void oracle_soc_f_sqrt_kernel(double* __restrict__ xi, double* __restrict__ x, double* __restrict__ D_scaled_part_mul_x_part, double* __restrict__ D_scaled_squared_part, double* __restrict__ temp_part, long* __restrict__ len, double* __restrict__ oracleVal_gpu) {
  // len not including the first element
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len){
    temp_part[j] = 1 / (1 + (2 * xi[0]) * D_scaled_squared_part[j]) * D_scaled_part_mul_x_part[j];
  }
}

__global__ void oracle_soc_f_sqrt_value_kernel(
    double xi, double* __restrict__ D_scaled_part_mul_x_part,
    double* __restrict__ D_scaled_squared_part,
    double* __restrict__ temp_part, long len) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < len) {
    temp_part[j] = D_scaled_part_mul_x_part[j] /
        (1.0 + (2.0 * xi) * D_scaled_squared_part[j]);
  }
}

__global__ void oracle_soc_f_sqrt_final_case0(double *__restrict__ xi, double* __restrict__ x, double* __restrict__ temp_part, long* __restrict__ len, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
    *oracleVal_gpu -= (x[0] / (1 - 2 * xi[0]));
    if (fabs(*oracleVal_gpu) < abs_tol){
      *d_return_flag = true;
    }else{
      *d_return_flag = false;
    }
    if (*oracleVal_gpu < 0.0){
      *d_auxiliary_flag = true;
    }else{
      *d_auxiliary_flag = false;
    }
}

__global__ void oracle_soc_f_sqrt_final_case1(double *__restrict__ xi, double* __restrict__ x, double* __restrict__ temp_part, long* __restrict__ len, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
    *oracleVal_gpu -= (x[0] / (1 - 2 * xi[0]));
    if (fabs(*oracleVal_gpu) < abs_tol){
      *d_return_flag = true;
    }else{
      *d_return_flag = false;
    }
    if (*oracleVal_gpu < 0.0){
      *d_auxiliary_flag = true;
    }else{
      *d_auxiliary_flag = false;
    }
}

extern "C" void oracle_soc_f_sqrt_case0(cublasHandle_t handle, double *xi, double *x, double *D_scaled_part_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len_cpu, long *len_gpu, int nThread, int nBlock, double* __restrict__ oracleVal_gpu,  bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
  oracle_soc_f_sqrt_kernel<<<nBlock, nThread>>>(xi, x, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part, len_gpu, oracleVal_gpu);
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracleVal_gpu);
  oracle_soc_f_sqrt_final_case0<<<1, 1>>>(xi, x, temp_part, len_cpu, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
}

extern "C" void oracle_soc_f_sqrt_case1(cublasHandle_t handle, double *xi, double *x, double *D_scaled_mul_x_part, double *D_scaled_squared_part, double *temp_part, long *len_cpu, long *len_gpu, int nThread, int nBlock,  double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, bool* __restrict__ d_auxiliary_flag, double abs_tol, double rel_tol) {
  oracle_soc_f_sqrt_kernel<<<nBlock, nThread>>>(xi, x, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_gpu, oracleVal_gpu);
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracleVal_gpu);
  oracle_soc_f_sqrt_final_case1<<<1, 1>>>(xi, x, temp_part, len_cpu, oracleVal_gpu, d_return_flag, d_auxiliary_flag, abs_tol, rel_tol);
}

__global__ void oracle_soc_h_scale_kernel(
    double xi, double* __restrict__ D_scaled_squared_part,
    double* __restrict__ temp_part, long* __restrict__ len) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < *len) {
    temp_part[j] /= sqrt(fmax(2.0 * xi + D_scaled_squared_part[j], 1e-16));
  }
}

__global__ void oracle_soc_h_scale_value_kernel(
    double xi, double* __restrict__ D_scaled_squared_part,
    double* __restrict__ temp_part, long len) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < len) {
    temp_part[j] /= sqrt(fmax(2.0 * xi + D_scaled_squared_part[j], 1e-16));
  }
}

#if PDCS_ENABLE_FUSED_SOC_ORACLE
// One direct reduction avoids writing `temp_part`, and the Newton variant
// reduces the value and derivative accumulators in the same kernel launch.
// `result` points at the cone's temporary array and has at least two doubles
// for every SOC handled by the grid-wise path.
__global__ void oracle_soc_f_reduce_kernel(
    double xi, const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double local_value = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    double y = D_scaled_mul_x_part[j] /
               (1.0 + (2.0 * xi) * D_scaled_squared_part[j]);
    local_value += y * y;
  }
  partial[threadIdx.x] = local_value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(result, partial[0]);
}

__global__ void oracle_soc_h_reduce_kernel(
    double xi, const double* __restrict__ D_scaled_mul_x_part,
    const double* __restrict__ D_scaled_squared_part, long len,
    double* __restrict__ result) {
  extern __shared__ double partial[];
  double* partial_value = partial;
  double* partial_derivative = partial + blockDim.x;
  double local_value = 0.0;
  double local_derivative = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    double y = D_scaled_mul_x_part[j] /
               (1.0 + (2.0 * xi) * D_scaled_squared_part[j]);
    double y_squared = y * y;
    local_value += y_squared;
    local_derivative += y_squared /
        fmax(2.0 * xi + D_scaled_squared_part[j], 1e-16);
  }
  partial_value[threadIdx.x] = local_value;
  partial_derivative[threadIdx.x] = local_derivative;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_value[threadIdx.x] += partial_value[threadIdx.x + stride];
      partial_derivative[threadIdx.x] +=
          partial_derivative[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, partial_value[0]);
    atomicAdd(result + 1, partial_derivative[0]);
  }
}
#endif

#if PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
// Form D .* x and evaluate both cheap cone tests in a single memory traversal.
// Warp partials keep the shared-memory footprint independent of block size.
__global__ void soc_initial_norm_pair_kernel(
    const double* __restrict__ x, const double* __restrict__ D_scaled,
    double* __restrict__ D_scaled_mul_x, long len,
    double* __restrict__ result) {
  extern __shared__ double warp_partial[];
  double* polar_partial = warp_partial;
  double* weighted_partial = warp_partial + ((blockDim.x + 31) / 32);
  double polar_squared = 0.0;
  double weighted_squared = 0.0;
  for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
       j < len; j += (long)gridDim.x * blockDim.x) {
    double xj = x[j];
    double dj = D_scaled[j];
    double divided = xj / dj;
    double weighted = dj * xj;
    D_scaled_mul_x[j] = weighted;
    polar_squared += divided * divided;
    weighted_squared += weighted * weighted;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    polar_squared += __shfl_down_sync(0xffffffffu, polar_squared, offset);
    weighted_squared +=
        __shfl_down_sync(0xffffffffu, weighted_squared, offset);
  }
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) / 32;
  if (lane == 0) {
    polar_partial[warp] = polar_squared;
    weighted_partial[warp] = weighted_squared;
  }
  __syncthreads();
  if (warp == 0) {
    polar_squared = lane < warp_count ? polar_partial[lane] : 0.0;
    weighted_squared = lane < warp_count ? weighted_partial[lane] : 0.0;
    for (int offset = 16; offset > 0; offset /= 2) {
      polar_squared += __shfl_down_sync(0xffffffffu, polar_squared, offset);
      weighted_squared +=
          __shfl_down_sync(0xffffffffu, weighted_squared, offset);
    }
    if (lane == 0) {
      atomicAdd(result, polar_squared);
      atomicAdd(result + 1, weighted_squared);
    }
  }
}
#endif

static double oracle_soc_f_host(
    cublasHandle_t handle, double xi, double sol0,
    double* xi_gpu, double* D_scaled_mul_x_part,
    double* D_scaled_squared_part, double* temp_part,
    long* len_cpu, long* len_gpu, int nThread, int nBlock,
    double* oracle_gpu) {
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  cudaMemset(oracle_gpu, 0, sizeof(double));
  oracle_soc_f_reduce_kernel<<<nBlock, nThread,
      nThread * sizeof(double)>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, *len_cpu, oracle_gpu);
  double value_sum;
  cudaMemcpy(&value_sum, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost);
  return sqrt(fmax(value_sum, 0.0)) - sol0 / (1.0 - 2.0 * xi);
#else
  oracle_soc_f_sqrt_value_kernel<<<nBlock, nThread>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, *len_cpu);
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracle_gpu);
  double norm;
  cudaMemcpy(&norm, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost);
  return norm - sol0 / (1.0 - 2.0 * xi);
#endif
}

static void oracle_soc_h_host(
    cublasHandle_t handle, double xi, double sol0,
    double* xi_gpu, double* D_scaled_mul_x_part,
    double* D_scaled_squared_part, double* temp_part,
    long* len_cpu, long* len_gpu, int nThread, int nBlock,
    double* oracle_gpu, double* f, double* h) {
#if PDCS_ENABLE_FUSED_SOC_ORACLE
  cudaMemset(oracle_gpu, 0, 2 * sizeof(double));
  oracle_soc_h_reduce_kernel<<<nBlock, nThread,
      2 * nThread * sizeof(double)>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, *len_cpu, oracle_gpu);
  double sums[2];
  cudaMemcpy(sums, oracle_gpu, 2 * sizeof(double), cudaMemcpyDeviceToHost);
  double denominator = 1.0 - 2.0 * xi;
  double right = (sol0 / denominator) * (sol0 / denominator);
  *f = sums[0] - right;
  *h = -4.0 * (sums[1] + right / denominator);
#else
  oracle_soc_f_sqrt_value_kernel<<<nBlock, nThread>>>(
      xi, D_scaled_mul_x_part, D_scaled_squared_part, temp_part, *len_cpu);
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracle_gpu);
  double norm1;
  cudaMemcpy(&norm1, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost);

  oracle_soc_h_scale_value_kernel<<<nBlock, nThread>>>(
      xi, D_scaled_squared_part, temp_part, *len_cpu);
  cublasDnrm2_v2(handle, *len_cpu, temp_part, 1, oracle_gpu);
  double norm2;
  cudaMemcpy(&norm2, oracle_gpu, sizeof(double), cudaMemcpyDeviceToHost);

  double denominator = 1.0 - 2.0 * xi;
  double right = (sol0 / denominator) * (sol0 / denominator);
  *f = norm1 * norm1 - right;
  *h = -4.0 * (norm2 * norm2 + right / denominator);
#endif
}

static bool soc_safeguarded_candidate_host(
    double x, double f, double h, double left, double right,
    bool increasing, double rel_tol, double* candidate) {
  double width = right - left;
  if (!isfinite(f) || !isfinite(h) || fabs(h) <= 1e-18 || width <= 0.0) {
    return false;
  }
#if PDCS_SOC_COORDINATE_MODE == 0
  *candidate = x - f / h;
  double guard = fmax(rel_tol, 1e-8 * width);
  return isfinite(*candidate) && *candidate > left + guard &&
         *candidate < right - guard;
#else
  double shift = pdcs_soc_coordinate_shift(rel_tol);
  bool use_log_coordinate = true;
#if PDCS_SOC_COORDINATE_MODE == 3
  use_log_coordinate = pdcs_soc_newton_needs_log_coordinate(
      x, left, right, increasing, shift);
#endif
  if (!use_log_coordinate) {
    *candidate = x - f / h;
    double guard = fmax(rel_tol, 1e-8 * width);
    return isfinite(*candidate) && *candidate > left + guard &&
           *candidate < right - guard;
  }
  double u = pdcs_soc_root_to_coordinate(x, increasing, shift);
  double u_left = pdcs_soc_root_to_coordinate(left, increasing, shift);
  double u_right = pdcs_soc_root_to_coordinate(right, increasing, shift);
  double derivative_u = h * pdcs_soc_root_coordinate_derivative(
      x, increasing, shift);
  if (!isfinite(derivative_u) || fabs(derivative_u) <= 1e-18) return false;
  double candidate_u = u - f / derivative_u;
  *candidate = pdcs_soc_coordinate_to_root(
      candidate_u, increasing, shift);
  double u_guard = 1e-8 * (u_right - u_left);
  return isfinite(candidate_u) && isfinite(*candidate) &&
         candidate_u > u_left + u_guard &&
         candidate_u < u_right - u_guard &&
         *candidate > left && *candidate < right;
#endif
}

static bool soc_exponent_expansion_bracket_host(
    cublasHandle_t handle, double sol0, double* xi_gpu,
    double* D_scaled_mul_x_part, double* D_scaled_squared_part,
    double* temp_part, long* len_cpu, long* len_gpu, int nThread,
    int nBlock, double* oracle_gpu, double* left, double* right,
    double* f, double abs_tol) {
#if !PDCS_ENABLE_EXPONENT_EXPANSION
  return false;
#else
  *f = oracle_soc_f_host(handle, *right, sol0, xi_gpu,
      D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu,
      len_gpu, nThread, nBlock, oracle_gpu);
  if (!isfinite(*f)) return false;
  if (fabs(*f) <= abs_tol) {
    *left = *right;
    return true;
  }
  if (*f >= 0.0) return true;
  double base_q = *right - 0.5;
  if (!(base_q > 0.0) || !isfinite(base_q)) return false;
  int low_exponent = 0;
  int high_exponent = 1;
  bool found = false;
  while (high_exponent <= 1020) {
    double candidate = 0.5 + ldexp(base_q, high_exponent);
    if (!isfinite(candidate)) break;
    double candidate_f = oracle_soc_f_host(handle, candidate, sol0, xi_gpu,
        D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu,
        len_gpu, nThread, nBlock, oracle_gpu);
    if (!isfinite(candidate_f)) break;
    if (fabs(candidate_f) <= abs_tol) {
      *left = candidate;
      *right = candidate;
      *f = candidate_f;
      return true;
    }
    if (candidate_f >= 0.0) {
      *f = candidate_f;
      found = true;
      break;
    }
    low_exponent = high_exponent;
    if (high_exponent > 510) break;
    high_exponent *= 2;
  }
  if (!found) return false;
  while (high_exponent - low_exponent > 1) {
    int mid_exponent = low_exponent +
                       (high_exponent - low_exponent) / 2;
    double candidate = 0.5 + ldexp(base_q, mid_exponent);
    double candidate_f = oracle_soc_f_host(handle, candidate, sol0, xi_gpu,
        D_scaled_mul_x_part, D_scaled_squared_part, temp_part, len_cpu,
        len_gpu, nThread, nBlock, oracle_gpu);
    if (!isfinite(candidate_f)) return false;
    if (fabs(candidate_f) <= abs_tol) {
      *left = candidate;
      *right = candidate;
      *f = candidate_f;
      return true;
    }
    if (candidate_f >= 0.0) {
      high_exponent = mid_exponent;
      *f = candidate_f;
    }
    else low_exponent = mid_exponent;
  }
  *left = 0.5 + ldexp(base_q, low_exponent);
  *right = 0.5 + ldexp(base_q, high_exponent);
  return true;
#endif
}

__global__ void recover_sol_case01(double* __restrict__ sol, double* __restrict__ t_warm_start, double* __restrict__ D_scaled_squared, long* __restrict__ n) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j == 0){
    sol[0] = sol[0] / (1 - 2 * t_warm_start[0]) * minVal;
  }
  if (j > 0 && j < *n){
    sol[j] = sol[j] / (1 + 2 * t_warm_start[0] * D_scaled_squared[j]) * minVal;
  }
}

__global__ void binary_search_case0(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu,  double* __restrict__ oracleVal_gpu, double *t_warm_start_gpu, bool* __restrict__ d_auxiliary_flag, bool* __restrict__ d_return_flag, double abs_tol, double rel_tol) {
  if (*d_auxiliary_flag){
    *xiRight_gpu = *t_warm_start_gpu;
  }else{
    *xiLeft_gpu = *t_warm_start_gpu;
  }
  if ((xiRight_gpu[0] - xiLeft_gpu[0]) / (1 + xiRight_gpu[0] + xiLeft_gpu[0]) <= rel_tol || fabs(oracleVal_gpu[0]) <= abs_tol){
    *d_return_flag = true;
  }
}

__global__ void binary_search_case1(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ oracleVal_gpu, double *t_warm_start_gpu, bool* __restrict__ d_auxiliary_flag, bool* __restrict__ d_return_flag, double abs_tol, double rel_tol) {
  // For t < 0, f(xi) is increasing on (1/2, +inf): a negative value
  // means the root is to the right of the current midpoint.
  if (*d_auxiliary_flag){
    *xiLeft_gpu = *t_warm_start_gpu;
  }else{
    *xiRight_gpu = *t_warm_start_gpu;
  }
  if ((xiRight_gpu[0] - xiLeft_gpu[0]) / (1 + xiRight_gpu[0] + xiLeft_gpu[0]) <= rel_tol || fabs(oracleVal_gpu[0]) <= abs_tol){
    *d_return_flag = true;
  }
}

__global__ void average_xi(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ t_warm_start_gpu) {
  *t_warm_start_gpu = (*xiLeft_gpu + *xiRight_gpu) / 2;
}

__global__ void end_while_loop(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu, double* __restrict__ oracleVal_gpu, bool* __restrict__ d_return_flag, double rel_tol, double abs_tol) {
  if ((xiRight_gpu[0] - xiLeft_gpu[0]) / (1 + xiRight_gpu[0] + xiLeft_gpu[0]) <= rel_tol || fabs(oracleVal_gpu[0]) <= abs_tol){
    *d_return_flag = true;
  }
}

__global__ void enlarge_xi_right(double* __restrict__ xiLeft_gpu, double* __restrict__ xiRight_gpu) {
    *xiLeft_gpu = *xiRight_gpu;
    *xiRight_gpu *= 2;
}

__global__ void recover_sol_case3(double* __restrict__ sol_gpu, double* __restrict__ temp_gpu, double* __restrict__ D_scaled_gpu, long* __restrict__ n_gpu, double* __restrict__ t_warm_start_gpu, double* __restrict__ D_scaled_squared_gpu) {
  long j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j > 0 && j < *n_gpu){
    sol_gpu[j] = sol_gpu[j] / (1 + D_scaled_squared_gpu[j]) * minVal;
    temp_gpu[j] = D_scaled_gpu[j] * sol_gpu[j];
  }
}

extern "C" void soc_proj_diagonal(cublasHandle_t handle,
                                 double* sol_gpu,
                                 long* len_gpu,
                                 long* n_gpu,
                                 long* len_cpu,
                                 long* n_cpu,
                                 double* D_scaled_gpu,
                                 double* D_scaled_squared_gpu,
                                 double* D_scaled_mul_x_gpu,
                                 double* temp_gpu,
                                 double* t_warm_start_gpu,
                                 int nThread,
                                 int nBlock,
                                 bool* d_return_flag,
                                 bool* d_auxiliary_flag,
                                 double abs_tol,
                                 double rel_tol) {
  // This is deliberately a cuBLAS implementation: Dnrm2 performs the
  // large-vector reductions while small kernels only evaluate/recover the
  // scalar root.  The temporary root bounds must not alias D_scaled[0] or
  // D_scaled_squared[0], which are persistent solver scaling data.
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  const double scale_factor_inv = minVal_inv;
  cublasDscal_v2(handle, *n_cpu, &scale_factor_inv, sol_gpu, 1);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  double* x_tail = sol_gpu + 1;
  double* d_tail = D_scaled_gpu + 1;
  double* d_squared_tail = D_scaled_squared_gpu + 1;
  double* d_times_x_tail = D_scaled_mul_x_gpu + 1;
  double* work_tail = temp_gpu + 1;
  double* oracle = temp_gpu;
#if !PDCS_ENABLE_GRID_SOC_FASTPATH
  bool host_flag = false;
#endif

#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  thrust::device_ptr<double> x_thrust = thrust::device_pointer_cast(x_tail);
  thrust::device_ptr<double> d_thrust = thrust::device_pointer_cast(d_tail);
  thrust::device_ptr<double> d_times_x_thrust = thrust::device_pointer_cast(d_times_x_tail);
  thrust::device_ptr<double> work_thrust = thrust::device_pointer_cast(work_tail);
#endif

  double sol0;
  cudaMemcpy(&sol0, sol_gpu, sizeof(double), cudaMemcpyDeviceToHost);

  // Test the polar cone with ||x ./ D|| and feasibility with ||D .* x||.
#if PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  cudaMemset(oracle, 0, 2 * sizeof(double));
  int warp_count = (nThread + 31) / 32;
  soc_initial_norm_pair_kernel<<<nBlock, nThread,
      2 * warp_count * sizeof(double)>>>(
      x_tail, d_tail, d_times_x_tail, *len_cpu, oracle);
  double initial_sums[2];
  cudaMemcpy(initial_sums, oracle, 2 * sizeof(double), cudaMemcpyDeviceToHost);
  double polar_norm = sqrt(fmax(initial_sums[0], 0.0));
  double weighted_norm = sqrt(fmax(initial_sums[1], 0.0));
#else
  thrust::transform(x_thrust, x_thrust + *len_cpu, d_thrust, work_thrust,
                    thrust::divides<double>());
  cublasDnrm2_v2(handle, *len_cpu, work_tail, 1, oracle);
#endif
#if PDCS_ENABLE_GRID_SOC_FASTPATH
#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  double polar_norm;
  cudaMemcpy(&polar_norm, oracle, sizeof(double), cudaMemcpyDeviceToHost);
#endif
  if (polar_norm <= -sol0 && sol0 <= 0.0) {
    cudaMemset(sol_gpu, 0, *n_cpu * sizeof(double));
    return;
  }
#else
  cudaMemset(d_return_flag, 0, sizeof(bool));
  soc_cone_dual<<<nBlock, nThread>>>(sol_gpu, n_gpu, oracle, d_return_flag);
  cudaMemcpy(&host_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  if (host_flag) return;
#endif

#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  thrust::transform(d_thrust, d_thrust + *len_cpu, x_thrust,
                    d_times_x_thrust, thrust::multiplies<double>());
  cublasDnrm2_v2(handle, *len_cpu, d_times_x_tail, 1, oracle);
#endif
#if PDCS_ENABLE_GRID_SOC_FASTPATH
#if !PDCS_ENABLE_FUSED_SOC_INITIAL_TESTS
  double weighted_norm;
  cudaMemcpy(&weighted_norm, oracle, sizeof(double), cudaMemcpyDeviceToHost);
#endif
  if (weighted_norm <= sol0) {
    const double scale_factor = minVal;
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasDscal_v2(handle, *n_cpu, &scale_factor, sol_gpu, 1);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    return;
  }
#else
  cudaMemset(d_return_flag, 0, sizeof(bool));
  soc_cone_heuristic<<<1, 1>>>(sol_gpu, oracle, d_return_flag);
  cudaMemcpy(&host_flag, d_return_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  if (host_flag) {
    const double scale_factor = minVal;
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasDscal_v2(handle, *n_cpu, &scale_factor, sol_gpu, 1);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    return;
  }
#endif

  double warm_x;
  cudaMemcpy(&warm_x, t_warm_start_gpu, sizeof(double), cudaMemcpyDeviceToHost);

  if (sol0 >= rel_tol) {
    double left = 0.0;
    double right = 0.5;
    double f = 1.0;
    if (warm_x > left && warm_x < right) {
      double h;
      oracle_soc_h_host(handle, warm_x, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle, &f, &h);
      double x = warm_x;
      for (int iter = 0;
           iter < PDCS_SOC_NEWTON_STEPS && PDCS_ENABLE_SAFEGUARDED_NEWTON;
           ++iter) {
        if (f < 0.0) right = x; else left = x;
        if (fabs(f) <= abs_tol * abs_tol) {
          left = x;
          right = x;
          break;
        }
        double candidate;
        if (!soc_safeguarded_candidate_host(
                x, f, h, left, right, false, rel_tol, &candidate)) break;
        double old_abs_f = fabs(f);
        x = candidate;
        oracle_soc_h_host(handle, x, sol0, t_warm_start_gpu,
            d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
            nThread, nBlock, oracle, &f, &h);
        if (fabs(f) >= old_abs_f) {
          if (f < 0.0) right = x; else left = x;
          break;
        }
      }
    }
    if (left != right) f = 1.0;
    double xi = 0.5 * (left + right);
    while ((right - left) / (1.0 + right + left) > rel_tol &&
           fabs(f) > abs_tol) {
      xi = pdcs_soc_bisection_midpoint(left, right, false, rel_tol);
      f = oracle_soc_f_host(handle, xi, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle);
      if (f < 0.0) right = xi; else left = xi;
    }
    cudaMemcpy(t_warm_start_gpu, &xi, sizeof(double), cudaMemcpyHostToDevice);
    recover_sol_case01<<<nBlock, nThread>>>(sol_gpu, t_warm_start_gpu,
                                              D_scaled_squared_gpu, n_gpu);
  } else if (sol0 <= -rel_tol) {
    double left = 0.5, right = 1.0;
    double f = 1.0;
    if (warm_x > left) {
      double h;
      oracle_soc_h_host(handle, warm_x, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle, &f, &h);
      double x = warm_x;
      for (int iter = 0;
           iter < PDCS_SOC_NEWTON_STEPS && PDCS_ENABLE_SAFEGUARDED_NEWTON;
           ++iter) {
        if (f > 0.0) right = x; else left = x;
        if (fabs(f) <= abs_tol * abs_tol) {
          left = x;
          right = x;
          break;
        }
        double candidate;
        if (!soc_safeguarded_candidate_host(
                x, f, h, left, right, true, rel_tol, &candidate)) break;
        double old_abs_f = fabs(f);
        x = candidate;
        oracle_soc_h_host(handle, x, sol0, t_warm_start_gpu,
            d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
            nThread, nBlock, oracle, &f, &h);
        if (fabs(f) >= old_abs_f) {
          if (f > 0.0) right = x; else left = x;
          break;
        }
      }
    }
    bool exponent_bracketed = soc_exponent_expansion_bracket_host(
        handle, sol0, t_warm_start_gpu, d_times_x_tail, d_squared_tail,
        work_tail, len_cpu, len_gpu, nThread, nBlock, oracle, &left,
        &right, &f, abs_tol);
    if (!exponent_bracketed) {
      f = oracle_soc_f_host(handle, right, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle);
      while (f < 0.0) {
        left = right;
        right *= 2.0;
        f = oracle_soc_f_host(handle, right, sol0, t_warm_start_gpu,
            d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
            nThread, nBlock, oracle);
      }
    }
    double xi = 0.5 * (left + right);
    while ((right - left) / (1.0 + right + left) > rel_tol &&
           fabs(f) > abs_tol) {
      xi = pdcs_soc_bisection_midpoint(left, right, true, rel_tol);
      f = oracle_soc_f_host(handle, xi, sol0, t_warm_start_gpu,
          d_times_x_tail, d_squared_tail, work_tail, len_cpu, len_gpu,
          nThread, nBlock, oracle);
      if (f > 0.0) right = xi; else left = xi;
    }
    cudaMemcpy(t_warm_start_gpu, &xi, sizeof(double), cudaMemcpyHostToDevice);
    recover_sol_case01<<<nBlock, nThread>>>(sol_gpu, t_warm_start_gpu,
                                              D_scaled_squared_gpu, n_gpu);
  } else {
    recover_sol_case3<<<nBlock, nThread>>>(sol_gpu, temp_gpu, D_scaled_gpu,
                                            n_gpu, t_warm_start_gpu,
                                            D_scaled_squared_gpu);
    cublasDnrm2_v2(handle, *len_cpu, temp_gpu + 1, 1, sol_gpu);
  }
}


// function for setting function pointers
//     0: dual_free_proj!
//     1: dual_free_proj_diagonal!
//     2: con_zero_proj!
//     3: dual_positive_proj!
//     4: dual_positive_proj_diagonal!
//     5: dual_soc_proj!
//     6: dual_soc_proj_diagonal!
//     7: dual_soc_proj_const_scale_diagonal!
//     8: dual_rsoc_proj!
//     9: dual_rsoc_proj_diagonal!
//     10: dual_rsoc_proj_const_scale_diagonal!
//     11: dual_EXP_proj!
//     12: dual_EXP_proj_diagonal!
//     13: con_EXP_proj!
//     14: dual_DUALEXP_proj!
//     15: dual_DUALEXP_proj_diagonal!
//     16: con_DUALEXP_proj!
//     17: box_proj!
//     18: box_proj_diagonal!
//     19: slack_box_proj!
//     20: soc_cone_proj!
//     21: soc_cone_proj_const_scale!
//     22: soc_cone_proj_diagonal!
//     23: rsoc_cone_proj!
//     24: rsoc_cone_proj_const_scale!
//     25: rsoc_cone_proj_diagonal!
//     26: EXP_proj!
//     27: EXP_proj_diagonal!
//     28: DUALEXP_proj!
//     29: DUALEXPonent_proj_diagonal!

extern "C" void few_block_proj(cublasHandle_t handle,
                             double* arr, 
                             double* bl, 
                             double* bu, 
                             double* D_scaled,  
                             double* D_scaled_squared,  
                             double* D_scaled_mul_x, 
                             double* temp, 
                             double* t_warm_start, 
                             long* cpu_head_start,  
                             long* ns_gpu, 
                             long* ns_cpu, 
                             int blkNum, 
                             long* cpu_proj_type,  
                             int ThreadPerBlock, 
                             int nBlock,
                             double abs_tol,
                             double rel_tol)
{
  for (int i = 0; i < blkNum; ++i)
  {
    long *n_gpu = ns_gpu+i;
    long n_cpu = ns_cpu[i];
    double *sol = arr + cpu_head_start[i];
    double *sub_D_scaled = D_scaled + cpu_head_start[i];
    double *sub_D_scaled_squared = D_scaled_squared + cpu_head_start[i];
    double *sub_D_scaled_mul_x = D_scaled_mul_x + cpu_head_start[i];
    double *sub_temp = temp + cpu_head_start[i];
    double *sub_bl = bl + cpu_head_start[i];
    double *sub_bu = bu + cpu_head_start[i];
    if (cpu_proj_type[i] == 0 || cpu_proj_type[i] == 1){
      // dual_free_proj
      ;
    }
    else if (cpu_proj_type[i] == 17 || cpu_proj_type[i] == 19 || cpu_proj_type[i] == 18){
      // box
      box_proj<<<nBlock, ThreadPerBlock, 0>>>(sol, sub_bl, sub_bu, n_gpu);
    }
    else if (cpu_proj_type[i] == 2){
      // zeros<<<nBlock, ThreadPerBlock, 0>>>(sol, n_gpu);
      cudaMemset(sol, 0, n_cpu * sizeof(double));
    }
    else if (cpu_proj_type[i] == 3 || cpu_proj_type[i] == 4){
      // dual_positive
      positive_proj<<<nBlock, ThreadPerBlock, 0>>>(sol, n_gpu);
    }
    else if (cpu_proj_type[i] == 5 || cpu_proj_type[i] == 7 || cpu_proj_type[i] == 20 || cpu_proj_type[i] == 21){
      long len_cpu = n_cpu - 1;
      // sub_temp is already device memory supplied by the caller.  The old
      // code passed it as a host source to cudaMemcpy, which is invalid, and
      // allocated/freed a scalar on every projection. cublasDnrm2 overwrites
      // the workspace before it is read, so use the existing device scalar.
      soc_proj(handle, sol, &n_cpu, n_gpu, &len_cpu, sub_temp, ThreadPerBlock, nBlock);
    }
    else if (cpu_proj_type[i] == 6 || cpu_proj_type[i] == 22){
      long len_cpu = n_cpu - 1;
#if PDCS_ENABLE_GRID_SOC_FASTPATH
      bool* d_auxiliary_flag = nullptr;
      bool* d_return_flag = nullptr;
      long* len_gpu = nullptr;
#else
      bool* d_auxiliary_flag;
      bool h_auxiliary_flag = false;
      cudaMalloc(&d_auxiliary_flag, sizeof(bool));
      cudaMemcpy(d_auxiliary_flag, &h_auxiliary_flag, sizeof(bool), cudaMemcpyHostToDevice);
      bool* d_return_flag;
      bool h_return_flag = false;
      cudaMalloc(&d_return_flag, sizeof(bool));
      cudaMemcpy(d_return_flag, &h_return_flag, sizeof(bool), cudaMemcpyHostToDevice);
      long* len_gpu;
      cudaMalloc(&len_gpu, sizeof(long));
      cudaMemcpy(len_gpu, &len_cpu, sizeof(long), cudaMemcpyHostToDevice);
#endif
      soc_proj_diagonal(handle,
                       sol, 
                       len_gpu,
                       n_gpu, 
                       &len_cpu,
                       &n_cpu,
                       sub_D_scaled, 
                       sub_D_scaled_squared, 
                       sub_D_scaled_mul_x, 
                       sub_temp, 
                       &t_warm_start[i], 
                       ThreadPerBlock, 
                       nBlock, 
                       d_return_flag, 
                       d_auxiliary_flag,
                       abs_tol,
                       rel_tol);
#if !PDCS_ENABLE_GRID_SOC_FASTPATH
      cudaFree(len_gpu);
      cudaFree(d_return_flag);
      cudaFree(d_auxiliary_flag);
#endif
    }
    else if (cpu_proj_type[i] == 8 || cpu_proj_type[i] == 10 || cpu_proj_type[i] == 23 || cpu_proj_type[i] == 24){
      printf("use cublas for rsoc projection is developing!\n");
    //   rsoc_proj(handle, sol, &n, sub_D_scaled_mul_x, sub_temp, ThreadPerBlock, nBlock);
    }
    else if (cpu_proj_type[i] == 9 || cpu_proj_type[i] == 25){
      printf("use cublas for rsoc diagonal projection is developing!\n");
    //   rsoc_proj_diagonal(handle, sol, &n, sub_D_scaled, sub_D_scaled_squared, sub_D_scaled_mul_x, sub_temp, &t_warm_start[i], ThreadPerBlock, nBlock);
    }
    else if (cpu_proj_type[i] == 11 || cpu_proj_type[i] == 16 || cpu_proj_type[i] == 28){
      // dualExponent_proj
      dualExponent_proj_kernel<<<1, 1>>>(sol, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (cpu_proj_type[i] == 14 || cpu_proj_type[i] == 13 || cpu_proj_type[i] == 26 ){
      // exponent_proj
      exponent_proj_kernel<<<1, 1>>>(sol, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (cpu_proj_type[i] == 12 || cpu_proj_type[i] == 29){
      // dualExponent_proj_diagonal
      dualExponent_proj_diagonal_kernel<<<1, 1>>>(sol, sub_D_scaled, sub_temp, &t_warm_start[i], abs_tol, rel_tol);
    }
    else if (cpu_proj_type[i] == 15 || cpu_proj_type[i] == 27){
      // exponent_proj_diagonal
      invert_exp_diagonal<<<1, 3>>>(sub_D_scaled, sub_temp);
      exponent_proj_diagonal_kernel<<<1, 1>>>(sol, sub_temp, &t_warm_start[i], abs_tol, rel_tol);
    }
  }
}
