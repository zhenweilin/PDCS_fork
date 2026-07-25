#pragma once
#include <stdint.h>

// Fixed-width diagnostic ABI. A value of -1 means that a counter was not
// emitted by the selected diagnostic path; consumers must never treat -1 as 0.
struct RootProfileRecord {
  int32_t branch_code;
  int32_t interval_expansion_iterations;
  int32_t newton_attempts;
  int32_t newton_accepts;
  int32_t bisection_iterations;
  int32_t oracle_evaluations;
  int32_t warm_start_attempted;
  int32_t warm_start_accepted;
  int32_t max_iter_reached;
  int32_t reserved;
  double final_residual;
};

static_assert(sizeof(RootProfileRecord) == 48, "RootProfileRecord ABI changed");

__device__ RootProfileRecord* pdcs_root_profile_records = nullptr;
__device__ long pdcs_root_profile_count = 0;

__device__ inline long pdcs_root_profile_index() {
#if defined(PDCS_PROFILE_MAPPING_THREAD)
  return (long)blockIdx.x * blockDim.x + threadIdx.x;
#elif defined(PDCS_PROFILE_MAPPING_WARP)
  return ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
#elif defined(PDCS_PROFILE_MAPPING_BLOCK)
  return (long)blockIdx.x;
#else
  return -1;
#endif
}

__device__ inline bool pdcs_root_profile_leader() {
#if defined(PDCS_PROFILE_MAPPING_THREAD)
  return true;
#elif defined(PDCS_PROFILE_MAPPING_WARP)
  return (threadIdx.x & 31) == 0;
#elif defined(PDCS_PROFILE_MAPPING_BLOCK)
  return threadIdx.x == 0;
#else
  return false;
#endif
}

#define PDCS_PROFILE_ADD(field, amount) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    atomicAdd(&pdcs_root_profile_records[_pdcs_i].field, (int)(amount)); \
} while (0)
#define PDCS_PROFILE_BISECTION() PDCS_PROFILE_ADD(bisection_iterations, 1)
#define PDCS_PROFILE_EXPANSION() PDCS_PROFILE_ADD(interval_expansion_iterations, 1)
#define PDCS_PROFILE_ORACLE() PDCS_PROFILE_ADD(oracle_evaluations, 1)
#define PDCS_PROFILE_NEWTON_ATTEMPT() PDCS_PROFILE_ADD(newton_attempts, 1)
#define PDCS_PROFILE_NEWTON_ACCEPT() PDCS_PROFILE_ADD(newton_accepts, 1)
#define PDCS_PROFILE_MAX_ITER() do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].max_iter_reached = 1; \
} while (0)
#define PDCS_PROFILE_RESIDUAL(value) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].final_residual = (double)(value); \
} while (0)
#define PDCS_PROFILE_BRANCH(code) do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].branch_code = (int32_t)(code); \
} while (0)
#define PDCS_PROFILE_WARM_ATTEMPT() do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].warm_start_attempted = 1; \
} while (0)
#define PDCS_PROFILE_WARM_ACCEPT() do { \
  long _pdcs_i = pdcs_root_profile_index(); \
  if (pdcs_root_profile_records && pdcs_root_profile_leader() && \
      _pdcs_i >= 0 && _pdcs_i < pdcs_root_profile_count) \
    pdcs_root_profile_records[_pdcs_i].warm_start_accepted = 1; \
} while (0)

__device__ inline RootProfileRecord pdcs_empty_root_record() {
  RootProfileRecord r;
  r.branch_code = -1;
  r.interval_expansion_iterations = 0;
  r.newton_attempts = 0;
  r.newton_accepts = 0;
  r.bisection_iterations = 0;
  r.oracle_evaluations = 0;
  r.warm_start_attempted = 0;
  r.warm_start_accepted = 0;
  r.max_iter_reached = 0;
  r.reserved = 0;
  r.final_residual = 0.0/0.0;
  return r;
}

extern "C" __global__ void pdcs_root_profile_initialize(
    RootProfileRecord* records, long count) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    pdcs_root_profile_records = records;
    pdcs_root_profile_count = count;
  }
  if (i < count) records[i] = pdcs_empty_root_record();
}
