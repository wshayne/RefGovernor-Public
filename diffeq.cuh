#ifndef DIIFEQ_REFGOV_H
#define DIFFEQ_REFGOV_H
#include "functions.cuh"
#include "curand_kernel.h"

typedef void (*diffeq_fn) (const void*, const float*, const float, float*);
typedef void (*diffeq_step_fn) (const void*, diffeq_fn, const float*, float*, float*, const float, const float, size_t);

__host__ __device__
void rk4_step(const void* param, diffeq_fn f, const float* y, float* buffer, float* y_next, const float step, const float kappa, size_t n);

__host__ __device__
void forward_euler_step(const void* param, diffeq_fn f, const float* y, float* buffer, float* y_next, const float step, const float kappa, size_t n);

__host__ __device__
void fuelcell_step(const void* parameters, const float* y, const float kappa, float* dy);

__host__
void threaded_diffeq(const void* param, diffeq_step_fn sf, diffeq_fn f, float* y, float* disturbances, const float* y0, const float* kappa, const float step, size_t n_steps, size_t thread_count, size_t kappa_count, size_t disturbance_count, size_t n);

__host__
void cpu_rk4_timing(const param* p, bool* valid, float* disturbances, const float* y0, const float* kappa, const float step, size_t n_steps, size_t thread_count, size_t kappa_count, size_t disturbance_count);

__global__
void find_kappa_rk4_kernel(param* p, float* kappa, bool* valid, float* disturbances, float time_step, size_t steps, float* x0, size_t num_kappa);

__global__
void fe_timing_kernel(param* p, float* kappa, bool* valid, float* disturbances, float time_step, size_t steps, float* x0, size_t num_kappa);

__global__
void setup_rand_kernel(curandState *state, unsigned long long seed);

__global__
void generate_disturbances_kernel(float* disturbances, curandState *state, size_t n_steps);

__host__
float find_valid_kappa(float* kappa, size_t num_kappa, size_t num_disturbances, bool* results);
#endif