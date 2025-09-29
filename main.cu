#include <chrono>
#include "fileops.h"
#include "curand.h"
#include "diffeq.cuh"

#define N_KAPPA 32
#define N_STEPS 1024
#define T_STEP 0.01
#define MAX_DISTURBANCES 8192

__global__
void blank_kernel() {
    
}

int main() {
    blank_kernel<<<1, 1>>>();
    for (int k=0; k<200; ++k) {
        float kappa[N_KAPPA];
        for (int i=0; i<N_KAPPA; ++i) {
            kappa[i] = (float)i/(N_KAPPA-1);
        }

        param p = param();

        float* times = new float[MAX_DISTURBANCES / 32 * 3];
        int run = 0;
        for (unsigned long long i=32; i<=MAX_DISTURBANCES; i+=32) {
            bool *valid = new bool[i*N_KAPPA];
            float *disturbances = new float[i * N_KAPPA * N_STEPS * 4];
            float x0[4] = {208310, 8420, 223399, -0.0155};
            dim3 threadsPerBlock(16, min(i, (unsigned long long) 16));
            dim3 numBlocks(2, i/threadsPerBlock.y);

            cudaEvent_t start, stop, nomem_start, nomem_stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventCreate(&nomem_start);
            cudaEventCreate(&nomem_stop);

            cudaEventRecord(start);
            float *dev_kappa, *dev_x0, *dev_disturbances;
            bool *dev_valid;
            param* dev_p;
            cudaMalloc((void**) &dev_kappa, N_KAPPA * sizeof(float));
            cudaMalloc((void**) &dev_p, sizeof(param));
            cudaMalloc((void**) &dev_x0, 4 * sizeof(float));
            cudaMalloc((void**) &dev_valid, i * N_KAPPA * sizeof(bool));
            cudaMalloc((void**) &dev_disturbances, i * N_KAPPA * N_STEPS * 4 * sizeof(float));

            curandState *state;
            cudaMalloc((void**) &state, i * N_KAPPA * sizeof(curandState) * 4);
            dim3 genThreadsPerBlock(4, min(i * N_KAPPA, (unsigned long long) 64));
            dim3 genNumBlocks(1, i * N_KAPPA / genThreadsPerBlock.y);
            setup_rand_kernel<<<genNumBlocks, genThreadsPerBlock>>>(state, 0);
            cudaDeviceSynchronize();
            generate_disturbances_kernel<<<genNumBlocks, genThreadsPerBlock>>>(dev_disturbances, state, N_STEPS);
            cudaDeviceSynchronize();
            cudaFree(state);

            cudaMemcpy(dev_kappa, kappa, N_KAPPA * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_p, &p, sizeof(param), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_x0, x0, 4 * sizeof(float), cudaMemcpyHostToDevice);
            cudaEventRecord(nomem_start);
            find_kappa_rk4_kernel<<<numBlocks, threadsPerBlock, (4 + 4*5) * sizeof(float) * threadsPerBlock.x * threadsPerBlock.y>>>(dev_p, dev_kappa, dev_valid, dev_disturbances, T_STEP, N_STEPS, dev_x0, N_KAPPA);
            cudaEventRecord(nomem_stop);
            cudaMemcpy(valid, dev_valid, i * N_KAPPA * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaMemcpy(disturbances, dev_disturbances, i * N_KAPPA * N_STEPS * 4 * sizeof(float), cudaMemcpyDeviceToHost);

            cudaEventSynchronize(stop);
            float milliseconds=0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            times[3*run] = milliseconds;
            milliseconds=0;
            cudaEventElapsedTime(&milliseconds, nomem_start, nomem_stop);
            times[3*run+1] = milliseconds;

            cudaFree(dev_kappa);
            cudaFree(dev_p);
            cudaFree(dev_x0);
            cudaFree(dev_valid);
            cudaFree(dev_disturbances);
            delete[] valid;

            bool* cpu_valid = new bool[i * N_KAPPA];
            auto cpu_start = std::chrono::high_resolution_clock::now();
            cpu_rk4_timing(&p, cpu_valid, disturbances, x0, kappa, T_STEP, N_STEPS, 16, N_KAPPA, i);
            auto cpu_stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start);
            printf("%d, %lld, %lld, %.4f\n", run, i, i * N_KAPPA * N_STEPS * 4, (float)duration.count()/1000);
            times[3*run + 2] = (float)duration.count()/1000;

            delete[] cpu_valid;
            delete[] disturbances;
            run += 1;
        }
        char buffer[100];
        sprintf(buffer, "results/times_%d.csv", k);
        ptr_array_to_csv(times, buffer, run, 3);
    }

    return 0;
}