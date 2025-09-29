#include <chrono>
#include "fileops.h"
#include "curand.h"
#include "diffeq.cuh"


int main(int argc, char* argv[]) {

    int N_KAPPA, N_STEPS, MAX_DISTURBANCES, SIM_STEPS;
    float T_STEP;
    if (argc > 1) {
        N_KAPPA = atoi(argv[1]);
    } else {
        N_KAPPA = 32;
    }
    if (argc > 2) {
        N_STEPS = atoi(argv[2]);
    } else {
        N_STEPS = 1024;
    }
    if (argc > 3) {
        T_STEP = atof(argv[3]);
    } else {
        T_STEP = 0.01;
    }
    if (argc > 4) {
        MAX_DISTURBANCES = atoi(argv[4]);
    } else {
        MAX_DISTURBANCES = 8192;
    }
    if (argc > 5) {
        SIM_STEPS = atoi(argv[5]);
    } else {
        SIM_STEPS = 3000;
    }

    float* kappa = new float[N_KAPPA];
    for (int i=0; i<N_KAPPA; ++i) {
        kappa[i] = (float)i/(N_KAPPA-1);
    }

    param p = param();
    bool *valid = new bool[MAX_DISTURBANCES*N_KAPPA];
    float *disturbances = new float[MAX_DISTURBANCES * N_KAPPA * N_STEPS * 4];
    float x0[4] = {208310, 8420, 223399, -0.0155};
    dim3 threadsPerBlock(16, min((unsigned long long) MAX_DISTURBANCES, (unsigned long long) 16));
    dim3 numBlocks(2, MAX_DISTURBANCES/threadsPerBlock.y);

    float *dev_kappa, *dev_x0, *dev_disturbances;
    bool *dev_valid;
    param* dev_p;
    cudaMalloc((void**) &dev_kappa, N_KAPPA * sizeof(float));
    cudaMalloc((void**) &dev_p, sizeof(param));
    cudaMalloc((void**) &dev_x0, 4 * sizeof(float));
    cudaMalloc((void**) &dev_valid, MAX_DISTURBANCES * N_KAPPA * sizeof(bool));
    cudaMalloc((void**) &dev_disturbances, MAX_DISTURBANCES * N_KAPPA * N_STEPS * 4 * sizeof(float));

    curandState *state;
    cudaMalloc((void**) &state, MAX_DISTURBANCES * N_KAPPA * sizeof(curandState) * 4);
    dim3 genThreadsPerBlock(4, min((unsigned long long) MAX_DISTURBANCES * N_KAPPA, (unsigned long long) 64));
    dim3 genNumBlocks(1, MAX_DISTURBANCES * N_KAPPA / genThreadsPerBlock.y);
    setup_rand_kernel<<<genNumBlocks, genThreadsPerBlock>>>(state, 0);
    cudaDeviceSynchronize();
    generate_disturbances_kernel<<<genNumBlocks, genThreadsPerBlock>>>(dev_disturbances, state, N_STEPS);
    cudaDeviceSynchronize();

    cudaMemcpy(dev_kappa, kappa, N_KAPPA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, &p, sizeof(param), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x0, x0, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(disturbances, dev_disturbances, MAX_DISTURBANCES * N_KAPPA * N_STEPS * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    float step_kappa, buffer[4*5];
    float* y_all = new float[SIM_STEPS * 4];
    // Example Setpoint Curve
    float* desired_setpoint = new float[SIM_STEPS];
    for (int i=0; i<SIM_STEPS; ++i) {
        if (i < SIM_STEPS/4) {
            desired_setpoint[i] = 180;
        } else if (i < SIM_STEPS/2) {
            desired_setpoint[i] = 250;
        } else if (i < 3 * SIM_STEPS / 4) {
            desired_setpoint[i] = 300;
        } else {
            desired_setpoint[i] = 200;
        }
    }
    float* real_setpoint = new float[SIM_STEPS];
    float* kappa_history = new float[SIM_STEPS];
    float* lambda = new float[SIM_STEPS];

    float* y = y_all;
    for (int i=0; i<4; ++i) {
        y[i] = x0[i];
    }
    for (int i=0; i<SIM_STEPS; ++i) {
        // at each step generate new disturbances, then find the highest valid kappa
        generate_disturbances_kernel<<<genNumBlocks, genThreadsPerBlock>>>(dev_disturbances, state, N_STEPS);
        find_kappa_rk4_kernel<<<numBlocks, threadsPerBlock, (4 + 4*5) * sizeof(float) * threadsPerBlock.x * threadsPerBlock.y>>>(dev_p, dev_kappa, dev_valid, dev_disturbances, T_STEP, N_STEPS, dev_x0, N_KAPPA);
        cudaMemcpy(valid, dev_valid, MAX_DISTURBANCES * N_KAPPA * sizeof(bool), cudaMemcpyDeviceToHost);
        step_kappa = find_valid_kappa(kappa, N_KAPPA, MAX_DISTURBANCES, valid);
        kappa_history[i] = step_kappa;

        // take a step in the "real" simulation with that kappa value
        rk4_step(&p, fuelcell_step, y, buffer, y+4, T_STEP, step_kappa, 4);

        y += 4;

        // add disturbances to simulated values
        for (int j=0; j<4; ++j) {
            y[j] += disturbances[j];
        }
        disturbances += 4;

        // calculate the new setpoint requirements, and send them to the GPU
        p.base_voltage += p.voltage_offset * step_kappa;
        p.voltage_offset = desired_setpoint[i] - p.base_voltage;
        real_setpoint[i] = p.base_voltage;
        lambda[i] = p.c23 * (y[2] - y[0]) / (p.c24 * p.base_voltage);
        cudaMemcpy(dev_p, &p, sizeof(param), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_x0, y, 4 * sizeof(float), cudaMemcpyHostToDevice);
    }

    ptr_array_to_csv(y_all, "results/fuelcell_output.csv", SIM_STEPS, 4);
    ptr_array_to_csv(real_setpoint, "results/setpoint_output.csv", SIM_STEPS, 1);
    ptr_array_to_csv(lambda, "results/lambda_o2_output.csv", SIM_STEPS, 1);
    ptr_array_to_csv(kappa_history, "results/kappa_output.csv", SIM_STEPS, 1);
    ptr_array_to_csv(desired_setpoint, "results/desired_setpoint.csv", SIM_STEPS, 1);

    cudaFree(dev_kappa);
    cudaFree(dev_p);
    cudaFree(dev_x0);
    cudaFree(dev_valid);
    cudaFree(dev_disturbances);
    cudaFree(state);
    delete[] valid;
    
    return 0;
}