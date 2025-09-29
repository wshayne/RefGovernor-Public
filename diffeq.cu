#include "diffeq.cuh"
#include "functions.cuh"

__host__ __device__
void rk4_step(const void* param, diffeq_fn f, const float* y, float* buffer, float* y_next, const float step, const float kappa, size_t n) {
    float* k1 = buffer;
    float* k2 = k1 + n;
    float* k3 = k2 + n;
    float* k4 = k3 + n;
    float* buff = k4 + n;

    f(param, y, kappa, k1);

    for (int i=0; i<n; ++i) {
        buff[i] = y[i] + step * k1[i] / 2.0f;
    }

    f(param, buff, kappa, k2);

    for (int i=0; i<n; ++i) {
        buff[i] = y[i] + step * k2[i] / 2.0f;
    }

    f(param, buff, kappa, k3);

    for (int i=0; i<n; ++i) {
        buff[i] = y[i] + step * k3[i];
    }

    f(param, buff, kappa, k4);

    for (int i=0; i<n; ++i) {
        y_next[i] = y[i] + step / 6.0f * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }
}

__host__ __device__
void forward_euler_step(const void* param, diffeq_fn f, const float* y, float* buffer, float* y_next, const float step, const float kappa, size_t n) {
    float* dy = buffer;
    f(param, y, kappa, dy);
    for (int i=0; i<n; ++i) {
        y_next[i] = y[i] + step * dy[i];
    }
}

__host__ __device__
void fuelcell_step(const void* parameters, const float* y, const float kappa, float* dy) {
    param* p = (param*) parameters;
    float v = p->base_voltage + p->voltage_offset * kappa;
    float Wcpref = p->c25 * v;
    float Ncp = y[1] * 60.0f / (2.0f * (float) M_PI);
    float Ncr = Ncp / std::sqrtf(p->theta);
    float pr = y[2]/p->Patm;
    float Uc = p->dc * Ncr * (float) M_PI/60.0f;
    float M = Uc / std::sqrtf(p->gamma * p->airgasconstant * p->Ta);
    float phimax = p->phimax4 * std::powf(M, 4.0f) + p->phimax3 * std::powf(M, 3.0f) + p->phimax2 * std::powf(M, 2.0f) + p->phimax1 * M + p->phimax0;
    float beta = p->beta2 * std::powf(M, 2.0f) + p->beta1 * M + p->beta0;
    float psimax = p->psimax5 * std::powf(M, 5.0f) + p->psimax4 * std::powf(M, 4.0f) + p->psimax3 * std::powf(M, 3.0f) + p->psimax2 * std::powf(M, 2.0f) + p->psimax1 * M + p->psimax0;
    float Psi = p->Cp * p->Ta * (std::powf(pr, (p->gamma - 1.0f)/p->gamma) - 1.0f) / (std::powf(Uc, 2.0f) / 2.0f);
    float Phi = phimax * (1.0f - std::expf(beta * (Psi / psimax - 1.0f)));
    float Wcr = Phi * p->rho * (float) M_PI/4.0f * p->dc * p->dc * Uc;
    float Wcp = Wcr / std::sqrtf(p->theta);
    float Vcm = 0.6814f * v + 33.8741f + p->Kp * (p->c25 * v - Wcp) + p->Ki * y[3];

    dy[0] = -p->mu1 * y[0] + p->mu2 * y[2] + p->mu3 - p->mu4 * v;
    dy[1] = p->c13 * Vcm - p->c9 * y[1] - p->c10 / y[1] * (std::powf(y[2] / p->c11, p->c12) - 1.0f) * Wcp;
    dy[2] = p->c14 * (1.0f + p->c15 * (std::powf(y[2]/p->c11, p->c12) - 1.0f)) * (Wcp - p->c16 * (y[2] - y[0]));
    dy[3] = Wcpref - Wcp;
}

__host__
void threaded_diffeq(const void* param, diffeq_step_fn sf, diffeq_fn f, float* y, float* disturbances, const float* y0, const float* kappa, const float step, size_t n_steps, size_t thread_count, size_t kappa_count, size_t disturbance_count, size_t n) {
    std::thread *threads = new std::thread[thread_count];
    auto lambda = [=](size_t kappa_start, size_t kappa_end) {
        float* local_buffer = new float[n * 5];
        for (int i=kappa_start; i<kappa_end; ++i) {
            for (int j=0; j<disturbance_count; ++j) {
                for (int k=0; k<n; ++k) {
                    y[k + (i*disturbance_count + j)*n_steps*n] = y0[k] + disturbances[k + (i*disturbance_count + j)*n_steps*n] - 1;
                }
                for (int k=0; k<n_steps-1; ++k) {
                    sf(param, f, y + ((i*disturbance_count + j)*n_steps + k) * n, local_buffer, y + ((i*disturbance_count + j)*n_steps + k + 1) * n, step, kappa[i], n);
                    for (int l=0; l<n; ++l) {
                        y[l + ((i*disturbance_count + j)*n_steps + k + 1)*n] += disturbances[l + ((i*disturbance_count + j)*n_steps + k + 1)*n] - 1;
                    }
                }
            }
        }
        delete[] local_buffer;
    };
    for (int i=0; i<thread_count-1; ++i) {
        threads[i] = std::thread(lambda, i*kappa_count/thread_count, (i+1)*kappa_count/thread_count);
    }
    threads[thread_count-1] = std::thread(lambda, (thread_count-1)*kappa_count/thread_count, kappa_count);
    for (int i=0; i<thread_count; ++i) {
        threads[i].join();
    }
    delete[] threads;
}

__host__
void cpu_rk4_timing(const param* p, bool* valid, float* disturbances, const float* y0, const float* kappa, const float step, size_t n_steps, size_t thread_count, size_t kappa_count, size_t disturbance_count) {
    std::thread *threads = new std::thread[thread_count];
    auto lambda = [=](size_t kappa_start, size_t kappa_end) {
        float* local_buffer = new float[4*5];
        float* local_y = new float[4];
        float lambda, Ncp, Ncr, pr, Uc, M, phimax, beta, psimax, Psi, Phi, Wcr, Wcp;
        for (int i=kappa_start; i<kappa_end; ++i) {
            for (int j=0; j<disturbance_count; ++j) {
                valid[i + kappa_count * j] = true;
                for (int k=0; k<4; ++k) {
                    local_y[k] = y0[k] + disturbances[k + (i * disturbance_count + j) * n_steps * 4];
                }
                for (int k=0; k<n_steps-1; ++k) {
                    rk4_step(p, fuelcell_step, local_y, local_buffer, local_y, step, kappa[i], 4);
                
                    lambda = p->c23 * (local_y[2] - local_y[0]) / (p->c24 * (p->base_voltage + p->voltage_offset * kappa[i]));
            
                    if (lambda < 1.9f) {
                        valid[i + kappa_count * j] = false;
                        break;
                    }
                    
                    Ncp = local_y[1] * 60.0f / (2.0f * (float) M_PI);
                    Ncr = Ncp / std::sqrtf(p->theta);
                    pr = local_y[2]/p->Patm;
                    Uc = p->dc * Ncr * (float) M_PI/60.0f;
                    M = Uc / std::sqrtf(p->gamma * p->airgasconstant * p->Ta);
                    phimax = p->phimax4 * std::powf(M, 4.0f) + p->phimax3 * std::powf(M, 3.0f) + p->phimax2 * std::powf(M, 2.0f) + p->phimax1 * M + p->phimax0;
                    beta = p->beta2 * std::powf(M, 2.0f) + p->beta1 * M + p->beta0;
                    psimax = p->psimax5 * std::powf(M, 5.0f) + p->psimax4 * std::powf(M, 4.0f) + p->psimax3 * std::powf(M, 3.0f) + p->psimax2 * std::powf(M, 2.0f) + p->psimax1 * M + p->psimax0;
                    Psi = p->Cp * p->Ta * (std::powf(pr, (p->gamma - 1.0f)/p->gamma) - 1.0f) / (std::powf(Uc, 2.0f) / 2.0f);
                    Phi = phimax * (1.0f - std::expf(beta * (Psi / psimax - 1.0f)));
                    Wcr = Phi * p->rho * (float) M_PI/4.0f * p->dc * p->dc * Uc;
                    Wcp = Wcr / std::sqrtf(p->theta);

                    if (local_y[2]/101325.0f >= 50.0f * Wcp - 0.1f || local_y[2]/101325.0f <= 15.27f * Wcp + 0.6f) {
                        valid[i + kappa_count * j] = false;
                        break;
                    }
                    for (int l=0; l<4; ++l) {
                        local_y[l] += disturbances[l + ((i*disturbance_count + j)*n_steps + k + 1)*4];
                    }
                }
            }
        }
        delete[] local_y;
        delete[] local_buffer;
    };
    for (int i=0; i<thread_count-1; ++i) {
        threads[i] = std::thread(lambda, i*kappa_count/thread_count, (i+1)*kappa_count/thread_count);
    }
    threads[thread_count-1] = std::thread(lambda, (thread_count-1)*kappa_count/thread_count, kappa_count);
    for (int i=0; i<thread_count; ++i) {
        threads[i].join();
    }
    delete[] threads;
}

__global__
void find_kappa_rk4_kernel(param* p, float* kappa, bool* valid, float* disturbances, float time_step, size_t steps, float* x0, size_t num_kappa) {
    extern __shared__ float s[];
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidx + tidy * blockDim.x * gridDim.x;

    float local_kappa = kappa[tidx];
    float* local_y = s + 24 * (threadIdx.x + blockDim.x * threadIdx.y);
    float* local_disturbances = disturbances + tid * steps * 4;
    
    for (int i=0; i<4; ++i) {
        local_y[i] = x0[i] + local_disturbances[i];
    }
    local_disturbances += 4;

    float* local_buffer = &local_y[4];
    float lambda, Ncp, Ncr, pr, Uc, M, phimax, beta, psimax, Psi, Phi, Wcr, Wcp;

    valid[tid] = true;
    
    for (int i=0; i<steps-1; ++i) {
        rk4_step(p, fuelcell_step, local_y, local_buffer, local_y, time_step, local_kappa, 4);

        lambda = p->c23 * (local_y[2] - local_y[0]) / (p->c24 * (p->base_voltage + p->voltage_offset * local_kappa));
        
        if (lambda < 1.9f) {
            valid[tid] = false;
            break;
        }
        
        Ncp = local_y[1] * 60.0f / (2.0f * (float) M_PI);
        Ncr = Ncp / std::sqrtf(p->theta);
        pr = local_y[2]/p->Patm;
        Uc = p->dc * Ncr * (float) M_PI/60.0f;
        M = Uc / std::sqrtf(p->gamma * p->airgasconstant * p->Ta);
        phimax = p->phimax4 * std::powf(M, 4.0f) + p->phimax3 * std::powf(M, 3.0f) + p->phimax2 * std::powf(M, 2.0f) + p->phimax1 * M + p->phimax0;
        beta = p->beta2 * std::powf(M, 2.0f) + p->beta1 * M + p->beta0;
        psimax = p->psimax5 * std::powf(M, 5.0f) + p->psimax4 * std::powf(M, 4.0f) + p->psimax3 * std::powf(M, 3.0f) + p->psimax2 * std::powf(M, 2.0f) + p->psimax1 * M + p->psimax0;
        Psi = p->Cp * p->Ta * (std::powf(pr, (p->gamma - 1.0f)/p->gamma) - 1.0f) / (std::powf(Uc, 2.0f) / 2.0f);
        Phi = phimax * (1.0f - std::expf(beta * (Psi / psimax - 1.0f)));
        Wcr = Phi * p->rho * (float) M_PI/4.0f * p->dc * p->dc * Uc;
        Wcp = Wcr / std::sqrtf(p->theta);

        if (local_y[2]/101325.0f >= 50.0f * Wcp - 0.1f || local_y[2]/101325.0f <= 15.27f * Wcp + 0.6f) {
            // printf("%d\n", i);
            valid[tid] = false;
            break;
        }

        for (int j=0; j<4; ++j) {
            local_y[j] += local_disturbances[j];
        }
        local_disturbances += 4;
    }
}

__global__
void setup_rand_kernel(curandState *state, unsigned long long seed) {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    curand_init(seed, tid, 0, state + tid);
}

__global__
void generate_disturbances_kernel(float* disturbances, curandState* state, size_t n_steps) {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidx + tidy * blockDim.x;
    curandState local_state = state[tid];
    
    float r;

    for (int i=0; i<n_steps; ++i) {
        if (tidx == 0 || tidx == 2) {
            r = curand_uniform(&local_state) * 100 - 50;
        }
        else if (tidx == 1) {
            r = curand_uniform(&local_state) * 20 - 10;
        }
        else {
            r = 0;
        }

        disturbances[tidy * n_steps + i*4 + tidx] = r;
    }

    state[tid] = local_state;
}

__host__
float find_valid_kappa(float* kappa, size_t num_kappa, size_t num_disturbances, bool* results) {
    bool* valid = new bool[num_kappa];
    for (size_t i=0; i<num_kappa; ++i) {
        valid[i] = true;
    }
    for (size_t i=0; i<num_kappa; ++i) {
        for (size_t k=0; k<num_disturbances; ++k) {
            if (!results[i + k * num_kappa]) {
                valid[i] = false;
            }
        }
    }
    size_t valid_idx = 0;
    for (size_t i=0; i<num_kappa; ++i) {
        if (valid[i]) {
            valid_idx = i;
        }
    }
    return kappa[valid_idx];
}