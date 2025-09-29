#define _USE_MATH_DEFINES
#include "functions.cuh"
#include <cmath>

param::param() {
    base_voltage = 160.0;
    voltage_offset = 60.0;
    fc_numberofcells = 381.0;
    fc_cathode_volume = 0.01;
    fc_inlet_flow_constant = 0.36293861487e-5;
    Cp = 1004;
    gamma = 1.4;
    rho = 1.23;
    airgasconstant = 286.9;
    oxygenmolarmass = 32e-3;
    nitrogenmolarmass = 28e-3;
    faradays = 96485;
    universalgasconstant = 8.31451;
    dc = 9*0.0254;
    Ta = 30 + 273.15;
    cp_inertia = 0.00005;
    phimax4 = -0.00003699056149;
    phimax3 = 0.00027039932787;
    phimax2 = -0.00053623540171;
    phimax1 = -0.00004636845793;
    phimax0 = 0.00221194680963;
    beta2 = 1.76566519765589;
    beta1 = -1.34836801554691;
    beta0 = 2.44418790092002;
    psimax5 = -0.00978754892186;
    psimax4 = 0.10580782154809;
    psimax3 = -0.42936513568305;
    psimax2 = 0.80120999043258;
    psimax1 = -0.68344136821569;
    psimax0 = 0.43331043048640;
    cm_kv = 0.0153;
    cm_kt = 0.0225;
    cm_R = 1.2;
    im_volume = 0.02;
    om_Cd = 0.0124;
    om_At = 0.002;
    Ma = 28.84e-3;
    xo2 = 0.23301;
    watm = 0.009835;
    Tst = 353;
    Tatm = 298;
    Psat_Tst = 47066.5;
    Patm = 101325;
    etacm = 0.98;
    etacp = 0.8;
    c1 = universalgasconstant * Tst * fc_inlet_flow_constant / (oxygenmolarmass * fc_cathode_volume) * xo2/(1 + watm);
    c2 = Psat_Tst;
    c3 = universalgasconstant * Tst / fc_cathode_volume;
    c7 = universalgasconstant * Tst * fc_numberofcells / (4 * faradays * fc_cathode_volume);
    c8 = universalgasconstant * Tst * fc_inlet_flow_constant / (nitrogenmolarmass * fc_cathode_volume) * (1 - xo2)/(1 + watm);
    c9 = etacm * cm_kv * cm_kt / (cm_R * cp_inertia);
    c10 = Cp * Tatm / (cp_inertia * etacp);
    c11 = Patm;
    c12 = (gamma - 1)/gamma;
    c13 = etacm * cm_kt / (cp_inertia * cm_R);
    c14 = universalgasconstant * Tatm / (Ma * im_volume);
    c15 = 1/gamma;
    c16 = fc_inlet_flow_constant;
    c20 = om_Cd * om_At / std::sqrt(universalgasconstant * Tst) * std::pow(gamma, 0.5) * (std::pow(2/ (gamma + 1), (gamma + 1) / (2 * (gamma - 1))));
    c23 = fc_inlet_flow_constant * xo2 / (1 + watm);
    c24 = fc_numberofcells * oxygenmolarmass / (4 * faradays);
    c25 = (1 + watm) * 2 / xo2 * 381 * oxygenmolarmass / (4 * faradays);
    kap = 0.026428;
    mu1 = c1 + c8 + c3 * c20 / kap * 0.88;
    mu2 = c1 + c8;
    mu3 = c2 * c3 * c20 / kap * 0.88;
    mu4 = c7;
    theta = 298.0/288.0;
    Kp = 100;
    Ki = 500;
}

void run_forward_euler_kappa_range(float* y, float* y0, float* dy, param p, float T_s, float base_voltage, int nt, float* kappa, int kappa_start, int kappa_end) {
    for (int i=kappa_start; i<kappa_end; ++i) {
        float v = base_voltage + 30 * kappa[i];
        for (int j=0; j<4; ++j) {
            y[j + i*nt*4] = y0[j];
        }
        for (int j=0; j<nt-1; ++j) {
            int offset = i * nt * 4 + j * 4;
            float Wcpref = p.c25 * v;
            float Ncp = y[1 + offset] * 60 / (2 * M_PI);
            float Ncr = Ncp / std::sqrt(p.theta);
            float pr = y[2 + offset]/p.Patm;
            float Uc = p.dc * Ncr * M_PI/60;
            float M = Uc / std::sqrt(p.gamma * p.airgasconstant * p.Ta);
            float phimax = p.phimax4 * std::pow(M, 4) + p.phimax3 * std::pow(M, 3) + p.phimax2 * std::pow(M, 2) + p.phimax1 * M + p.phimax0;
            float beta = p.beta2 * std::pow(M, 2) + p.beta1 * M + p.beta0;
            float psimax = p.psimax5 * std::pow(M, 5) + p.psimax4 * std::pow(M, 4) + p.psimax3 * std::pow(M, 3) + p.psimax2 * std::pow(M, 2) + p.psimax1 * M + p.psimax0;
            float Psi = p.Cp * p.Ta * (std::pow(pr, (p.gamma - 1.0)/p.gamma) - 1.0) / (std::pow(Uc, 2) / 2.0);
            float Phi = phimax * (1.0 - std::exp(beta * (Psi / psimax - 1.0)));
            float Wcr = Phi * p.rho * M_PI/4 * p.dc * p.dc * Uc;
            float Wcp = Wcr / std::sqrt(p.theta);
            float Vcm = 0.6814 * v + 33.8741 + p.Kp * (p.c25 * v - Wcp) + p.Ki * y[3 + offset];

            dy[0] = -p.mu1 * y[0 + offset] + p.mu2 * y[2 + offset] + p.mu3 - p.mu4 * v;
            dy[1] = p.c13 * Vcm - p.c9 * y[1 + offset] - p.c10 / y[1 + offset] * (std::pow(y[2 + offset] / p.c11, p.c12) - 1.0) * Wcp;
            dy[2] = p.c14 * (1.0 + p.c15 * (std::pow(y[2 + offset]/p.c11, p.c12) - 1.0)) * (Wcp - p.c16 * (y[2 + offset] - y[0 + offset]));
            dy[3] = Wcpref - Wcp;

            for (int k=0; k<4; ++k) {
                y[k + offset + 4] = y[k + offset] + T_s * dy[k];
            }
        }
    }
}

void run_forward_euler_kappa(float* y, float* y0, float* dy, param p, float T_s, float base_voltage, int nt, float* kappa, int thread_count, int kappa_count) {
    std::thread *threads = new std::thread[thread_count];
    for (int i=0; i<thread_count-1; ++i) {
        threads[i] = std::thread(run_forward_euler_kappa_range, y, y0, dy, p, T_s, base_voltage, nt, kappa, i*kappa_count/thread_count, (i+1) * kappa_count/thread_count);
    }
    threads[thread_count-1] = std::thread(run_forward_euler_kappa_range, y, y0, dy, p, T_s, base_voltage, nt, kappa, (thread_count-1) * kappa_count/thread_count, kappa_count);
    for (int i=0; i<thread_count; ++i) {
        threads[i].join();
    }
    delete[] threads;
}