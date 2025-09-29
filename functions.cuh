#ifndef FUNCTIONS_REFGOV_H
#define FUNCTIONS_REFGOV_H
#define _USE_MATH_DEFINES
#include <cmath>
#include <thread>

struct param {
    float base_voltage;
    float voltage_offset;
    float fc_numberofcells;
    float fc_cathode_volume;
    float fc_inlet_flow_constant;
    float Cp;
    float gamma;
    float rho;
    float airgasconstant;
    float oxygenmolarmass;
    float nitrogenmolarmass;
    float faradays;
    float universalgasconstant;
    float dc;
    float Ta;
    float cp_inertia;
    float phimax4;
    float phimax3;
    float phimax2;
    float phimax1;
    float phimax0;
    float beta2;
    float beta1;
    float beta0;
    float psimax5;
    float psimax4;
    float psimax3;
    float psimax2;
    float psimax1;
    float psimax0;
    float cm_kv;
    float cm_kt;
    float cm_R;
    float im_volume;
    float om_Cd;
    float om_At;
    float Ma;
    float xo2;
    float watm;
    float Tst;
    float Tatm;
    float Psat_Tst;
    float Patm;
    float etacm;
    float etacp;
    float c1;
    float c2;
    float c3;
    float c7;
    float c8;
    float c9;
    float c10;
    float c11;
    float c12;
    float c13;
    float c14;
    float c15;
    float c16;
    float c20;
    float c23;
    float c24;
    float c25;
    float kap;
    float mu1;
    float mu2;
    float mu3;
    float mu4;
    float theta;
    float Kp;
    float Ki;
    param();
};

template<int row, int col>
void discrete_step(float y[row][col], float* dy, int t_ind, float v, param p, float T_s) {
    float Wcpref = p.c25 * v;
    float Ncp = y[1][t_ind-1] * 60 / (2 * M_PI);
    float Ncr = Ncp / std::sqrt(p.theta);
    float pr = y[2][t_ind-1]/p.Patm;
    float Uc = p.dc * Ncr * M_PI/60;
    float M = Uc / std::sqrt(p.gamma * p.airgasconstant * p.Ta);
    float phimax = p.phimax4 * std::pow(M, 4) + p.phimax3 * std::pow(M, 3) + p.phimax2 * std::pow(M, 2) + p.phimax1 * M + p.phimax0;
    float beta = p.beta2 * std::pow(M, 2) + p.beta1 * M + p.beta0;
    float psimax = p.psimax5 * std::pow(M, 5) + p.psimax4 * std::pow(M, 4) + p.psimax3 * std::pow(M, 3) + p.psimax2 * std::pow(M, 2) + p.psimax1 * M + p.psimax0;
    float Psi = p.Cp * p.Ta * (std::pow(pr, (p.gamma - 1.0)/p.gamma) - 1.0) / (std::pow(Uc, 2) / 2.0);
    float Phi = phimax * (1.0 - std::exp(beta * (Psi / psimax - 1.0)));
    float Wcr = Phi * p.rho * M_PI/4 * p.dc * p.dc * Uc;
    float Wcp = Wcr / std::sqrt(p.theta);
    float Vcm = 0.6814 * v + 33.8741 + p.Kp * (p.c25 * v - Wcp) + p.Ki * y[3][t_ind-1];

    dy[0] = -p.mu1 * y[0][t_ind-1] + p.mu2 * y[2][t_ind-1] + p.mu3 - p.mu4 * v;
    dy[1] = p.c13 * Vcm - p.c9 * y[1][t_ind-1] - p.c10 / y[1][t_ind-1] * (std::pow(y[2][t_ind-1] / p.c11, p.c12) - 1.0) * Wcp;
    dy[2] = p.c14 * (1.0 + p.c15 * (std::pow(y[2][t_ind-1]/p.c11, p.c12) - 1.0)) * (Wcp - p.c16 * (y[2][t_ind-1] - y[0][t_ind-1]));
    dy[3] = Wcpref - Wcp;

    for (int i=0; i<4; ++i) {
        y[i][t_ind] = y[i][t_ind-1] + T_s * dy[i];
    }
}

template<int row, int col>
void run_forward_euler(float y[row][col], float* dy, int t_ind, float v, param p, float T_s, int nt) {
    for (int i=0; i<nt; ++i) {
        discrete_step<row, col>(y, dy, t_ind, v, p, T_s);
        t_ind++;
    }
}

void run_forward_euler_kappa_range(float* y, float* y0, float* dy, param p, float T_s, float base_voltage, int nt, float* kappa, int kappa_start, int kappa_end);

void run_forward_euler_kappa(float* y, float* y0, float* dy, param p, float T_s, float base_voltage, int nt, float* kappa, int thread_count, int kappa_count);

#endif