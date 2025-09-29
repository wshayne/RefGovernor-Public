import subprocess, os, sys, numpy as np, matplotlib.pyplot as plt

N_KAPPA = 32
N_STEPS = 1024
T_STEP = 0.01
DISTURBANCES = 1024
SIM_STEPS = 1024

if __name__ == "__main__":
    try:
        os.mkdir("results")
    except:
        pass

    try:
        subprocess.run(["nvcc", "-o", "sim.exe", "main.cu", "diffeq.cu", "functions.cpp"])
        subprocess.run(["./sim.exe", f"{N_KAPPA}", f"{N_STEPS}", f"{T_STEP}", f"{DISTURBANCES}", f"{SIM_STEPS}"])
    except Exception as e:
        print(e)
        sys.exit()
    
    desired_setpoints = np.loadtxt("results/desired_setpoint.csv")
    real_setpoints = np.loadtxt("results/setpoint_output.csv")
    lambda_o2 = np.loadtxt("results/lambda_o2_output.csv")
    kappa_history = np.loadtxt("results/kappa_output.csv")
    x = [0.01 * i for i in range(len(real_setpoints))]

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(x, real_setpoints, label="Real Setpoints")
    axes[0].plot(x, desired_setpoints, label="Desired Setpoints")
    axes[1].plot(x, lambda_o2, label="Excess O2 Ratio")
    axes[2].plot(x, kappa_history, label="Kappa Values")
    axes[0].legend()
    axes[0].set_ylabel("Stack Current (A)")
    axes[1].set_ylabel("O2 Excess Ratio")
    axes[2].set_ylabel("Kappa")
    fig.supxlabel("Time (s)")
    plt.savefig("results/sim_output.png")

    plt.show()