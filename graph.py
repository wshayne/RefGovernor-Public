import matplotlib.pyplot as plt
import numpy as np

FOLDER = "results/"

desired_setpoints = np.loadtxt(f"{FOLDER}desired_setpoint.csv")
real_setpoints = np.loadtxt(f"{FOLDER}setpoint_output.csv")
lambda_o2 = np.loadtxt(f"{FOLDER}lambda_o2_output.csv")
kappa_history = np.loadtxt(f"{FOLDER}kappa_output.csv")
x = [0.01 * i for i in range(len(real_setpoints))]
constraint = [1.9 for _ in range(len(lambda_o2))]

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(x, real_setpoints, label="Modified Setpoints")
axes[0].plot(x, desired_setpoints, label="Reference Setpoints")
axes[1].plot(x, lambda_o2, label="O2 Excess Ratio")
axes[1].plot(x, constraint, label="Constraint", linestyle="--", color="r")
axes[2].plot(x, kappa_history, label="Kappa Values")
axes[0].legend()
axes[1].legend()
axes[0].set_ylabel("Stack Current (A)")
axes[1].set_ylabel("O2 Excess Ratio")
axes[2].set_ylabel("Kappa")
fig.supxlabel("Time (s)")

plt.savefig(f"{FOLDER}sim_output.png", dpi=1200)
plt.show()