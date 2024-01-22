import numpy as np
import matplotlib.pyplot as plt

# u_t = -c * u_x
# u(0, x) = u0(x)
# u(t, 0) = u(t, 1)
# u(t, x) = true_solution

c = 1

def u0(x):
    return np.exp(-50 * (x - 0.5) ** 2)

def true_solution(t, x):
    return u0(np.mod(x - c * t, 1))

x0 = 0
xf = 1
Nx = 101
x = np.linspace(x0, xf, Nx)
dx = x[1] - x[0]
print("Nx = {}".format(Nx))
print("dx = {}".format(dx))

t0 = 0
tf = 1
Nt = 101
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
print("Nt = {}".format(Nt))
print("dt = {}".format(dt))

plot_rows = 3
plot_cols = 3
num_plots = plot_rows * plot_cols
t_plot = np.linspace(t0, tf, num_plots)

fig = plt.figure()
fig.suptitle("True Solution")
for j, tval in enumerate(t_plot):
    k = round(tval / dt)
    ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
    ax.set_title("t = {:.2f}".format(tval))
    ax.plot(x, true_solution(tval, x))
    ax.set_ylim((-1, 1))
plt.subplots_adjust(hspace=0.5)
plt.show()

# Lax-Friedrichs
A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
A = (-c / (2 * dx)) * A

B = np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)
B[0, -1] = 1
B[-1, 0] = 1
B = 0.5 * B

U = np.zeros((Nx, Nt))
U[:, 0] = u0(x)

for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = B @ U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

fig = plt.figure()
fig.suptitle("Lax-Friedrichs")
for j, tval in enumerate(t_plot):
    k = round(tval / dt)
    ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
    ax.set_title("t = {:.2f}".format(tval))
    ax.plot(x, U[:, k])
    ax.set_ylim((-1, 1))
plt.subplots_adjust(hspace=0.5)
plt.show()