import numpy as np
import matplotlib.pyplot as plt

# u_t = -c * u_x
# u(0, x) = sin(pi * x)
# u(t, 0) = u(t, 1)
# u(t, x) = u0(x)

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
Nt = 201
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

A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
A = (-c / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:, 0] = u0(x)

# Forward Euler
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

fig = plt.figure()
fig.suptitle("Forward Euler")
for j, tval in enumerate(t_plot):
    k = round(tval / dt)
    ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
    ax.set_title("t = {:.2f}".format(tval))
    ax.plot(x, U[:, k])
plt.subplots_adjust(hspace=0.5)
plt.show()

# Backward Euler
A_backward = np.eye(Nx - 1) - dt * A
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(A_backward, U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

fig = plt.figure()
fig.suptitle("Backward Euler")
for j, tval in enumerate(t_plot):
    k = round(tval / dt)
    ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
    ax.set_title("t = {:.2f}".format(tval))
    ax.plot(x, U[:, k])
    ax.set_ylim((-1, 1))
plt.subplots_adjust(hspace=0.5)
plt.show()

# Midpoint
U[:-1, 1:2] = U[:-1, 0:1] + dt * A @ U[:-1, 0:1]
for k in range(Nt - 2):
    U[:-1, (k + 2):(k + 3)] = U[:-1, k:(k + 1)] + 2 * dt * A @ U[:-1, (k + 1):(k + 2)]
U[-1, :] = U[0, :]

print("Ratio = {}".format(np.abs(c * dt / dx)))

fig = plt.figure()
fig.suptitle("Midpoint Method")
for j, tval in enumerate(t_plot):
    k = round(tval / dt)
    ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
    ax.set_title("t = {:.2f}".format(tval))
    ax.plot(x, U[:, k])
    ax.set_ylim((-1, 1))
plt.subplots_adjust(hspace=0.5)
plt.show()

vals, _ = np.linalg.eig(A)
print(sorted(dt * vals))