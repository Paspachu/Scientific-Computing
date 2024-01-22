#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
"""Problem 1"""
c = 3
x0 = -10
xf = 10
t0 = 0
tf = 3

#%%
Nx = 81
x = np.linspace(x0, xf, Nx)
dx = x[1] - x[0]
# print("Nx = {}".format(Nx))
# print("dx = {}".format(dx))

#%%
def u0(x):
    if x >= -1 and  x <= 0:
        return x + 1
    elif x > 0 and x <= 1:
        return 1 - x
    else:
        return 0

def plot_method(title, x, U, dt):
    plot_rows = 3
    plot_cols = 3
    num_plots = plot_rows * plot_cols
    t_plot = np.linspace(t0, tf, num_plots)

    fig = plt.figure()
    fig.suptitle(title)
    for j, tval in enumerate(t_plot):
        k = round(tval / dt)
        ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
        ax.set_title("t = {:.2f}".format(tval))
        ax.plot(x, U[:, k])
    plt.subplots_adjust(hspace=0.5)
    plt.show()

#%%
Nt = 31
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
# print("Nt = {}".format(Nt))
# print("dt = {}".format(dt))

#%%
A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
A = (-c / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:, 0] = np.array([u0(x_val) for x_val in x])

#%%
# Forward Euler
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

# plot_method("Forward Euler", x, U, dt)

A1 = U[76, -1]

#%%
# Trapezoidal
I = np.eye(Nx - 1)
A_lhs = (I - (dt / 2) * A)
A_rhs = (I + (dt / 2) * A)
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

# plot_method("Trapezoidal", x, U, dt)

A3 = U[76, -1]

#%%
# Midpoint
U[:-1, 1:2] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, 0:1])
for k in range(Nt - 2):
    U[:-1, (k + 2):(k + 3)] = U[:-1, k:(k + 1)] + 2 * dt * A @ U[:-1, (k + 1):(k + 2)]
U[-1, :] = U[0, :]

# plot_method("Midpoint", x, U, dt)

A5 = U[76, -1]

#%%
# Lax-Friedrichs
B = np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)
B[0, -1] = 1
B[-1, 0] = 1
B = 0.5 * B

for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = B @ U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

# plot_method("Lax-Friedrichs", x, U, dt)

A7 = U[76, -1]

#%%
Nt = 301
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
# print("Nt = {}".format(Nt))
# print("dt = {}".format(dt))

#%%
A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
A = (-c / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:, 0] = np.array([u0(x_val) for x_val in x])

#%%
# Forward Euler
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

# plot_method("Forward Euler", x, U, dt)

A2 = U[76, -1]

#%%
# Trapezoidal
I = np.eye(Nx - 1)
A_lhs = (I - (dt / 2) * A)
A_rhs = (I + (dt / 2) * A)
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

# plot_method("Trapezoidal", x, U, dt)

A4 = U[76, -1]

#%%
# Midpoint
U[:-1, 1:2] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, 0:1])
for k in range(Nt - 2):
    U[:-1, (k + 2):(k + 3)] = U[:-1, k:(k + 1)] + 2 * dt * A @ U[:-1, (k + 1):(k + 2)]
U[-1, :] = U[0, :]

# plot_method("Midpoint", x, U, dt)

A6 = U[76, -1]

#%%
# Lax-Friedrichs
B = np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)
B[0, -1] = 1
B[-1, 0] = 1
B = 0.5 * B

for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = B @ U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

# plot_method("Lax-Friedrichs", x, U, dt)

A8 = U[76, -1]

#%%
"""Problem 2"""
x0 = 0
xf = 2 * np.pi
t0 = 0
tf = 8

#%%
Nx = 101
x = np.linspace(x0, xf, Nx)
dx = x[1] - x[0]
# print("Nx = {}".format(Nx))
# print("dx = {}".format(dx))

#%%
def c(x):
    return 0.2 + np.sin(x-1) ** 2

def u0(x):
    return np.cos(x)

def plot_method(title, x, U, dt):
    plot_rows = 3
    plot_cols = 3
    num_plots = plot_rows * plot_cols
    t_plot = np.linspace(t0, tf, num_plots)

    fig = plt.figure()
    fig.suptitle(title)
    for j, tval in enumerate(t_plot):
        k = round(tval / dt)
        ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
        ax.set_title("t = {:.2f}".format(tval))
        ax.plot(x, U[:, k])
    plt.subplots_adjust(hspace=0.5)
    plt.show()

#%%
Nt = 9
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
# print("Nt = {}".format(Nt))
# print("dt = {}".format(dt))

#%%
A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
for i, row in enumerate(A):
    A[i, :] = c(x[i]) * row
A = (-1 / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:, 0] = u0(x)

#%%
# Backward Euler
A_backward = np.eye(Nx - 1) - dt * A
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(A_backward, U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

# plot_method("Backward Euler", x, U, dt)

A9 = U[50, 4]

#%%
# Trapezoidal
I = np.eye(Nx - 1)
A_lhs = (I - (dt / 2) * A)
A_rhs = (I + (dt / 2) * A)
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

# plot_method("Trapezoidal", x, U, dt)

A11 = U[50, 4]

#%%
Nt = 161
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
# print("Nt = {}".format(Nt))
# print("dt = {}".format(dt))

#%%
A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
for i, row in enumerate(A):
    A[i, :] = c(x[i]) * row
A = (-1 / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:, 0] = u0(x)

#%%
# Midpoint
I = np.eye(Nx - 1)
A_lhs = (I - (dt / 2) * A)
A_rhs = (I + (dt / 2) * A)
U[:-1, 1:2] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, 0:1])
for k in range(Nt - 2):
    U[:-1, (k + 2):(k + 3)] = U[:-1, k:(k + 1)] + 2 * dt * A @ U[:-1, (k + 1):(k + 2)]
U[-1, :] = U[0, :]

# plot_method("Midpoint", x, U, dt)

A13 = U[50, 80]

#%%
# Lax-Friedrichs
B = np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)
B[0, -1] = 1
B[-1, 0] = 1
B = 0.5 * B

for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = B @ U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

# plot_method("Lax-Friedrichs", x, U, dt)

A15 = U[50, 80]

#%%
Nt = 801
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
# print("Nt = {}".format(Nt))
# print("dt = {}".format(dt))

#%%
A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
for i, row in enumerate(A):
    A[i, :] = c(x[i]) * row
A = (-1 / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:, 0] = u0(x)

#%%
# Backward Euler
A_backward = np.eye(Nx - 1) - dt * A
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(A_backward, U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

# plot_method("Backward Euler", x, U, dt)

A10 = U[50, 400]

#%%
# Trapezoidal
I = np.eye(Nx - 1)
A_lhs = (I - (dt / 2) * A)
A_rhs = (I + (dt / 2) * A)
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

# plot_method("Trapezoidal", x, U, dt)

A12 = U[50, 400]

#%%
# Midpoint
U[:-1, 1:2] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, 0:1])
for k in range(Nt - 2):
    U[:-1, (k + 2):(k + 3)] = U[:-1, k:(k + 1)] + 2 * dt * A @ U[:-1, (k + 1):(k + 2)]
U[-1, :] = U[0, :]

# plot_method("Midpoint", x, U, dt)

A14 = U[50, 400]

#%%
# Lax-Friedrichs
B = np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)
B[0, -1] = 1
B[-1, 0] = 1
B = 0.5 * B

for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = B @ U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

# plot_method("Lax-Friedrichs", x, U, dt)

A16 = U[50, 400]

# %%
"""Problem 3"""
c = 1
x0 = 0
xf = 25
t0 = 0
tf = 17

#%%
Nx = 501
x = np.linspace(x0, xf, Nx)
dx = x[1] - x[0]
# print("Nx = {}".format(Nx))
# print("dx = {}".format(dx))

#%%
def u0(x):
    return np.exp(-20 * (x-2) ** 2) + np.exp(-(x-5) ** 2)

def true_solution(t, x):
    return u0(x - t) % 25

def plot_method(title, x, U, dt):
    plot_rows = 3
    plot_cols = 3
    num_plots = plot_rows * plot_cols
    t_plot = np.linspace(t0, tf, num_plots)

    fig = plt.figure()
    fig.suptitle(title)
    for j, tval in enumerate(t_plot):
        k = round(tval / dt)
        ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
        ax.set_title("t = {:.2f}".format(tval))
        ax.plot(x, U[:, k])
    plt.subplots_adjust(hspace=0.5)
    plt.show()

#%%
Nt = 375
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
# print("Nt = {}".format(Nt))
# print("dt = {}".format(dt))

#%%
# plot_rows = 3
# plot_cols = 3
# num_plots = plot_rows * plot_cols
# t_plot = np.linspace(t0, tf, num_plots)

# fig = plt.figure()
# fig.suptitle("True Solution")
# for j, tval in enumerate(t_plot):
#     k = round(tval / dt)
#     ax = fig.add_subplot(plot_rows, plot_cols, j + 1)
#     ax.set_title("t = {:.2f}".format(tval))
#     ax.plot(x, true_solution(tval, x))
#     ax.set_ylim((-1, 1))
# plt.subplots_adjust(hspace=0.5)
# plt.show()

#%%
A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
A = (-c / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:, 0] = np.array([u0(x_val) for x_val in x])

#%%
# Lax-Friedrichs
B = np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)
B[0, -1] = 1
B[-1, 0] = 1
B = 0.5 * B

for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = B @ U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

# plot_method("Lax-Friedrichs", x, U, dt)

A17 = U[380, -1]

#%%
# Midpoint
I = np.eye(Nx - 1)
A_lhs = (I - (dt / 2) * A)
A_rhs = (I + (dt / 2) * A)
U[:-1, 1:2] = np.linalg.solve(A_lhs, A_rhs @ U[:-1, 0:1])
for k in range(Nt - 2):
    U[:-1, (k + 2):(k + 3)] = U[:-1, k:(k + 1)] + 2 * dt * A @ U[:-1, (k + 1):(k + 2)]
U[-1, :] = U[0, :]

# plot_method("Midpoint", x, U, dt)

A18 = U[380, -1]

#%%
