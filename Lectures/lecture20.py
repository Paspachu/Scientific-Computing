import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# u_t = u_xx
# u(t, 0) = 0
# u(t, 1) = 0
# u(0, x) = sin(pi * x)
# u(t, x) = e^(-pi^2*t) * sin(pi * x)

Nx = 101
x0 = 0
xf = 1
x = np.linspace(x0, xf, Nx)
dx = x[1] - x[0]
print("Nx = {}".format(Nx))
print("dx = {}".format(dx))

t0 = 0
tf = 0.5
Nt = 101
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
print("Nt = {}".format(Nt))
print("dt = {}".format(dt))

U = np.zeros((Nx, Nt))
U[:, 0] = np.sin(np.pi * x)
U[0, :] = 0
U[-1, :] = 0

# Crank-Nicolson Method
# Solve with Trapezoidal Method

A = (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2
# (I - dt/2 * A) * u_(k+1) = (I + dt / 2 * A) * u_k
I = np.eye(Nx - 2)
A_lhs = (I - (dt / 2) * A)
A_rhs = (I + (dt / 2) * A)

for k in range(Nt - 1):
    U[1:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U[1:-1, k:(k + 1)])

T, X = np.meshgrid(t, x)

ax = plt.axes(projection='3d')
ax.plot_surface(T, X, U)
plt.show()

def true_solution(t, x):
    return np.exp(-(np.pi ** 2) * t) * np.sin(np.pi * x)

err = U - true_solution(T, X)
ax = plt.axes(projection='3d')
ax.plot_surface(T, X, err)
plt.show()

global_err = np.max(np.abs(err))
print("Global Error = {}".format(global_err))