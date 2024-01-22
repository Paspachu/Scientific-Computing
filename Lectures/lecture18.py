import numpy as np

# u_t = u_xx
# u(t, 0) = 0
# u(t, 1) = 0
# u(0, x) = sin(pi * x)

Nx = 11
x0 = 0
xf = 1
x = np.linspace(x0, xf, Nx)
dx = x[1] - x[0]

t0 = 0
tf = 0.5
Nt = 101
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]

U = np.zeros((Nx, Nt))
U[:, 0] = np.sin(np.pi * x)
U[0, :] = 0
U[-1, :] = 0

A = (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2

def f(t, u):
    return A @ u

for k in range(Nt - 1):
    U[1:-1, (k + 1):(k + 2)] = U[1:-1, k:(k + 1)] + dt * f(t[k], U[1:-1, k:(k + 1)])

print(U)