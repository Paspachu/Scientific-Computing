import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# u_t = -2*u_x
# u(t, 0) = u(t, 1)
# u(0, x) = sin(pi * x)
# u(t, x) = sin(pi * (x - 2*t)) <- That's not right!

# Just to make this code operational, I switched the initial
# condition to
# u(0, x) = sin(2 * pi * x)
# and so the true solution really is
# u(t, x) = sin(2 * pi * (x - 2 * t))

c = 2

Nx = 101
x0 = 0
xN = 1
x = np.linspace(x0, xN, Nx)
dx = x[1] - x[0]

t0 = 0
tf = 4
Nt = 1001
t = np.linspace(t0, tf, Nt)
dt = t[1] - t[0]
print("dt = {}".format(dt))

A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
A = (- c / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:-1, 0] = np.sin(2 * np.pi * x[:-1])

# Forward Euler
# for k in range(Nt - 1):
#     U[:-1, (k + 1):(k + 2)] = U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
# U[-1, :] = U[0, :]

# Backward Euler
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = np.linalg.solve(np.eye(Nx - 1) - dt * A, U[:-1, k:(k + 1)])
U[-1, :] = U[0, :]

T, X = np.meshgrid(t, x)

ax = plt.axes(projection='3d')
ax.plot_surface(T, X, U)
plt.title("Solution")
plt.show()

def true_solution(t, x):
    return np.sin(2 * np.pi * (x - c * t)) # <- This was wrong in class

err = U - true_solution(T, X)
ax = plt.axes(projection='3d')
ax.plot_surface(T, X, err)
plt.title("Error")
plt.show()
print("Global Error = {}".format(np.max(np.abs(err))))

vals, _ = np.linalg.eig(A)
print(sorted(dt * vals))