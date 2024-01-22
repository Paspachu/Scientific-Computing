from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

# u_xx + u_yy = 0
# u(x, 0) = x^2 - x
# All other boundaries are zero
# u(x, 1) = u(0, y) = u(1, y) = 0

x0 = 0
xN = 1
y0 = 0
yN = 1

N = 101
x = np.linspace(x0, xN, N)
y = np.linspace(y0, yN, N)
dx = x[1] - x[0]
print("N = {}".format(N))
print("dx = {}".format(dx))

# Boundary condition u(x, 0)
def a(x):
    return x ** 2 - x

# Jacobi Iteration
U = np.zeros((N, N))
U[0, :] = a(x)
Unew = U.copy()

max_steps = 6000
for k in range(max_steps):
    for m in range(1, N-1):
        for n in range(1, N-1):
            Unew[m, n] = (U[m, n - 1] + U[m - 1, n] + U[m + 1, n] + U[m, n + 1]) / 4
    U[:] = Unew[:]

# Gauss-Seidel Iteration
U = np.zeros((N, N))
U[0, :] = a(x)
Unew = U.copy()

max_steps = 3000
for k in range(max_steps):
    for m in range(1, N-1):
        for n in range(1, N-1):
            U[m, n] = (U[m, n - 1] + U[m - 1, n] + U[m + 1, n] + U[m, n + 1]) / 4

# These vectors are just to plot the boundary conditions
zero_vector = np.zeros_like(x)
one_vector = np.ones_like(x)

X, Y = np.meshgrid(x, y)

ax = plt.axes(projection='3d')
# Plot the solution
ax.plot_surface(X, Y, U)
# u(0, y) = 0
ax.plot3D(zero_vector, y, zero_vector, 'r')
# u(x, 1) = 0
ax.plot3D(x, one_vector, zero_vector, 'r')
# u(1, y) = 0
ax.plot3D(one_vector, y, zero_vector, 'r')
# u(x, 0) = a(x)
ax.plot3D(x, zero_vector, a(x), 'r')
plt.show()