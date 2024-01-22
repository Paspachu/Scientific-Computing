from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

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

# Solve Au = b
N_total = N * N
print("N_total = {}".format(N_total))
print("Entries in matrix = {}".format(N_total ** 2))
A = np.zeros((N_total, N_total))
b = np.zeros((N_total, 1))

def point2ind(m, n):
    return n * N + m

for n in range(N):
    for m in range(N):
        k = point2ind(m, n)
        if n == 0:
            A[k, k] = 1
            b[k] = a(x[m])
        elif m == 0 or m == N-1 or n == N-1:
            A[k, k] = 1
            b[k] = 0
        else:
            A[k, k] = -4 / dx ** 2
            A[k, k + 1] = 1 / dx ** 2
            A[k, k - 1] = 1 / dx ** 2
            A[k, k + N] = 1 / dx ** 2
            A[k, k - N] = 1 / dx ** 2
            b[k] = 0 # f(x[m], y[n])

u = np.linalg.solve(A, b)
U = u.reshape((N, N))


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