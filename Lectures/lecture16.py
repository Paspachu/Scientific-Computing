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

N = 501
x = np.linspace(x0, xN, N)
y = np.linspace(y0, yN, N)
dx = x[1] - x[0]
print("N = {}".format(N))
print("dx = {}".format(dx))

# Boundary condition u(x, 0)
def a(x):
    return x ** 2 - x

t0 = time.time()
# Set up system Au = b
N_total = (N - 2) * (N - 2)
print("N_total = {}".format(N_total))
print("Entries in matrix = {}".format(N_total ** 2))
# A = np.zeros((N_total, N_total))
# A = scipy.sparse.dok_array((N_total, N_total))
row_vec = np.zeros(5 * N_total, dtype='int32')
col_vec = np.zeros(5 * N_total, dtype='int32')
data_vec = np.zeros(5 * N_total)
b = np.zeros((N_total, 1))

def point2ind(m, n):
    return (n - 1) * (N - 2) + m - 1

ind = 0
for n in range(1, N-1):
    for m in range(1, N-1):
        k = point2ind(m, n)
        # A[k, k] = -4 / dx ** 2
        row_vec[ind] = k
        col_vec[ind] = k
        data_vec[ind] = -4 / dx ** 2
        ind = ind + 1
        if m > 1:
            # A[k, k - 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k - 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        if n < N - 2:
            # A[k, k + N - 2] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + N - 2
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        if m < N - 2:
            # A[k, k + 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        if n > 1:
            # A[k, k - (N - 2)] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k - (N - 2)
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - a(x[m]) / dx ** 2

A = scipy.sparse.csc_array((data_vec[0:ind], (row_vec[0:ind], col_vec[0:ind])), shape=(N_total, N_total))
t1 = time.time()
print("Time to set up matrix = {}s".format(t1 - t0))

# Solve system
t2 = time.time()
# u = np.linalg.solve(A, b)
u = scipy.sparse.linalg.spsolve(A, b)
t3 = time.time()
print("Time to solve = {}s".format(t3 - t2))
print("Total time = {}s".format(t3 - t0))

U_int = u.reshape((N-2, N-2))
U = np.zeros((N, N))
U[1:(N-1), 1:(N-1)] = U_int
U[0, :] = a(x)

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