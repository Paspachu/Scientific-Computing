#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from matplotlib import cm

#%%
"""Problem 1"""
t0 = 0
theta0 = 0.5
tN = 6
thetaN = 0.5

N = 101
t = np.linspace(t0, tN, N)
dt = t[1] - t[0]
# print("dt = {}".format(dt))

#%%
def F(theta):
    z = np.zeros_like(theta)
    z[0] = theta[0] - theta0
    z[-1] = theta[-1] - thetaN
    for k in range(1, N - 1):
        z[k] = theta[k - 1] - 2 * theta[k] + theta[k + 1] + dt * 0.05 * (theta[k + 1] - theta[k - 1]) + dt ** 2 * np.sin(theta[k])
    return z

def jacobian(theta):
    J = np.zeros((N, N))
    J[0, 0] = 1
    J[-1, -1] = 1
    for k in range(1, N-1):
        J[k, k - 1] = 1 - dt * 0.05
        J[k, k] = -2 + dt ** 2 * np.cos(theta[k])
        J[k, k + 1] = 1 + dt * 0.05
    return J

#%%
"""Problem 1(a)"""
theta = 0.5 * np.ones(N)
k = 0
max_steps = 1000

while np.max(np.abs(F(theta))) >= 1e-8 and k < max_steps:
    change_in_theta = np.linalg.solve(jacobian(theta), F(theta))
    theta = theta - change_in_theta
    k = k + 1

theta = theta.reshape(N)
# print("k = {}".format(k))
# plt.plot(t, theta, 'k', t0, theta0, 'ro', tN, thetaN, 'ro')
# plt.show()

# %%
A1 = theta[50]
A2 = np.max(theta)
A3 = np.min(theta)

# print(A1, A2, A3)

# %%
"""Problem 1(b)"""
theta = 0.005 * t ** 4 - 0.07 * t ** 3 + 0.66 * t ** 2 - 2.56 * t + 0.55
k = 0
max_steps = 1000

while np.max(np.abs(F(theta))) >= 1e-8 and k < max_steps:
    change_in_theta = np.linalg.solve(jacobian(theta), F(theta))
    theta = theta - change_in_theta
    k = k + 1

theta = theta.reshape(N)
# print("k = {}".format(k))
# plt.plot(t, theta, 'k', t0, theta0, 'ro', tN, thetaN, 'ro')
# plt.show()

# %%
A4 = theta[50]
A5 = np.max(theta)
A6 = np.min(theta)

# print(A4, A5, A6)

# %%
"""Problelm 1 Note"""
theta = -3 * t ** 2 + 18 * t + 0.5
k = 0
max_steps = 1000

while np.max(np.abs(F(theta))) >= 1e-8 and k < max_steps:
    change_in_theta = np.linalg.solve(jacobian(theta), F(theta))
    theta = theta - change_in_theta
    k = k + 1

theta = theta.reshape(N)
# print("k = {}".format(k))
# plt.plot(t, theta, 'k', t0, theta0, 'ro', tN, thetaN, 'ro')
# plt.show()
# print(theta[50], np.max(theta), np.min(theta))

# %%
"""Problem 2"""
x0 = 0
xN = 3
y0 = 0
yN = 3

def bottom(x):
    return x ** 2 - 3 * x

def top(x):
    return np.sin(2 * np.pi * x / 3)

def left(y):
    return np.sin(np.pi * y / 3)

def right(y):
    return 3 * y - y ** 2

#%%
"""Problem 2(a)"""
N = 61
x = np.linspace(x0, xN, N)
y = np.linspace(y0, yN, N)
dx = x[1] - x[0]
# print("N = {}".format(N))
# print("dx = {}".format(dx))

def point2ind(m, n):
    return (n - 1) * (N - 2) + m - 1

#%%
# Set up system Au = b
N_total = (N - 2) * (N - 2)
# print("N_total = {}".format(N_total))
# print("Entries in matrix = {}".format(N_total ** 2))

#%%
row_vec = np.zeros(5 * N_total, dtype='int32')
col_vec = np.zeros(5 * N_total, dtype='int32')
data_vec = np.zeros(5 * N_total)
b = np.zeros((N_total, 1))

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
        else:
            b[k] = b[k] - left(y[n]) / dx ** 2
        
        if n < N - 2:
            # A[k, k + N - 2] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + N - 2
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - top(x[m]) / dx ** 2
        
        if m < N - 2:
            # A[k, k + 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - right(y[n]) / dx ** 2

        if n > 1:
            # A[k, k - (N - 2)] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k - (N - 2)
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - bottom(x[m]) / dx ** 2

A = scipy.sparse.csc_array((data_vec[0:ind], (row_vec[0:ind], col_vec[0:ind])),
shape=(N_total, N_total))

# Solve system
u = scipy.sparse.linalg.spsolve(A, b)

#%%
U_int = u.reshape((N-2, N-2))
U = np.zeros((N, N))
U[1:(N-1), 1:(N-1)] = U_int
U[0, :] = bottom(x)
U[-1, :] = top(x)
U[:, 0] = left(y)
U[:, -1] = right(y)

# # These vectors are just to plot the boundary conditions
# zero_vector = np.zeros_like(x)
# three_vector = 3 * np.ones_like(x)
# X, Y = np.meshgrid(x, y)
# ax = plt.axes(projection='3d')
# # Plot the solution
# ax.plot_surface(X, Y, U)
# # u(0, y) = 0
# ax.plot3D(zero_vector, y, left(y), 'r')
# # u(x, 3) = 0
# ax.plot3D(x, three_vector, top(x), 'r')
# # u(3, y) = 0
# ax.plot3D(three_vector, y, right(y), 'r')
# # u(x, 0) = a(x)
# ax.plot3D(x, zero_vector, bottom(x), 'r')
# plt.show()

#%%
A7 = U[20, 20]
A8 = U[40, 40]

# print(A7, A8)

# %%
"""Problem 2(b)"""
N = 201
x = np.linspace(x0, xN, N)
y = np.linspace(y0, yN, N)
dx = x[1] - x[0]
# print("N = {}".format(N))
# print("dx = {}".format(dx))

def point2ind(m, n):
    return (n - 1) * (N - 2) + m - 1

#%%
# Set up system Au = b
N_total = (N - 2) * (N - 2)
# print("N_total = {}".format(N_total))
# print("Entries in matrix = {}".format(N_total ** 2))

#%%
row_vec = np.zeros(5 * N_total, dtype='int32')
col_vec = np.zeros(5 * N_total, dtype='int32')
data_vec = np.zeros(5 * N_total)
b = np.zeros((N_total, 1))

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
        else:
            b[k] = b[k] - left(y[n]) / dx ** 2
        
        if n < N - 2:
            # A[k, k + N - 2] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + N - 2
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - top(x[m]) / dx ** 2
        
        if m < N - 2:
            # A[k, k + 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - right(y[n]) / dx ** 2

        if n > 1:
            # A[k, k - (N - 2)] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k - (N - 2)
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - bottom(x[m]) / dx ** 2

A = scipy.sparse.csc_array((data_vec[0:ind], (row_vec[0:ind], col_vec[0:ind])),
shape=(N_total, N_total))

# Solve system
u = scipy.sparse.linalg.spsolve(A, b)

#%%
U_int = u.reshape((N-2, N-2))
U = np.zeros((N, N))
U[1:(N-1), 1:(N-1)] = U_int
U[0, :] = bottom(x)
U[-1, :] = top(x)
U[:, 0] = left(y)
U[:, -1] = right(y)

# # These vectors are just to plot the boundary conditions
# zero_vector = np.zeros_like(x)
# three_vector = 3 * np.ones_like(x)
# X, Y = np.meshgrid(x, y)
# ax = plt.axes(projection='3d')
# # Plot the solution
# ax.plot_surface(X, Y, U)
# # u(0, y) = 0
# ax.plot3D(zero_vector, y, left(y), 'r')
# # u(x, 3) = 0
# ax.plot3D(x, three_vector, top(x), 'r')
# # u(3, y) = 0
# ax.plot3D(three_vector, y, right(y), 'r')
# # u(x, 0) = a(x)
# ax.plot3D(x, zero_vector, bottom(x), 'r')
# plt.show()

#%%
A9 = U[67, 67]
A10 = U[133, 133]

# print(A9, A10)

# %%
"""Problem 3"""
x0 = -1
xN = 1
y0 = -1
yN = 1

def top(x):
    return (x ** 3 - x) / 3

def f(x, y):
    return -np.exp(-2 * (x ** 2 + y ** 2))

#%%
"""Problem 3(a)"""
Nx = 21
Ny = 41
x = np.linspace(x0, xN, Nx)
y = np.linspace(y0, yN, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
# print("Nx = {}".format(Nx))
# print("dx = {}".format(dx))
# print("Ny = {}".format(Ny))
# print("dy = {}".format(dy))

def point2ind(m, n):
    return (n - 1) * (Nx - 2) + m - 1

#%%
# Set up system Au = b
N_total = (Nx - 2) * (Ny - 2)
# print("N_total = {}".format(N_total))
# print("Entries in matrix = {}".format(N_total ** 2))

#%%
row_vec = np.zeros(5 * N_total, dtype='int32')
col_vec = np.zeros(5 * N_total, dtype='int32')
data_vec = np.zeros(5 * N_total)
b = np.zeros((N_total, 1))

ind = 0
for n in range(1, Ny-1):
    for m in range(1, Nx-1):
        k = point2ind(m, n)
        b[k] = f(x[m], y[n])
        # A[k, k] = -4 / dx ** 2
        row_vec[ind] = k
        col_vec[ind] = k
        data_vec[ind] = -2 / dx ** 2 - 2/ dy ** 2
        ind = ind + 1
        if m > 1:
            # A[k, k - 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k - 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        
        if n < Ny - 2:
            # A[k, k + N - 2] = 1 / dy ** 2
            row_vec[ind] = k
            col_vec[ind] = k + (Nx - 2)
            data_vec[ind] = 1 / dy ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - top(x[m]) / dy ** 2
        
        if m < Nx - 2:
            # A[k, k + 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1

        if n > 1:
            # A[k, k - (N - 2)] = 1 / dy ** 2
            row_vec[ind] = k
            col_vec[ind] = k - (Nx - 2)
            data_vec[ind] = 1 / dy ** 2
            ind = ind + 1

A = scipy.sparse.csc_array((data_vec[0:ind], (row_vec[0:ind], col_vec[0:ind])),
shape=(N_total, N_total))

# Solve system
u = scipy.sparse.linalg.spsolve(A, b)

#%%
U_int = u.reshape((Ny-2, Nx-2))
U = np.zeros((Ny, Nx))
U[1:(Ny-1), 1:(Nx-1)] = U_int
U[-1, :] = top(x)

# # These vectors are just to plot the boundary conditions
# X, Y = np.meshgrid(x, y)
# ax = plt.axes(projection='3d')
# # Plot the solution
# ax.plot_surface(X, Y, U)
# plt.show()

#%%
A11 = U[20, 10]
A12 = U[30, 5]

# print(A11, A12)

# %%
"""Problem 3(b)"""
Nx = 201
Ny = 81
x = np.linspace(x0, xN, Nx)
y = np.linspace(y0, yN, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
# print("Nx = {}".format(Nx))
# print("dx = {}".format(dx))
# print("Ny = {}".format(Ny))
# print("dy = {}".format(dy))

def point2ind(m, n):
    return (n - 1) * (Nx - 2) + m - 1

#%%
# Set up system Au = b
N_total = (Nx - 2) * (Ny - 2)
# print("N_total = {}".format(N_total))
# print("Entries in matrix = {}".format(N_total ** 2))

#%%
row_vec = np.zeros(5 * N_total, dtype='int32')
col_vec = np.zeros(5 * N_total, dtype='int32')
data_vec = np.zeros(5 * N_total)
b = np.zeros((N_total, 1))

ind = 0
for n in range(1, Ny-1):
    for m in range(1, Nx-1):
        k = point2ind(m, n)
        b[k] = f(x[m], y[n])
        # A[k, k] = -4 / dx ** 2
        row_vec[ind] = k
        col_vec[ind] = k
        data_vec[ind] = -2 / dx ** 2 - 2/ dy ** 2
        ind = ind + 1
        if m > 1:
            # A[k, k - 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k - 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1
        
        if n < Ny - 2:
            # A[k, k + N - 2] = 1 / dy ** 2
            row_vec[ind] = k
            col_vec[ind] = k + (Nx - 2)
            data_vec[ind] = 1 / dy ** 2
            ind = ind + 1
        else:
            b[k] = b[k] - top(x[m]) / dy ** 2
        
        if m < Nx - 2:
            # A[k, k + 1] = 1 / dx ** 2
            row_vec[ind] = k
            col_vec[ind] = k + 1
            data_vec[ind] = 1 / dx ** 2
            ind = ind + 1

        if n > 1:
            # A[k, k - (N - 2)] = 1 / dy ** 2
            row_vec[ind] = k
            col_vec[ind] = k - (Nx - 2)
            data_vec[ind] = 1 / dy ** 2
            ind = ind + 1

A = scipy.sparse.csc_array((data_vec[0:ind], (row_vec[0:ind], col_vec[0:ind])),
shape=(N_total, N_total))

# Solve system
u = scipy.sparse.linalg.spsolve(A, b)

#%%
U_int = u.reshape((Ny-2, Nx-2))
U = np.zeros((Ny, Nx))
U[1:(Ny-1), 1:(Nx-1)] = U_int
U[-1, :] = top(x)

# # These vectors are just to plot the boundary conditions
# X, Y = np.meshgrid(x, y)
# ax = plt.axes(projection='3d')
# # Plot the solution
# ax.plot_surface(X, Y, U)
# plt.show()

#%%
A13 = U[40, 100]
A14 = U[60, 50]

# print(A13, A14)

# %%
"""Problem 4 (For Report)"""
x0 = 0
xN = 2
y0 = 0
yN = 2

def f(x, y):
    return -np.sin(np.pi * x) * np.sin(np.pi * y)

def u_func(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi ** 2)

def point2ind(m, n, N):
    return (n - 1) * (N - 2) + m - 1

#%%
"""Problem 4(a)"""
def get_errors_five_points(N, plot):
    x = np.linspace(x0, xN, N)
    y = np.linspace(y0, yN, N)
    dx = x[1] - x[0]
    print("N = {}".format(N))
    print("dx = {}".format(dx))

    N_total = (N - 2) * (N - 2)

    row_vec = np.zeros(5 * N_total, dtype='int32')
    col_vec = np.zeros(5 * N_total, dtype='int32')
    data_vec = np.zeros(5 * N_total)
    b = np.zeros((N_total, 1))

    ind = 0
    for n in range(1, N-1):
        for m in range(1, N-1):
            k = point2ind(m, n, N)
            b[k] = f(x[m], y[n])
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

    A = scipy.sparse.csc_array((data_vec[0:ind], (row_vec[0:ind], col_vec[0:ind])),
    shape=(N_total, N_total))
    # Solve system
    u = scipy.sparse.linalg.spsolve(A, b)   

    U_int = u.reshape((N-2, N-2))
    U = np.zeros((N, N))
    U[1:(N-1), 1:(N-1)] = U_int

    S = np.zeros((N, N))
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            S[i, j] = u_func(x_val, y_val)

    if plot:
        X, Y = np.meshgrid(x, y)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, U, cmap=cm.coolwarm)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, np.transpose(S), cmap=cm.coolwarm)
        plt.show()
    
    return dx, np.max(np.abs(S - U))

#%%
five_point = np.zeros(8)
dxs = np.zeros(8)
Ns = np.array([2 ** n + 1 for n in range(3, 11)])
for i, N in enumerate(Ns):
    dxs[i], five_point[i] = get_errors_five_points(N, False)

# %%
log_dxs = np.log(dxs)
log_five_point = np.log(five_point)
a, b = np.polyfit(log_dxs, log_five_point, 1)

plt.scatter(log_dxs, log_five_point, c = 'r', label='Exact error points')
plt.plot(log_dxs, a * log_dxs + b, label='Best fit line (linear)')
plt.title('5-points Laplacian log-log plot')
plt.xlabel('ln($\Delta x$)')
plt.ylabel('ln(E)')
plt.legend()
plt.show()
print(a, b)

# %%
"""Problem 4(b)"""
def get_errors_nine_points(N, plot):
    x = np.linspace(x0, xN, N)
    y = np.linspace(y0, yN, N)
    dx = x[1] - x[0]
    print("N = {}".format(N))
    print("dx = {}".format(dx))

    N_total = (N - 2) * (N - 2)

    row_vec = np.zeros(9 * N_total, dtype='int32')
    col_vec = np.zeros(9 * N_total, dtype='int32')
    data_vec = np.zeros(9 * N_total)
    b = np.zeros((N_total, 1))

    ind = 0
    for n in range(1, N-1):
        for m in range(1, N-1):
            k = point2ind(m, n, N)
            b[k] = f(x[m], y[n])
            # A[k, k] = -20 / (6 * dx ** 2)
            row_vec[ind] = k
            col_vec[ind] = k
            data_vec[ind] = -20 / (6 * dx ** 2)
            ind = ind + 1
            if m > 1:
                # A[k, k - 1] = 4 / (6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k - 1
                data_vec[ind] = 4 / (6 * dx ** 2)
                ind = ind + 1
            if n < N - 2:
                # A[k, k + N - 2] = 4 /(6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k + N - 2
                data_vec[ind] = 4 / (6 * dx ** 2)
                ind = ind + 1
            if m < N - 2:
                # A[k, k + 1] = 4 / (6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k + 1
                data_vec[ind] = 4 / (6 * dx ** 2)
                ind = ind + 1
            if n > 1:
                # A[k, k - (N - 2)] = 4 / (6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k - (N - 2)
                data_vec[ind] = 4 / (6 * dx ** 2)
                ind = ind + 1
            if m > 1 and n > 1:
                # A[k, k - (N - 2) - 1] = 1 / (6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k - (N - 2) - 1
                data_vec[ind] = 1 / (6 * dx ** 2)
                ind = ind + 1
            if m > 1 and n < N - 2:
                # A[k, k + N - 3] = 1 / (6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k + N - 3
                data_vec[ind] = 1 / (6 * dx ** 2)
                ind = ind + 1
            if m < N - 2 and n > 1:
                # A[k, k - (N - 2) + 1] = 1 / (6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k - (N - 2) + 1
                data_vec[ind] = 1 / (6 * dx ** 2)
                ind = ind + 1
            if m < N - 2 and n < N - 2:
                # A[k, k + N - 1] = 1 / (6 * dx ** 2)
                row_vec[ind] = k
                col_vec[ind] = k + N - 1
                data_vec[ind] = 1 / (6 * dx ** 2)
                ind = ind + 1

    A = scipy.sparse.csc_array((data_vec[0:ind], (row_vec[0:ind], col_vec[0:ind])),
    shape=(N_total, N_total))
    # Solve system
    u = scipy.sparse.linalg.spsolve(A, b)   

    U_int = u.reshape((N-2, N-2))
    U = np.zeros((N, N))
    U[1:(N-1), 1:(N-1)] = U_int

    S = np.zeros((N, N))
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            S[i, j] = u_func(x_val, y_val)
    
    if plot:
        X, Y = np.meshgrid(x, y)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, U, cmap=cm.coolwarm)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, np.transpose(S), cmap=cm.coolwarm)
        plt.show()
    
    return dx, np.max(np.abs(S - U))

#%%
nine_point = np.zeros(8)
dxs = np.zeros(8)
Ns = np.array([2 ** n + 1 for n in range(3, 11)])
for i, N in enumerate(Ns):
    dxs[i], nine_point[i] = get_errors_nine_points(N, False)

# %%
log_dxs = np.log(dxs)
log_nine_point = np.log(nine_point)
a, b = np.polyfit(log_dxs, log_nine_point, 1)

plt.scatter(log_dxs, log_nine_point, c = 'r', label='Exact error points')
plt.plot(log_dxs, a * log_dxs + b, label='Best fit line (linear)')
plt.title('9-points Laplacian log-log plot')
plt.xlabel('ln($\Delta x$)')
plt.ylabel('ln(E)')
plt.legend()
plt.show()
print(a, b)

#%%
"""Problem 4(c)"""
N = 101
x = np.linspace(x0, xN, N)
y = np.linspace(y0, yN, N)
dx = x[1] - x[0]
print("dx = {}".format(dx))

#%%
dis_error = np.zeros((N, N))
f_xx_plus_f_yy = np.zeros((N, N))
for i, x_val in enumerate(x):
    for j, y_val in enumerate(y):
        dis_error[i, j] = (4 * (u_func(x_val - dx, y_val) + u_func(x_val + dx, y_val) + u_func(x_val, y_val - dx) + u_func(x_val, y_val + dx)) + 
                        (u_func(x_val - dx, y_val - dx) + u_func(x_val - dx, y_val + dx) + u_func(x_val + dx, y_val - dx) + u_func(x_val + dx, y_val + dx)) - 
                        20 * u_func(x_val, y_val)) / (6 * dx ** 2) - f(x_val, y_val)
        f_xx_plus_f_yy[i, j] = 2 * (np.pi ** 2) * np.sin(np.pi * x_val) * np.sin(np.pi * y_val)

#%%
X, Y = np.meshgrid(x, y)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, dis_error, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Surface Plot of $\Delta_9^2 u(x, y) - f(x, y)$')
plt.show()

# %%
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f_xx_plus_f_yy, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z', verticalalignment='top')
ax.zaxis.labelpad = -2
ax.set_title('Surface Plot of $f_{xx} + f_{yy}$')
plt.show()

# %%
