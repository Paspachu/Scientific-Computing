#%%
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#%%
"""Problem 1"""
x0 = -1
xf = 2
t0 = 0
tf = 0.25

def true_solution(t, x):
    return np.exp(-5 * 16 * np.pi **2 * t / 9) * np.sin(4 * np.pi * (x + 1) / 3)

def forward_euler(Nx, Nt, plot):
    x = np.linspace(x0, xf, Nx)
    dx = x[1] - x[0]
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]

    if plot:
        print("Nx = {}".format(Nx))
        print("dx = {}".format(dx))
        print("Nt = {}".format(Nt))
        print("dt = {}".format(dt))

    U = np.zeros((Nx, Nt))
    U[:, 0] = np.sin(4 * np.pi * (x + 1) / 3)
    U[0, :] = 0
    U[-1, :] = 0

    A = 5 * (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2
    # Solve with Forward Euler
    for k in range(Nt - 1):
        U[1:-1, (k + 1):(k + 2)] = U[1:-1, k:(k + 1)] + dt * (A @ U[1:-1, k:(k + 1)])

    T, X = np.meshgrid(t, x)
    err = U - true_solution(T, X)

    approx = U[8, -1]
    global_err = np.max(np.abs(err))

    if plot:
        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, U)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, true_solution(T, X))
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, err)
        plt.show()

        print(approx, global_err)

    return approx, global_err

def backward_euler(Nx, Nt, plot):
    x = np.linspace(x0, xf, Nx)
    dx = x[1] - x[0]
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]

    if plot:
        print("Nx = {}".format(Nx))
        print("dx = {}".format(dx))
        print("Nt = {}".format(Nt))
        print("dt = {}".format(dt))

    U = np.zeros((Nx, Nt))
    U[:, 0] = np.sin(4 * np.pi * (x + 1) / 3)
    U[0, :] = 0
    U[-1, :] = 0

    A = 5 * (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2
    # Solve with Backward Euler
    for k in range(Nt - 1):
        U[1:-1, (k + 1):(k + 2)] = np.linalg.solve(np.eye(Nx - 2) - dt * A, U[1:-1, k:(k + 1)])

    T, X = np.meshgrid(t, x)
    err = U - true_solution(T, X)

    approx = U[8, -1]
    global_err = np.max(np.abs(err))

    if plot:
        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, U)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, true_solution(T, X))
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, err)
        plt.show()

        print(approx, global_err)

    return approx, global_err

#%%
"""Problem 1(a)"""
Nx = 25
Nt = 257

A1, A2 = forward_euler(Nx, Nt, False)

#%%
"""Problem 1(b)"""
Nx = 25
Nt = 139

A3, A4 = forward_euler(Nx, Nt, False)

#%%
"""Problem 1(c)"""
Nx = 25
Nt = 6

A5, A6 = backward_euler(Nx, Nt, False)

# %%
"""Problem 1(d)"""
Nx = 25
Nt = 51

A7, A8 = backward_euler(Nx, Nt, False)

# %%
"""Problem 2"""
x0 = 0
xf = 1
t0 = 0
tf = 0.1

def true_solution2(t, x):
    return 10 * t * x + np.exp(-(np.pi ** 2) * t) * np.sin(np.pi * x) - 0.8 * np.exp(-9 * (np.pi ** 2) * t) * np.sin(3 * np.pi * x)

def forward_euler2(Nx, Nt, plot):
    x = np.linspace(x0, xf, Nx)
    dx = x[1] - x[0]
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]

    if plot:
        print("Nx = {}".format(Nx))
        print("dx = {}".format(dx))
        print("Nt = {}".format(Nt))
        print("dt = {}".format(dt))

    U = np.zeros((Nx, Nt))
    U[:, 0] = np.sin(np.pi * x) - 0.8 * np.sin(3 * np.pi * x)
    U[0, :] = 0
    U[-1, :] = 10 * t

    A = (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2
    # Solve with Forward Euler
    bc = np.zeros((19, 1))
    for k in range(Nt - 1):
        bc[-1] = U[-1, k] / dx ** 2
        U[1:-1, (k + 1):(k + 2)] = U[1:-1, k:(k + 1)] + dt * (A @ U[1:-1, k:(k + 1)] + bc + 10 * x[1:-1].reshape(19, 1))

    T, X = np.meshgrid(t, x)
    S = true_solution2(T, X)
    err = U - S

    approx = U[10, -1]
    global_err = np.max(np.abs(err))

    if plot:
        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, U)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, S)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, err)
        plt.show()

        print(S[10, -1], approx, global_err)

    return approx, global_err

def trapezoidal(Nx, Nt, plot):
    x = np.linspace(x0, xf, Nx)
    dx = x[1] - x[0]
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]

    if plot:
        print("Nx = {}".format(Nx))
        print("dx = {}".format(dx))
        print("Nt = {}".format(Nt))
        print("dt = {}".format(dt))

    U = np.zeros((Nx, Nt))
    U[:, 0] = np.sin(np.pi * x) - 0.8 * np.sin(3 * np.pi * x)
    U[0, :] = 0
    U[-1, :] = 10 * t

    A = (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2
    I = np.eye(Nx - 2)
    A_lhs = (I - (dt / 2) * A)
    A_rhs = (I + (dt / 2) * A)
    # Solve with Trapezoidal Method
    bc = np.zeros((19, 1))
    for k in range(Nt - 1):
        bc[-1] = (U[-1, k] + U[-1, k + 1]) / (2 * dx ** 2)
        U[1:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U[1:-1, k:(k + 1)] + dt * (bc + 10 * x[1:-1].reshape(19, 1)))

    T, X = np.meshgrid(t, x)
    S = true_solution2(T, X)
    err = U - S

    approx = U[10, -1]
    global_err = np.max(np.abs(err))

    if plot:
        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, U)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, S)
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, err)
        plt.show()

        print(S[10, -1], approx, global_err)

    return approx, global_err

#%%
"""Problem 2(a)"""
Nx = 21
Nt = 56

A9, A10 = forward_euler2(Nx, Nt, False)

#%%
"""Problem 2(b)"""
Nx = 21
Nt = 201

A11, A12 = forward_euler2(Nx, Nt, False)

# %%
"""Problem 2(c)"""
Nx = 21
Nt = 11

A13, A14 = trapezoidal(Nx, Nt, False)

#%%
"""Problem 2(d)"""
Nx = 21
Nt = 101

A15, A16 = trapezoidal(Nx, Nt, False)

# %%
"""Problem 3"""
kappa = 0.1
x0 = 0
xf = 1
y0 = 0
yf = 1
t0 = 0
tf = 1
N = 11

def bc_t0(x, y):
    if x == 0 or y == 0 or x == 1 or y == 1:
        return 0
    return np.exp(-10 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

def point2ind(m, n):
    return (n - 1) * (N - 2) + m - 1

def forward_euler3(Nt):
    x = np.linspace(x0, xf, N)
    y = np.linspace(y0, yf, N)
    dx = x[1] - x[0]
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]

    N_total = (N - 2) * (N - 2)
    A = np.zeros((N_total, N_total))
    U_int = np.zeros((N_total, Nt))

    for n in range(1, N-1):
        for m in range(1, N-1):
            k = point2ind(m, n)
            A[k, k] = -4 * kappa / dx ** 2
            U_int[k, 0] = bc_t0(x[m], y[n])
            if m > 1:
                A[k, k - 1] = 1 * kappa / dx ** 2
            if n < N - 2:
                A[k, k + N - 2] = 1 * kappa / dx ** 2
            if m < N - 2:
                A[k, k + 1] = 1 * kappa / dx ** 2
            if n > 1:
                A[k, k - (N - 2)] = 1 * kappa / dx ** 2

    # Solve with Forward Euler
    for k in range(Nt - 1):
        U_int[:, (k + 1):(k + 2)] = U_int[:, k:(k + 1)] + dt * (A @ U_int[:, k:(k + 1)])
    
    U = np.zeros((N, N, Nt))
    U[1:(N-1), 1:(N-1), :] = U_int.reshape((N-2, N-2, Nt))

    return U[5, 5, -1], U

def trapezoidal3(Nt):
    x = np.linspace(x0, xf, N)
    y = np.linspace(y0, yf, N)
    dx = x[1] - x[0]
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]

    N_total = (N - 2) * (N - 2)
    A = np.zeros((N_total, N_total))
    U_int = np.zeros((N_total, Nt))

    for n in range(1, N-1):
        for m in range(1, N-1):
            k = point2ind(m, n)
            A[k, k] = -4 * kappa / dx ** 2
            U_int[k, 0] = bc_t0(x[m], y[n])
            if m > 1:
                A[k, k - 1] = 1 * kappa / dx ** 2
            if n < N - 2:
                A[k, k + N - 2] = 1 * kappa / dx ** 2
            if m < N - 2:
                A[k, k + 1] = 1 * kappa / dx ** 2
            if n > 1:
                A[k, k - (N - 2)] = 1 * kappa / dx ** 2

    # Solve with Trapezoidal Method
    I = np.eye(N_total)
    A_lhs = (I - (dt / 2) * A)
    A_rhs = (I + (dt / 2) * A)
    for k in range(Nt - 1):
        U_int[:, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U_int[:, k:(k + 1)])
    
    U = np.zeros((N, N, Nt))
    U[1:(N-1), 1:(N-1), :] = U_int.reshape((N-2, N-2, Nt))

    return U[5, 5, -1], U

def plot_at_ts(U, indices, four, suptitle, png_title):
    x = np.linspace(x0, xf, N)
    y = np.linspace(y0, yf, N)
    X, Y = np.meshgrid(x, y)

    fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='3d'))
    fig.suptitle(suptitle, fontsize=36, fontweight='bold')

    ax1.plot_surface(X, Y, U[:, :, indices[0]])
    ax1.set_title('t = 0', fontsize=24, fontweight='bold')
    ax1.set_xlabel('x', fontsize=20)
    ax1.set_ylabel('y', fontsize=20)
    ax1.set_zlabel('U', fontsize=20)
 
    ax2.plot_surface(X, Y, U[:, :, indices[1]])
    if four:
        ax2.set_title('t = 1/3', fontsize=24, fontweight='bold')
    else:
        ax2.set_title('t = 0.33', fontsize=24, fontweight='bold')
    ax2.set_xlabel('x', fontsize=20)
    ax2.set_ylabel('y', fontsize=20)
    ax2.set_zlabel('U', fontsize=20)

    ax3.plot_surface(X, Y, U[:, :, indices[2]])
    if four:
        ax3.set_title('t = 2/3', fontsize=24, fontweight='bold')
    else:
        ax3.set_title('t = 0.67', fontsize=24, fontweight='bold')
    ax3.set_xlabel('x', fontsize=20)
    ax3.set_ylabel('y', fontsize=20)
    ax3.set_zlabel('U', fontsize=20)

    ax4.plot_surface(X, Y, U[:, :, indices[3]])
    ax4.set_title('t = 1', fontsize=24, fontweight='bold')
    ax4.set_xlabel('x', fontsize=20)
    ax4.set_ylabel('y', fontsize=20)
    ax4.set_zlabel('U', fontsize=20)


    fig.savefig(png_title, pad_inches=0.5, bbox_inches='tight')

#%%
"""Problem 3(a)"""
Nt = 4
A17, U = forward_euler3(Nt)
plot_at_ts(U, np.array([0, 1, 2, 3]), True, 'Forward Euler with $\Delta t = 1/3$', 'Prob3a.png')

# %%
"""Problem 3(b)"""
Nt = 101
A18, U = forward_euler3(Nt)
plot_at_ts(U, np.array([0, 33, 67, -1]), False, 'Forward Euler with $\Delta t = 0.01$', 'Prob3b.png')

# %%
"""Problem 3(c)"""
Nt = 4
A19, U = trapezoidal3(Nt)
plot_at_ts(U, np.array([0, 1, 2, 3]), True, 'Trapezoidal with $\Delta t = 1/3$', 'Prob3c.png')

# %%
"""Problem 3(d)"""
Nt = 101
A20, U = trapezoidal3(Nt)
plot_at_ts(U, np.array([0, 33, 67, -1]), False, 'Trapezoidal with $\Delta t = 0.01$', 'Prob3d.png')

# %%
