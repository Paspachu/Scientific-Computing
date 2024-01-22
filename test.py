#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%%
# u''(x) = f(x)
# u(x0) = 0 = u(xf)

def f(x):
    return 2 ** x

def true_sol(x):
    return (-3 * x + 2 ** (x + 1) - 2) / (-2 * np.log10(2) ** 2)

#%%
N = 3
x0 = 0
xf = 2
x = np.linspace(x0, xf, N + 1)

A = np.zeros((N + 1, N + 1))
b = np.zeros((N + 1, 1))
b[0] = 0
b[N] = 0
for i in range(1, N):
    b[i] = f(x[i])

# Fill in first row: u(x0) = 0
for j in range(N + 1):
    A[0, j] = x[0] ** (N - j)

# Fill in last row: u(xf) = 0
for j in range(N + 1):
    A[N, j] = x[N] ** (N - j)

for i in range(1, N):
    for j in range(N - 1):
        A[i, j] = (N - j) * (N - j - 1) * x[i] ** (N - j - 2)

c = np.linalg.solve(A, b)

def u(x):
    # y = 0
    # for j in range(N + 1):
    #     y = y + c[j] * x ** (N - j)
    # return y
    return np.polyval(c, x)

xplot = np.linspace(x0, xf, 1000)
plt.plot(xplot, true_sol(xplot), 'k')
plt.plot(xplot, u(xplot), 'b')
plt.show()

err = np.max(np.abs(true_sol(x) - u(x)))
print("Error = {}".format(err))


#%%
A = np.diag(2 * np.ones(5), 1) + np.diag(-2 * np.ones(5), -1)
A[0, -1] = -2
A[-1, 0] = 2
A

#%%
vals, _ = np.linalg.eig(A)
sorted(0.029*vals)

#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

#%%
# u_xx + u_yy = 0
# u(x, 0) = x^2 - x
# All other boundaries are zero
# u(x, 1) = u(0, y) = u(1, y) = 0
x0 = 0
xN = 3
y0 = 0
yN = 3
N = 4
x = np.linspace(x0, xN, N)
y = np.linspace(y0, yN, N)
dx = x[1] - x[0]
print("N = {}".format(N))
print("dx = {}".format(dx))
# Boundary condition u(x, 0)
def a(y):
  return 3*y - y ** 2

# Set up system Au = b
N_total = (N - 2) * (N - 2)
print("N_total = {}".format(N_total))
print("Entries in matrix = {}".format(N_total ** 2))

# A = np.zeros((N_total, N_total))
A = scipy.sparse.dok_array((N_total, N_total))
b = np.ones((N_total, 1))
def point2ind(m, n):
  return (n - 1) * (N - 2) + m - 1
for n in range(1, N-1):
  for m in range(1, N-1):
    k = point2ind(m, n)
    A[k, k] = -4 / dx ** 2
    if m > 1:
      A[k, k - 1] = 1 / dx ** 2
    if n < N - 2:
      A[k, k + N - 2] = 1 / dx ** 2
    if m < N - 2:
      A[k, k + 1] = 1 / dx ** 2
    else:
      b[k] = b[k] - a(y[n]) / dx ** 2
    if n > 1:
      A[k, k - (N - 2)] = 1 / dx ** 2
    

A = A.tocsc()
print(A)
print(b)

#%%


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
Annotations



# %%
def f(t, x):
    return x ** 2 - t

dt = 0.1
t = np.arange(0, 5.1, dt)
x = np.zeros_like(t)

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k], t[k])

plt.plot(t, x)
plt.show()

# %%
tspan = np.array([0, 5])
sol = solve_ivp(lambda x, y: y ** 2 - x, tspan, np.array([0]))

plt.plot(sol.t, sol.y[0])
plt.show()

# %%
def f(t, x):
    return x^2 - t

tspan = np.array([0, 5])
sol = solve_ivp(f, tspan, np.array([0]))

plt.plot(sol.t, sol.y[0])
plt.show()

# %%
tspan = np.array([0, 5])
sol = solve_ivp(lambda t, x: x ** 2 - t, np.array([0]), tspan)

plt.plot(sol.t, sol.y[0])
plt.show()

# %%
def f(t, x):
  return 2 * x

solve_ivp(f, np.array([0, 2]), np.array([1]), method="BackwardEuler")
# %%
def f(x):
  return 2 * x

solve_ivp(f, np.array([0, 2]), np.array([1]), method="Radau")
# %%
import numpy as np
from scipy.integrate import solve_ivp



x0 = np.array([1, 0])
tspan = np.array([0, 10])
sol = solve_ivp(f, tspan, x0)


