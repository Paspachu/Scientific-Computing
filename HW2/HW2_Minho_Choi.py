#%%
# Import all the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# %%
"""Problem 1"""
# Initializations of boundary conditions
L = 1
x0 = -L
y0 = 0

xf = L
yf = 0

guess = 1

# Functions for Problem 1
def bisection(f, a, b, tol):
    x = (a + b) / 2
    while np.abs(b - a) >= tol:
        if np.sign(f(x)) == np.sign(f(a)):
            a = x
        else:
            b = x
        x = (a + b) / 2
    return x 

def V(x):
    return -100 * (np.sin(2 * x) + 1)

def f(x, y, lam):
    return np.array([y[1], -(V(x) + lam) * y[0]])

def shoot(lam):
    tspan = np.array([x0, xf])
    init_condition = np.array([y0, guess])
    sol = solve_ivp(lambda x, y: f(x, y, lam), tspan, init_condition)
    return sol.y[0, -1]

def print_eigenvals(eigenvals):
    for k in range(len(eigenvals)):
        print("eigenval = {}".format(eigenvals[k]))

# %%
"""Problem 1 (a)"""
lam = 23
sign = np.sign(shoot(lam))
dlam = 0.1

num_modes = 3
eigenvals = np.zeros(num_modes)
k = 0

while k < num_modes:
    lam_next = lam + dlam
    sign_next = np.sign(shoot(lam_next))
    if sign != sign_next:
        eigenvals[k] = bisection(shoot, lam, lam_next, 1e-8)
        k = k + 1
    lam = lam_next
    sign = sign_next

A1 = eigenvals[0]

# %%
"""Problem 1 (b)"""
def f_lam1(x, y):
    return np.array([y[1], -(V(x) + A1) * y[0]])

tspan = np.array([x0, xf])
init_condition = np.array([y0, guess])
sol = solve_ivp(f_lam1, tspan, init_condition,t_eval=np.array([-1, 0, 1]))
A2 = sol.y[0, 1]

# %%
"""Problem 1 (c)"""
A3 = eigenvals[1]

#%%
"""Problem 1 (d)"""
def f_lam2(x, y):
    return np.array([y[1], -(V(x) + A3) * y[0]])

tspan = np.array([x0, xf])
init_condition = np.array([y0, guess])
sol = solve_ivp(f_lam2, tspan, init_condition,t_eval=np.array([-1, 0, 1]))
A4 = sol.y[0, 1]

# %%
"""Problem 1 (e)"""
A5 = eigenvals[2]

#%%
"""Problem 1 (f)"""
def f_lam3(x, y):
    return np.array([y[1], -(V(x) + A5) * y[0]])

tspan = np.array([x0, xf])
init_condition = np.array([y0, guess])
sol = solve_ivp(f_lam3, tspan, init_condition,t_eval=np.array([-1, 0, 1]))
A6 = sol.y[0, 1]

# %%
"""Problem 2 (a)"""
def true_solution(t):
    return 0.5 * (np.exp(t) + np.exp(-t))

def f(x):
    return np.array([x[1], x[0]])

def get_errors(t, x):
    return np.abs(true_solution(t) - x)

def plotting_true(t0, tN):
    tplot = np.linspace(t0, tN, 1000)
    xplot = true_solution(tplot)
    plt.plot(tplot, xplot, 'k')

def plotting_method(t, x):
    plt.plot(t, x, 'ro')
    plt.show()


#%%
x0 = 1
xprime0 = 0
t0 = 0
tN = 1
dt = 0.1

t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = xprime0

for k in range(len(t) - 1):
    x[:, k + 1] = (4 + dt ** 2) / (4 - dt ** 2) * x[:, k] + (4 * dt) / (4 - dt ** 2) * f(x[:, k])

x_pred = x[0, :]

A7 = x_pred[-1]
A8 = get_errors(t[-1], x_pred[-1])

# print(A7, A8)
plotting_true(t0, tN)
plotting_method(t, x_pred)

#%%
"""Problem 2 (b)"""
x0 = 1
xprime0 = 0
t0 = 0
tN = 1
dt = 0.01

t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = xprime0

for k in range(len(t) - 1):
    x[:, k + 1] = (4 + dt ** 2) / (4 - dt ** 2) * x[:, k] + (4 * dt) / (4 - dt ** 2) * f(x[:, k])

x_pred = x[0, :]

A9 = x_pred[-1]
A10 = get_errors(t[-1], x_pred[-1])

# print(A9, A10)
# plotting_true(t0, tN)
# plotting_method(t, x_pred)

# %%
"""Problem 2 (c)"""
def true_solution2(t):
    return np.cos(t)

def f2(x):
    return np.array([x[1], -x[0]])

def get_errors2(t, x):
    return np.abs(true_solution2(t) - x)

def plotting_true2(t0, tN):
    tplot = np.linspace(t0, tN, 1000)
    xplot = true_solution2(tplot)
    plt.plot(tplot, xplot, 'k')

def plotting_method2(t, x):
    plt.plot(t, x, 'ro')
    plt.show()

#%%
x0 = 1
xprime0 = 0
t0 = 0
tN = 1
dt = 0.1

t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = xprime0

x[:, 1] = x[:, 0] + dt * f2(x[:, 0])
for k in range(1, len(t) - 1):
    x[:, k + 1] = x[:, k - 1] + 2 * dt * f2(x[:, k])

x_pred = x[0, :]

A11 = x_pred[-1]
A12 = get_errors2(t[-1], x_pred[-1])

# print(A11, A12)
# plotting_true2(t0, tN)
# plotting_method2(t, x_pred)


# %%
"""Problem 2 (d)"""
x0 = 1
xprime0 = 0
t0 = 0
tN = 1
dt = 0.01

t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = xprime0

x[:, 1] = x[:, 0] + dt * f2(x[:, 0])
for k in range(1, len(t) - 1):
    x[:, k + 1] = x[:, k - 1] + 2 * dt * f2(x[:, k])

x_pred = x[0, :]

A13 = x_pred[-1]
A14 = get_errors2(t[-1], x_pred[-1])

# print(A13, A14)
# plotting_true2(t0, tN)
# plotting_method2(t, x_pred)

#%%
"""For Written Report"""
# (1) Trapezoidal
x0 = 1
xprime0 = 0
t0 = 0
tN = 1

def trapezoidal(t0, tN, dt, x0, xprime0):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros((2, len(t)))
    x[0, 0] = x0
    x[1, 0] = xprime0

    for k in range(len(t) - 1):
        x[:, k + 1] = (4 + dt ** 2) / (4 - dt ** 2) * x[:, k] + (4 * dt) / (4 - dt ** 2) * f(x[:, k])

    x_pred = x[0, :]
    return get_errors(t[-1], x_pred[-1])

trap = np.zeros(7)
dts = np.array([2**-dt for dt in range(5, 12)])
i = 0
for dt in dts:
    trap[i] = trapezoidal(t0, tN, dt, x0, xprime0)
    i += 1

log_dts = np.log(dts)
log_trap = np.log(trap)
a, b = np.polyfit(log_dts, log_trap, 1)

plt.scatter(log_dts, log_trap, c = 'r')
plt.plot(log_dts, a * log_dts + b)
plt.title('Trapezoidal log-log plot')
plt.xlabel('ln($\Delta t$)')
plt.ylabel('ln($E_N$)')
plt.show()
print(a, b)

#%%
# (2) Midpoint
x0 = 1
xprime0 = 0
t0 = 0
tN = 1

def midpoint(t0, tN, dt, x0, xprime0):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros((2, len(t)))
    x[0, 0] = x0
    x[1, 0] = xprime0

    x[:, 1] = x[:, 0] + dt * f2(x[:, 0])
    for k in range(1, len(t) - 1):
        x[:, k + 1] = x[:, k - 1] + 2 * dt * f2(x[:, k])

    x_pred = x[0, :]
    return get_errors2(t[-1], x_pred[-1])

mid = np.zeros(7)
dts = np.array([2**-dt for dt in range(5, 12)])
i = 0
for dt in dts:
    mid[i] = midpoint(t0, tN, dt, x0, xprime0)
    i += 1

log_dts = np.log(dts)
log_mid = np.log(mid)
a, b = np.polyfit(log_dts, log_mid, 1)

plt.scatter(log_dts, log_mid, c = 'r')
plt.plot(log_dts, a * log_dts + b)
plt.title('Midpoint log-log plot')
plt.xlabel('ln($\Delta t$)')
plt.ylabel('ln($E_N$)')
plt.show()
print(a, b)

#%%
plt.title('Stability Region of Trapezoidal Method')
plt.xlabel('Re($\lambda \Delta t$)')
plt.ylabel('Im($\lambda \Delta t$)')
plt.xticks(np.arange(-2, 2.1, step=0.5))
plt.yticks(np.arange(-2, 2.1, step=0.5))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.fill_between([-2, 0], -2, 2, alpha=0.6, color='blue')
plt.fill_between([0, 2], -2, 2, alpha=0.3, color='white')
plt.show()

#%%
"""Report Finished"""

# %%
"""Problem 3"""
def p(x):
    return -x / (1 - x ** 2)

def q(x, alpha):
    return alpha ** 2 / (1 - x ** 2)

r = 0 

#%%
"""Problem 3 (a)"""
x0 = -0.5
y0 = -0.5
xN = 0.5
yN = 0.5
alpha = 1

def true_sol(x):
    return x

N = 11
x = np.linspace(x0, xN, 11)
dx = x[1] - x[0]

A = np.zeros((N, N))
b = np.zeros((N, 1))

A[0, 0] = 1
b[0] = y0
A[N - 1, N - 1] = 1
b[N - 1] = yN

for k in range(1, N - 1):
    A[k, k - 1] = (1 - dx * p(x[k]) / 2)
    A[k, k] = (-2 + dx ** 2 * q(x[k], alpha))
    A[k, k + 1] = (1 + dx * p(x[k]) / 2)

y = np.linalg.solve(A, b).reshape(N)

A15 = y[5]
A16 = max(abs(y - true_sol(x)))

# print(A15, A16)
# plt.plot(x, true_sol(x), 'k')
# plt.plot(x, y, 'b', x0, y0, 'ro', xN, yN, 'ro')
# plt.show()

#%%
"""Problem 3 (b)"""
x0 = -0.5
y0 = 0.5
xN = 0.5
yN = 0.5
alpha = 2

def true_sol2(x):
    return 1 - 2 * x ** 2

N = 11
x = np.linspace(x0, xN, 11)
dx = x[1] - x[0]

A = np.zeros((N, N))
b = np.zeros((N, 1))

A[0, 0] = 1
b[0] = y0
A[N - 1, N - 1] = 1
b[N - 1] = yN

for k in range(1, N - 1):
    A[k, k - 1] = (1 - dx * p(x[k]) / 2)
    A[k, k] = (-2 + dx ** 2 * q(x[k], alpha))
    A[k, k + 1] = (1 + dx * p(x[k]) / 2)

y = np.linalg.solve(A, b).reshape(N)

A17 = y[5]
A18 = max(abs(y - true_sol2(x)))

# print(A17, A18)
# plt.plot(x, true_sol2(x), 'k')
# plt.plot(x, y, 'b', x0, y0, 'ro', xN, yN, 'ro')
# plt.show()

# %%
"""Problem 3 (b)"""
x0 = -0.5
y0 = -1 / 3 
xN = 0.5
yN = 1 / 3
alpha = 3

def true_sol3(x):
    return x - 4 * x ** 3 / 3

N = 11
x = np.linspace(x0, xN, 11)
dx = x[1] - x[0]

A = np.zeros((N, N))
b = np.zeros((N, 1))

A[0, 0] = 1
b[0] = y0
A[N - 1, N - 1] = 1
b[N - 1] = yN

for k in range(1, N - 1):
    A[k, k - 1] = (1 - dx * p(x[k]) / 2)
    A[k, k] = (-2 + dx ** 2 * q(x[k], alpha))
    A[k, k + 1] = (1 + dx * p(x[k]) / 2)

y = np.linalg.solve(A, b).reshape(N)

A19 = y[5]
A20 = max(abs(y - true_sol3(x)))

# print(A19, A20)
# plt.plot(x, true_sol3(x), 'k')
# plt.plot(x, y, 'b', x0, y0, 'ro', xN, yN, 'ro')
# plt.show()

# %%
