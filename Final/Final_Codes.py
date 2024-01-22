#%%
# Import all the required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
"""Reproduction"""
def f(y, x):
    return 1.2 * y

def true_y(x):
    return np.exp(1.2 * x)

# Milne's method
def successive_approximation(approx, dt, t):
    xn1 = approx[0]
    x0 = approx[1]
    x1 = approx[2]
    x2 = approx[3]

    dxn1dt = f(xn1, t[0] - dt)
    dx0dt = f(x0, t[0])
    dx1dt = f(x1, t[1])
    dx2dt = f(x2, t[2])

    x2 = x0 + (dt / 3) * (dx2dt + 4 * dx1dt + dx0dt)
    x1 = x0 + (dt / 24) * (-dx2dt + 13 * dx1dt + 13 * dx0dt - dxn1dt)
    xn1 = x1 - (dt / 3) * (dx1dt + 4 * dx0dt + dxn1dt)

    return np.array([xn1, x0, x1, x2])

def get_first_four_approx_reproduction(x0, dt, t):
    xn1 = x0 - dt * f(x0, t[0])
    x1 = x0 + dt * f(x0, t[0])
    x2 = x0 + 2 * dt * f(x0, t[0])

    approx = np.array([xn1, x0, x1, x2])
    new_approx = successive_approximation(approx, dt, t)

    while np.max(np.abs(new_approx - approx)) > 1e-8:
        approx = new_approx
        new_approx = successive_approximation(approx, dt, t)

    xn1 = new_approx[0]
    x0 = new_approx[1]
    x1 = new_approx[2]
    x2 = new_approx[3]

    return np.array([xn1, x0, x1, x2])

#%%
# Initial Conditions are y0 = 1 and x0 = 0.
x0 = 0
xf = 1
dx = 0.1
y0 = 1

x = np.arange(x0 - dx, xf, dx)
y = np.zeros_like(x)
y[0:4] = get_first_four_approx_reproduction(y0, dx, x) 

for k in range(3, len(x) - 1):
    ypred = y[k-3] + (4 * dx / 3) * (2 * f(y[k], x[k])  - f(y[k-1], x[k-1]) + 2 * f(y[k-2], x[k-2]))
    y[k+1] = y[k-1] + (dx / 3) * (f(ypred, x[k+1]) + 4 * f(y[k], x[k]) + f(y[k-1], x[k-1]))

reprodcution = pd.DataFrame(data=np.vstack((x, y, true_y(x) - y, f(y, x))).T, columns=['x', 'y', 'error', 'y prime'])
reprodcution

# %%
"""First Order ODE"""
def f(x, t):
    return -4 * x * np.sin(t)
    
def true_solution(x0, t):
    return x0 * np.exp(4 * (np.cos(t) - 1))

def get_errors(x0, t, x):
    return np.abs(true_solution(x0, t) - x)

def plotting_true(x0, t0, tN):
    tplot = np.linspace(t0, tN, 1000)
    xplot = true_solution(x0, tplot)
    plt.plot(tplot, xplot, 'k', label='True Solution')

def plotting_method(t, x, title):
    plt.plot(t, x, 'ro', label='Approximation')
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.show()

def plotting_error(method, x0, t0, tN, dt, plot, title):
    errors = np.zeros(6)
    dts = np.array([2**-dt for dt in range(4, 10)])
    i = 0
    for dt in dts:
        errors[i] = method(x0, t0, tN, dt, False)
        i += 1
    errors

    log_dts = np.log(dts)
    log_errors = np.log(errors)
    a, b = np.polyfit(log_dts, log_errors, 1)

    if plot:
        plt.scatter(log_dts, log_errors, c = 'r', label='Exact Values')
        plt.plot(log_dts, a * log_dts + b, label='Best-fit')
        plt.title("{} log-log plot".format(title))
        plt.xlabel('ln($\Delta t$)')
        plt.ylabel('ln($E_N$)')
        plt.legend()
        plt.show()
    
    return a, b

# %%
# Define all the methods to test for the ODE of first order
def forward_euler(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + dt * f(x[k], t[k])

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Forward Euler")
    
    return get_errors(x0, t[-1], x[-1])

def backward_euler(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] / (1 + 4 * dt * np.sin(t[k+1]))

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Backward Euler")
    
    return get_errors(x0, t[-1], x[-1])

def heun(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + 0.5 * dt * (f(x[k], t[k]) + f(x[k] + dt * f(x[k], t[k]), t[k+1]))

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Heun")

    return get_errors(x0, t[-1], x[-1])

def trapezoidal(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = (x[k] + dt * f(x[k], t[k]) / 2) / (1 + 2 * dt * np.sin(t[k+1]))

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Trapezoidal")

    return get_errors(x0, t[-1], x[-1])

def RK4(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        k1 = dt * f(x[k], t[k])
        k2 = dt * f(x[k] + k1 / 2, t[k] + dt / 2)
        k3 = dt * f(x[k] + k2 / 2, t[k] + dt / 2)
        k4 = dt * f(x[k] + k3, t[k] + dt)
        x[k + 1] = x[k] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Runge-Kutta of Order 4")

    return get_errors(x0, t[-1], x[-1])

#%%
# Milne's method
def get_first_four_approx(x0, dt, t):
    xn1 = x0 - dt * f(x0, t[0])
    x1 = x0 + dt * f(x0, t[0])
    x2 = x0 + 2 * dt * f(x0, t[0])

    approx = np.array([xn1, x0, x1, x2])
    new_approx = successive_approximation(approx, dt, t)

    while np.max(np.abs(new_approx - approx)) > 1e-8:
        approx = new_approx
        new_approx = successive_approximation(approx, dt, t)

    xn1 = new_approx[0]
    x0 = new_approx[1]
    x1 = new_approx[2]
    x2 = new_approx[3]

    xpred = xn1 + (4 * dt / 3) * (2 * f(x2, t[2])  - f(x1, t[1]) + 2 * f(x0, t[0]))
    x3 = x1 + (dt / 3) * (f(xpred, t[3]) + 4 * f(x2, t[2]) + f(x1, t[1]))

    return np.array([x0, x1, x2, x3])

def milne(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0:4] = get_first_four_approx(x0, dt, t)

    for k in range(3, len(t) - 1):
        xpred = x[k-3] + (4 * dt / 3) * (2 * f(x[k], t[k])  - f(x[k-1], t[k-1]) + 2 * f(x[k-2], t[k-2]))
        x[k+1] = x[k-1] + (dt / 3) * (f(xpred, t[k+1]) + 4 * f(x[k], t[k]) + f(x[k-1], t[k-1]))

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Milne")

    return get_errors(x0, t[-1], x[-1])

# %%
# Plot true solution vs approximation when dt = 0.1
x0 = 1
t0 = 0
tN = 8
dt = 0.1

print(forward_euler(x0, t0, tN, dt, True))
print(backward_euler(x0, t0, tN, dt, True))
print(heun(x0, t0, tN, dt, True))
print(trapezoidal(x0, t0, tN, dt, True))
print(RK4(x0, t0, tN, dt, True))
print(milne(x0, t0, tN, dt, True))

# %%
# Plot log-log plots for all methods to verify the order of accuracy
a, _ = plotting_error(forward_euler, x0, t0, tN, dt, True, "Forward Euler")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(backward_euler, x0, t0, tN, dt, True, "Backward Euler")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(heun, x0, t0, tN, dt, True, "Heun")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(trapezoidal, x0, t0, tN, dt, True, "Trapezoidal")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(RK4, x0, t0, tN, dt, True, "RK4")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(milne, x0, t0, tN, dt, True, "Milne")
print("Slope of the best fit line is {}".format(a))

# %%
"""ODE from paper"""
def f(x, t):
    return 1.2 * x
    
def true_solution(x0, t):
    return x0 * np.exp(1.2 * t)

# Implicit methods should be updated as ODE changed
def backward_euler(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] / (1 - dt * 1.2)

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Backward Euler")
    
    return get_errors(x0, t[-1], x[-1])

def trapezoidal(x0, t0, tN, dt, plot):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = (x[k] + dt * f(x[k], t[k]) / 2) / (1 - dt * 0.6)

    if plot:
        plotting_true(x0, t0, tN)
        plotting_method(t, x, "Trapezoidal")

    return get_errors(x0, t[-1], x[-1])

#%%
# Plot true solution vs approximation when dt = 0.1
x0 = 1
t0 = 0
tN = 1
dt = 0.1

print(forward_euler(x0, t0, tN, dt, True))
print(backward_euler(x0, t0, tN, dt, True))
print(heun(x0, t0, tN, dt, True))
print(trapezoidal(x0, t0, tN, dt, True))
print(RK4(x0, t0, tN, dt, True))
print(milne(x0, t0, tN, dt, True))

# %%
# Plot log-log plots for all methods to verify the order of accuracy
a, _ = plotting_error(forward_euler, x0, t0, tN, dt, True, "Forward Euler")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(backward_euler, x0, t0, tN, dt, True, "Backward Euler")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(heun, x0, t0, tN, dt, True, "Heun")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(trapezoidal, x0, t0, tN, dt, True, "Trapezoidal")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(RK4, x0, t0, tN, dt, True, "RK4")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(milne, x0, t0, tN, dt, True, "Milne")
print("Slope of the best fit line is {}".format(a))

# %%
"""Second Order ODE"""
def f(y):
    return np.array([y[1], -4 * y[1] - 5 * y[0]])
    
def true_sol(x):
    return 2 * np.exp(6) / np.sin(3) * np.exp(-2 * x) * np.sin(x)

def get_errors(x, y):
    return np.max(np.abs(y - true_sol(x)))

def plotting_true(x):
    y = true_sol(x)
    plt.plot(x, y, 'k', label='True Solution')

def plotting_method(x, y, title):
    plt.plot(x, y, 'ro', label='Approximation')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.legend()
    plt.show()

def plotting_error(method, x0, xN, N, y0, plot, title):
    errors = np.zeros(6)
    dxs = np.array([2**-dx for dx in range(4, 10)])
    i = 0
    for dx in dxs:
        N = int((xN - x0) / dx) + 1
        errors[i] = method(x0, xN, N, y0, False)
        i += 1
    errors

    log_dxs = np.log(dxs)
    log_errors = np.log(errors)
    a, b = np.polyfit(log_dxs, log_errors, 1)

    if plot:
        plt.scatter(log_dxs, log_errors, c = 'r', label='Exact Values')
        plt.plot(log_dxs, a * log_dxs + b, label='Best-fit')
        plt.title("{} log-log plot".format(title))
        plt.xlabel('ln($\Delta x$)')
        plt.ylabel('ln(Error)')
        plt.legend()
        plt.show()
    
    return a, b

#%%
# Define all the methods to test for the ODE of second order
def forward_euler(x0, xN, N, y0, plot):
    x = np.linspace(x0, xN, N)
    dx = x[1] - x[0]
    y = np.zeros((2, N))
    y[:, 0] = y0

    for k in range(len(x) - 1):
        y[:, k + 1] = y[:, k] + dx * f(y[:, k])

    if plot:
        plotting_true(x)
        plotting_method(x, y[0, :], "Forward Euler")
    
    return get_errors(x, y[0, :])

def backward_euler(x0, xN, N, y0, plot):
    x = np.linspace(x0, xN, N)
    dx = x[1] - x[0]
    y = np.zeros((2, N))
    y[:, 0] = y0

    for k in range(len(x) - 1):
        y[0, k + 1] = ((1 + 4 * dx) * y[0, k] + dx * y[1, k]) / (1 + 4 * dx + 5 * dx ** 2)
        y[1, k + 1] = (-5 * dx * y[0, k] + y[1, k]) / (1 + 4 * dx + 5 * dx ** 2)

    if plot:
        plotting_true(x)
        plotting_method(x, y[0, :], "Backward Euler")
    
    return get_errors(x, y[0, :])

def heun(x0, xN, N, y0, plot):
    x = np.linspace(x0, xN, N)
    dx = x[1] - x[0]
    y = np.zeros((2, N))
    y[:, 0] = y0

    for k in range(len(x) - 1):
        y[:, k + 1] = y[:, k] + 0.5 * dx * (f(y[:, k]) + f(y[:, k] + dx * f(y[:, k])))

    if plot:
        plotting_true(x)
        plotting_method(x, y[0, :], "Heun")
    
    return get_errors(x, y[0, :])

def trapezoidal(x0, xN, N, y0, plot):
    x = np.linspace(x0, xN, N)
    dx = x[1] - x[0]
    y = np.zeros((2, N))
    y[:, 0] = y0

    for k in range(len(x) - 1):
        y[0, k + 1] = ((1 + 2 * dx - 5 * dx ** 2 / 4) * y[0, k] + dx * y[1, k]) / (1 + 2 * dx + 5 * dx ** 2 / 4)
        y[1, k + 1] = (-5 * dx * y[0, k] + (1 - 2 * dx - 5 * dx ** 2 / 4) * y[1, k]) / (1 + 2 * dx + 5 * dx ** 2 / 4)

    if plot:
        plotting_true(x)
        plotting_method(x, y[0, :], "Trapezoidal")
    
    return get_errors(x, y[0, :])

def RK4(x0, xN, N, y0, plot):
    x = np.linspace(x0, xN, N)
    dx = x[1] - x[0]
    y = np.zeros((2, N))
    y[:, 0] = y0

    for k in range(len(x) - 1):
        k1 = dx * f(y[:, k])
        k2 = dx * f(y[:, k] + 0.5 * k1)
        k3 = dx * f(y[:, k] + 0.5 * k2)
        k4 = dx * f(y[:, k] + k3)
        y[:, k + 1] = y[:, k] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    if plot:
        plotting_true(x)
        plotting_method(x, y[0, :], "RK4")
    
    return get_errors(x, y[0, :])

#%%
# Milne's method
def f_m(y, yprime):
    return -4 * yprime - 5 * y

def successive_approximation(approx, dx):
    yn1 = approx[0]
    y0 = approx[1]
    y1 = approx[2]
    y2 = approx[3]
    yn1p = approx[4]
    y0p = approx[5]
    y1p = approx[6]
    y2p = approx[7]

    yn1pp = f_m(yn1, yn1p)
    y0pp = f_m(y0, y0p)
    y1pp = f_m(y1, y1p)
    y2pp = f_m(y2, y2p)

    y2p = y0p + (dx / 3) * (y2pp + 4 * y1pp + y0pp)
    y1p = y0p + (dx / 24) * (-y2pp + 13 * y1pp + 13 * y0pp - yn1pp)
    yn1p = y1p - (dx / 3) * (y1pp + 4 * y0pp + yn1pp)

    y2 = y0 + (dx / 3) * (y2p + 4 * y1p + y0p)
    y1 = y0 + (dx / 24) * (-y2p + 13 * y1p + 13 * y0p - yn1p)
    yn1 = y1 - (dx / 3) * (y1p + 4 * y0p + yn1p)

    return np.array([yn1, y0, y1, y2, yn1p, y0p, y1p, y2p])

def get_first_four_approx(y0, dx):
    yn1p= y0[1] - dx * f_m(y0[0], y0[1])
    y1p = y0[1] + dx * f_m(y0[0], y0[1])
    y2p = y0[1] + 2 * dx * f_m(y0[0], y0[1])

    yn1 = y0[0] - dx * yn1p
    y1 = y0[0] + dx * y1p
    y2 = y0[0] + 2 * dx * y2p

    approx = np.array([yn1, y0[0], y1, y2, yn1p, y0[1], y1p, y2p])
    new_approx = successive_approximation(approx, dx)

    while np.max(np.abs(new_approx - approx)) > 1e-8:
        approx = new_approx
        new_approx = successive_approximation(approx, dx)

    yn1 = new_approx[0]
    y0 = new_approx[1]
    y1 = new_approx[2]
    y2 = new_approx[3]
    yn1p = new_approx[4]
    y0p = new_approx[5]
    y1p = new_approx[6]
    y2p = new_approx[7]

    yppred = yn1p + (4 * dx / 3) * (2 * f_m(y2, y2p)  - f_m(y1, y1p) + 2 * f_m(y0, y0p))
    y3 = y1 + (dx / 3) * (yppred + 4 * y2p + y1p)
    y3p = y1p + (dx / 3) * (f_m(y3, yppred) + 4 * f_m(y2, y2p) + f_m(y1, y1p))

    return np.array([[y0, y1, y2, y3], [y0p, y1p, y2p, y3p]])

def milne(x0, xN, N, y0, plot):
    x = np.linspace(x0, xN, N)
    dx = x[1] - x[0]
    y = np.zeros((2, N))
    y[:, 0] = y0
    y[:, 0:4] = get_first_four_approx(y0, dx)

    for k in range(3, len(x) - 1):
        yppred = y[1, k-3] + (4 * dx / 3) * (2 * f_m(y[0, k], y[1, k])  - f_m(y[0, k-1], y[1, k-1]) + 2 * f_m(y[0, k-2], y[1, k-2]))
        y[0, k+1] = y[0, k-1] + (dx / 3) * (yppred + 4 * y[1, k] + y[1, k-1])
        y[1, k+1] = y[1, k-1] + (dx / 3) * (f_m(y[0, k+1], yppred) + 4 * f_m(y[0, k], y[1, k]) + f_m(y[0, k-1], y[1, k-1]))

    if plot:
        plotting_true(x)
        plotting_method(x, y[0, :], "Milne")
    
    return get_errors(x, y[0, :])


#%%
# Plot true solution vs approximation when dx = 0.1
N = 31
x0 = 0
xN = 3
y0 = [0, 2 * np.exp(6) / np.sin(3)]

print(forward_euler(x0, xN, N, y0, True))
print(backward_euler(x0, xN, N, y0, True))
print(heun(x0, xN, N, y0, True))
print(trapezoidal(x0, xN, N, y0, True))
print(RK4(x0, xN, N, y0, True))
print(milne(x0, xN, N, y0, True))

# %%
# Plot log-log plots for all methods to verify the order of accuracy
a, _ = plotting_error(forward_euler, x0, xN, N, y0, True, "Forward Euler")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(backward_euler, x0, xN, N, y0, True, "Backward Euler")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(heun, x0, xN, N, y0, True, "Heun")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(trapezoidal, x0, xN, N, y0, True, "Trapezoidal")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(RK4, x0, xN, N, y0, True, "RK4")
print("Slope of the best fit line is {}".format(a))

a, _ = plotting_error(milne, x0, xN, N, y0, True, "Milne")
print("Slope of the best fit line is {}".format(a))

# %%
