#%%
# Import all the required packages
import numpy as np
import matplotlib.pyplot as plt
import scipy

# %%
"""Problem 1 (a)"""
def f(x, t):
    return -4 * x * np.sin(t)

def true_solution(x0, t):
    return x0 * np.exp(4 * (np.cos(t) - 1))

def get_errors(x0, t, x):
    return np.abs(true_solution(x0, t) - x)

def plotting_true(x0, t0, tN):
    # True plot
    tplot = np.linspace(t0, tN, 1000)
    xplot = true_solution(x0, tplot)
    plt.plot(tplot, xplot, 'k')

def plotting_method(t, x):
    plt.plot(t, x, 'ro')
    plt.show()

# %%
# dt = 2^(-5)
x0 = 1
t0 = 0
tN = 8
dt = 2**-5

# Forward Euler method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k], t[k])

# Plot check
# plotting_true(x0, t0, tN)
# plotting_method(t, x)

#%%
# Local Truncation Error at n = 1
A1 = get_errors(x0, t[1], x[1])
A1

#%%
# Global Error
A2 = get_errors(x0, t[-1], x[-1]) 
A2

# %%
# dt = 2^(-6)
x0 = 1
t0 = 0
tN = 8
dt = 2**-6

# Forward Euler method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k], t[k])

# Plot check
# plotting_true(x0, t0, tN)
# plotting_method(t, x)

# %%
# Local Truncation Error at n = 1
A3 = get_errors(x0, t[1], x[1])
A3

#%%
# Global Error
A4 = get_errors(x0, t[-1], x[-1]) 
A4

# %%
"""Problem 1 (b)"""
# dt = 2^(-5)
x0 = 1
t0 = 0
tN = 8
dt = 2**-5

# Heun method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + 0.5 * dt * (f(x[k], t[k]) + f(x[k] + dt * f(x[k], t[k]), t[k+1]))

# Plot check
# plotting_true(x0, t0, tN)
# plotting_method(t, x)

#%%
# Local Truncation Error at n = 1
A5 = get_errors(x0, t[1], x[1])
A5

#%%
# Global Error
A6 = get_errors(x0, t[-1], x[-1]) 
A6

# %%
# dt = 2^(-6)
x0 = 1
t0 = 0
tN = 8
dt = 2**-6

# Heun method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + 0.5 * dt * (f(x[k], t[k]) + f(x[k] + dt * f(x[k], t[k]), t[k+1]))

# Plot check
# plotting_true(x0, t0, tN)
# plotting_method(t, x)

# %%
# Local Truncation Error at n = 1
A7 = get_errors(x0, t[1], x[1])
A7

#%%
# Global Error
A8 = get_errors(x0, t[-1], x[-1]) 
A8

# %%
"""Problem 1 (c)"""
# dt = 2^(-5)
x0 = 1
t0 = 0
tN = 8
dt = 2**-5

# RK2 method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k] + 0.5 * dt * f(x[k], t[k]), (t[k+1] + t[k]) / 2)

# Plot check
# plotting_true(x0, t0, tN)
# plotting_method(t, x)

#%%
# Local Truncation Error at n = 1
A9 = get_errors(x0, t[1], x[1])
A9

#%%
# Global Error
A10 = get_errors(x0, t[-1], x[-1]) 
A10

# %%
# dt = 2^(-6)
x0 = 1
t0 = 0
tN = 8
dt = 2**-6

# RK2 method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k] + 0.5 * dt * f(x[k], t[k]), (t[k+1] + t[k]) / 2)

# Plot check
# plotting_true(x0, t0, tN)
# plotting_method(t, x)

# %%
# Local Truncation Error at n = 1
A11 = get_errors(x0, t[1], x[1])
A11

#%%
# Global Error
A12 = get_errors(x0, t[-1], x[-1]) 
A12

#%%
"""For written report"""
x0 = 1
t0 = 0
tN = 8

#%%
# (a) Forward Euler
def forward_euler(t0, tN, dt, x0):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + dt * f(x[k], t[k])

    return get_errors(x0, t[-1], x[-1]) 

fe = np.zeros(7)
dts = np.array([2**-dt for dt in range(5, 12)])
i = 0
for dt in dts:
    fe[i] = forward_euler(t0, tN, dt, x0)
    i += 1
fe

#%%
log_dts = np.log(dts)
log_fe = np.log(fe)
a, b = np.polyfit(log_dts, log_fe, 1)

plt.scatter(log_dts, log_fe, c = 'r')
plt.plot(log_dts, a * log_dts + b)
plt.title('Forward Euler log-log plot')
plt.xlabel('ln($\Delta t$)')
plt.ylabel('ln($E_N$)')
plt.show()
print(a, b)

#%%
# (b) Heun's Method
def heun_method(t0, tN, dt, x0):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + 0.5 * dt * (f(x[k], t[k]) + f(x[k] + dt * f(x[k], t[k]), t[k+1]))

    return get_errors(x0, t[-1], x[-1]) 

fe = np.zeros(7)
dts = np.array([2**-dt for dt in range(5, 12)])
i = 0
for dt in dts:
    fe[i] = heun_method(t0, tN, dt, x0)
    i += 1
fe

#%%
log_dts = np.log(dts)
log_fe = np.log(fe)
a, b = np.polyfit(log_dts, log_fe, 1)

plt.scatter(log_dts, log_fe, c = 'r')
plt.plot(log_dts, a * log_dts + b)
plt.title("Heun's Method log-log plot")
plt.xlabel('ln($\Delta t$)')
plt.ylabel('ln($E_N$)')
plt.show()
print(a, b)

#%%
# (c) RK2
def RK2(t0, tN, dt, x0):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + dt * f(x[k] + 0.5 * dt * f(x[k], t[k]), (t[k+1] + t[k]) / 2)

    return get_errors(x0, t[-1], x[-1]) 

fe = np.zeros(7)
dts = np.array([2**-dt for dt in range(5, 12)])
i = 0
for dt in dts:
    fe[i] = RK2(t0, tN, dt, x0)
    i += 1
fe

#%%
log_dts = np.log(dts)
log_fe = np.log(fe)
a, b = np.polyfit(log_dts, log_fe, 1)

plt.scatter(log_dts, log_fe, c = 'r')
plt.plot(log_dts, a * log_dts + b)
plt.title("RK2 log-log plot")
plt.xlabel('ln($\Delta t$)')
plt.ylabel('ln($E_N$)')
plt.show()
print(a, b)

#%%
"""Report Finish"""

# %%
"""Problem 2 (a)"""
def f2(x):
    return 8 * np.sin(x)

def true_solution2(t):
    return 2 * np.arctan(np.e ** (8*t) / (1 + np.sqrt(2)))

def plotting_true2(t0, tN):
    tplot = np.linspace(t0, tN, 1000)
    xplot = true_solution2(tplot)
    plt.plot(tplot, xplot, 'k')

def plotting_method2(t, x):
    plt.plot(t, x, 'ro')
    plt.show()

# %%
x0 = np.pi / 4
t0 = 0
tN = 2
dt = 0.1

#%%
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0
x[1] = x[0] + dt * f2(x[0]+ 0.5 * dt * f2(x[0]))

for k in range(1, len(t) - 1):
    xP = x[k] + 0.5 * dt * (3 * f2(x[k]) - f2(x[k-1]))
    x[k + 1] = x[k] + 0.5 * dt * (f2(xP) + f2(x[k]))

# plotting_true2(t0, tN)
# plotting_method(t, x)

# %%
A13 = x[-1]
A14 = np.abs(x[-1] - true_solution2(tN))

#%%
"""Problem 2 (b)"""
x0 = np.pi / 4
t0 = 0
tN = 2
dt = 0.01

#%%
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0
x[1] = x[0] + dt * f2(x[0]+ 0.5 * dt * f2(x[0]))

for k in range(1, len(t) - 1):
    xP = x[k] + 0.5 * dt * (3 * f2(x[k]) - f2(x[k-1]))
    x[k + 1] = x[k] + 0.5 * dt * (f2(xP) + f2(x[k]))

# plotting_true2(t0, tN)
# plotting_method(t, x)

# %%
A15 = x[-1]
A16 = np.abs(x[-1] - true_solution2(tN))

# %%
"""Problem 3 (a)"""
v0 = 0.1
w0 = 1
t0 = 0
tN = 100

#%%
def I(t):
    return 0.1 * (5 + np.sin(np.pi * t / 10))

def f(t, x):
    a = 0.7
    b = 1
    tau = 12
    return np.array([x[0] - (x[0] ** 3) / 3 - x[1] + I(t), (a + x[0] - b * x[1]) / tau])


# %%
tspan = np.array([t0, tN])
init = np.array([v0, w0])
sol = scipy.integrate.solve_ivp(f, tspan, init, atol=1e-4, rtol=1e-4)
T = sol.t
X = sol.y[0, :]

#%%
avg_dt = 0
for k in range(1, len(T)):
    avg_dt += T[k] - T[k-1]
avg_dt /= len(T)

# %%
A17 = X[-1]
A18 = avg_dt

#%%
"""Problem 3 (b)"""
tspan = np.array([t0, tN])
init = np.array([v0, w0])
sol = scipy.integrate.solve_ivp(f, tspan, init, atol=1e-9, rtol=1e-9)
T = sol.t
X = sol.y[0, :]

#%%
avg_dt = 0
for k in range(1, len(T)):
    avg_dt += T[k] - T[k-1]
avg_dt /= len(T)

# %%
A19 = X[-1]
A20 = avg_dt

# %%
