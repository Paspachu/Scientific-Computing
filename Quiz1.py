#%%
# Import all the required packages
import numpy as np
import matplotlib.pyplot as plt

#%%
"""Problem 1 (a)"""
def f(x):
    return -x**2

def true_solution(x0, t):
    return x0 * 1 / t

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
x0 = 1
t0 = 0
tN = 2
dt = 0.5

# Forward Euler method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k])

# Plot check
plotting_true(x0, t0, tN)
plotting_method(t, x)

# %%
x[-1]
# %%
x

#%%


# %%
def f(x, t):
    return np.cos(t) + np.exp(-x)

x0 = -1
t0 = 0
tN = 2
dt = 0.2

# RK2 method
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k] + 0.5 * dt * f(x[k], t[k]), (t[k+1] + t[k]) / 2)

x[1]
# %%
