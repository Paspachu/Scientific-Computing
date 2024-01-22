import numpy as np
import matplotlib.pyplot as plt

# x'(t) = 0.5 * x = f(x, t)
# x(0) = 0.1
# t0 = 0, tN = 2, dt = 0.5

x0 = 0.1
t0 = 0
tN = 2
dt = 0.5

def f(x, t):
    return 0.5 * x

def true_solution(t):
    return x0 * np.exp(0.5 * t)

tplot = np.linspace(t0, tN, 1000)
xplot = true_solution(tplot)
plt.plot(tplot, xplot, 'k')

# plt.plot(t0, x0, 'ro')
#
# x1 = x0 + dt * f(x0, t0)
# t1 = t0 + dt
# plt.plot(t1, x1, 'ro')
#
# x2 = x1 + dt * f(x1, t1)
# t2 = t1 + dt
# plt.plot(t2, x2, 'ro')

t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros_like(t)
x[0] = x0

for k in range(len(t) - 1):
    x[k + 1] = x[k] + dt * f(x[k], t[k])

plt.plot(t, x, 'ro')
plt.show()