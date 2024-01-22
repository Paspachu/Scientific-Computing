import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def bisection(f, a, b, tol):
    x = (a + b) / 2
    while np.abs(f(x)) >= tol:
        if np.sign(f(x)) == np.sign(f(a)):
            a = x
        else:
            b = x
        x = (a + b) / 2

    return x

t0 = 0
x0 = 4

tf = 1
xf = 1

def f(t, x):
    return np.array([x[1], 1.5 * x[0] ** 2])

def shoot(A):
    tspan = np.array([t0, tf])
    init_condition = np.array([x0, A])
    sol = solve_ivp(f, tspan, init_condition)
    return sol.y[0, -1]

A = bisection(lambda A: shoot(A) - xf, -30, -40, 1e-8)

tspan = np.array([t0, tf])
init_condition = np.array([x0, A])
t_eval = np.linspace(t0, tf, 1000)
sol = solve_ivp(f, tspan, init_condition, t_eval=t_eval)
t = sol.t
x = sol.y[0, :]
plt.plot(t, x, 'k', tf, xf, 'ro')
plt.show()

print("A = {}".format(A))