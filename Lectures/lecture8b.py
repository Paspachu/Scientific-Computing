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

L = 4
x0 = -L
y0 = 0

xf = L
yf = 0

def f(x, y, beta):
    return np.array([y[1], -beta * y[0]])

def shoot(beta):
    A = 1
    tspan = np.array([x0, xf])
    init_condition = np.array([y0, A])
    sol = solve_ivp(lambda x, y: f(x, y, beta), tspan, init_condition)
    return sol.y[0, -1]

# Guess an initial velocity
A = 1
tspan = np.array([x0, xf])
x_eval = np.linspace(x0, xf, 1000)
init_condition = np.array([y0, A])

beta = bisection(shoot, 0.1, 0.2, 1e-8)
print("beta = {}".format(beta))
n = 1
print("beta was supposed to be {}".format((n * np.pi / (2 * L)) ** 2))
sol = solve_ivp(lambda x, y: f(x, y, beta), tspan, init_condition, t_eval=x_eval)
x = sol.t
y = sol.y[0, :]
plt.plot(x, y, 'k', xf, yf, 'ro')
plt.show()
