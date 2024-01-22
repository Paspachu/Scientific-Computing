import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def bisection(f, a, b, tol):
    x = (a + b) / 2
    while np.abs(b - a) >= tol:
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

A = 1
def shoot(beta):
    tspan = np.array([x0, xf])
    init_condition = np.array([y0, A])
    sol = solve_ivp(lambda x, y: f(x, y, beta), tspan, init_condition)
    return sol.y[0, -1]

beta = 0
sign = np.sign(shoot(beta))
dbeta = 0.1

num_modes = 5
eigenvals = np.zeros(num_modes)
k = 0

while k < num_modes:
    beta_next = beta + dbeta
    sign_next = np.sign(shoot(beta_next))
    if sign != sign_next:
        eigenvals[k] = bisection(shoot, beta, beta_next, 1e-8)
        k = k + 1
    beta = beta_next
    sign = sign_next

for k in range(num_modes):
    print("eigenval = {}".format(eigenvals[k]))
    print("Predicted value = {}".format(((k + 1) * np.pi / (2 * L)) ** 2))
    tspan = np.array([x0, xf])
    x_eval = np.linspace(x0, xf, 1000)
    init_condition = np.array([y0, A])
    sol = solve_ivp(lambda x, y: f(x, y, eigenvals[k]),
                    tspan, init_condition,
                    t_eval=x_eval)
    x = sol.t
    y = sol.y[0, :]
    plt.plot(x, y)

plt.show()