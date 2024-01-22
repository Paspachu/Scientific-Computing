import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

# Solve the ODE x' = 1x with x(0) = 1
# from t=0 to t=10
def f(t, x):
    return 1 * x

t0 = 0
tf = 10
tspan = np.array([t0, tf])
x0 = np.array([1])

sol = sp.solve_ivp(f, tspan, x0)
print(sol)

t = sol.t
x = sol.y[0, :]
plt.plot(t, x, 'k')
plt.show()

# Can produce more points.  This makes the plot smoother, but
# does NOT increase the accuracy.
tplot = np.linspace(t0, tf, 1000)

sol = sp.solve_ivp(f, tspan, x0, t_eval=tplot)
xplot = sol.y[0, :]
plt.plot(tplot, xplot, 'k')
plt.show()