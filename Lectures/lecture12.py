import numpy as np
import matplotlib.pyplot as plt

# y'' + y^3 = 0
# y(0) = 0
# y(1) = 2

x0 = 0
y0 = 0
xN = 1
yN = 2

N = 101
x = np.linspace(x0, xN, N)
dx = x[1] - x[0]
# print("x = {}".format(x))
print("dx = {}".format(dx))

def F(y):
    z = np.zeros_like(y)
    z[0] = y[0] - y0
    z[-1] = y[-1] - yN
    for k in range(1, N - 1):
        z[k] = y[k - 1] - 2 * y[k] + y[k + 1] + dx ** 2 * y[k] ** 3
    return z

def jacobian(y):
    J = np.zeros((N, N))
    J[0, 0] = 1
    J[-1, -1] = 1
    for k in range(1, N-1):
        J[k, k - 1] = 1
        J[k, k] = -2 + 3 * dx ** 2 * y[k, 0] ** 2
        J[k, k + 1] = 1
    return J

y = np.ones((N, 1))
k = 0
max_steps = 500
while np.max(np.abs(F(y))) >= 1e-8 and k < max_steps:
    change_in_y = np.linalg.solve(jacobian(y), F(y))
    y = y - change_in_y
    k = k + 1

y = y.reshape(N)
print("k = {}".format(k))

plt.plot(x, y, 'k', x0, y0, 'ro', xN, yN, 'ro')
plt.show()