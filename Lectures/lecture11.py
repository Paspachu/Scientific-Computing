import numpy as np
import matplotlib.pyplot as plt

# y'' + p(x)*y' + q(x)*y = r(x)
# y'' + 4*y = 0
# y(0) = 0 and y'(pi) = 1

x0 = 0
y0 = 0
xN = np.pi
yprimeN = 1

def true_sol(x):
    return np.sin(2 * x) / 2

p = 0
q = 4
r = 0

N = 31
x = np.linspace(x0, xN, N)
dx = x[1] - x[0]
print("dx = {}".format(dx))

A = np.zeros((N, N))
b = np.zeros((N, 1))

A[0, 0] = 1
b[0] = y0
A[N-1, N-1] = 1
A[N-1, N-2] = -1
b[N-1] = dx * yprimeN

for k in range(1, N - 1):
    A[k, k-1] = (1 - dx * p / 2)
    A[k, k] = (-2 + dx ** 2 * q)
    A[k, k + 1] = (1 + dx * p / 2)

y = np.linalg.solve(A, b).reshape(N)

plt.plot(x, true_sol(x), 'k')
plt.plot(x, y, 'b', x0, y0, 'ro')
plt.show()

err = np.max(np.abs(y - true_sol(x)))
print("Error = {}".format(err))