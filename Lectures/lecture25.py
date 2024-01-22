import numpy as np
import matplotlib.pyplot as plt

# u''(x) = f(x)
# u(x0) = 0 = u(xf)

def f(x):
    return 1 / (1 + 16 * x ** 2)

def true_sol(x):
    return (1 / 32) * (-np.log(16 * x ** 2 + 1) + np.log(17) + 8 * x * np.arctan(4 * x) - 8 * np.arctan(4))

N = 5
x0 = -1
xf = 1
x = np.linspace(x0, xf, N + 1)

A = np.zeros((N + 1, N + 1))
b = np.zeros((N + 1, 1))
b[0] = 0
b[N] = 0
for i in range(1, N):
    b[i] = f(x[i])

# Fill in first row: u(x0) = 0
for j in range(N + 1):
    A[0, j] = x[0] ** (N - j)

# Fill in last row: u(xf) = 0
for j in range(N + 1):
    A[N, j] = x[N] ** (N - j)

for i in range(1, N):
    for j in range(N - 1):
        A[i, j] = (N - j) * (N - j - 1) * x[i] ** (N - j - 2)

c = np.linalg.solve(A, b)

def u(x):
    # y = 0
    # for j in range(N + 1):
    #     y = y + c[j] * x ** (N - j)
    # return y
    return np.polyval(c, x)

xplot = np.linspace(x0, xf, 1000)
plt.plot(xplot, true_sol(xplot), 'k')
plt.plot(xplot, u(xplot), 'b')
plt.show()

err = np.max(np.abs(true_sol(x) - u(x)))
print("Error = {}".format(err))