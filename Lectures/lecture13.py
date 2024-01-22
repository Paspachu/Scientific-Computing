from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# u_xx + u_yy = 0
# u(x, 0) = x^2 - x
# All other boundaries are zero
# u(x, 1) = u(0, y) = u(1, y) = 0

x0 = 0
xN = 1
y0 = 0
yN = 1

N = 11
x = np.linspace(x0, xN, N)
y = np.linspace(y0, yN, N)

# Boundary condition u(x, 0)
def a(x):
    return x ** 2 - x

# These vectors are just to plot the boundary conditions
zero_vector = np.zeros_like(x)
one_vector = np.ones_like(x)

ax = plt.axes(projection='3d')
# u(0, y) = 0
ax.plot3D(zero_vector, y, zero_vector, 'r')
# u(x, 1) = 0
ax.plot3D(x, one_vector, zero_vector, 'r')
# u(1, y) = 0
ax.plot3D(one_vector, y, zero_vector, 'r')
# u(x, 0) = a(x)
ax.plot3D(x, zero_vector, a(x), 'r')
plt.show()