import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'out/advectionsol.txt'
with open(file_path, 'r') as file:
    header = file.readline().strip()
    ndims, nodes_per_dim = header.split()
    ndims = int(ndims)
    nodes_per_dim = int(nodes_per_dim)

    assert(ndims == 2)

    ndofs = nodes_per_dim**ndims

    coeffs = np.array([float(line.strip()) for line in file])

coeffs = coeffs.reshape(nodes_per_dim, nodes_per_dim)

x_grid = np.linspace(0, 1.0, nodes_per_dim)
y_grid = np.linspace(0, 1.0, nodes_per_dim)
h = 1.0 / nodes_per_dim
X, Y = np.meshgrid(x_grid, y_grid)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(projection='3d')

z_min, z_max = np.min(coeffs), np.max(coeffs)
z_range = z_max - z_min

Z = coeffs[:, :]
ax.plot_surface(X, Y, Z, edgecolor='white', linewidth=500/nodes_per_dim)
ax.set_title(f'Linear Advection')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$u(x,y)$')
ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)

plt.show()
