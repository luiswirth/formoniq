import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f(x, y, z):
    return np.exp(x * y * z)

def read_coefficients(file_path):
    with open(file_path, 'r') as file:
        header = file.readline().strip()
        dim, nodes_per_dim = map(int, header.split())
        coefficients = np.array([float(line.strip()) for line in file])
    return dim, nodes_per_dim, coefficients

def trilinear_interpolation(x, y, z, x_grid, y_grid, z_grid, values, nodes_per_dim):
    idx_x = np.searchsorted(x_grid, x) - 1
    idx_y = np.searchsorted(y_grid, y) - 1
    idx_z = np.searchsorted(z_grid, z) - 1

    if idx_x >= nodes_per_dim - 1 or idx_y >= nodes_per_dim - 1 or idx_z >= nodes_per_dim - 1:
        return values[idx_z * nodes_per_dim**2 + idx_y * nodes_per_dim + idx_x]

    x_diff = (x - x_grid[idx_x]) / (x_grid[idx_x + 1] - x_grid[idx_x])
    y_diff = (y - y_grid[idx_y]) / (y_grid[idx_y + 1] - y_grid[idx_y])
    z_diff = (z - z_grid[idx_z]) / (z_grid[idx_z + 1] - z_grid[idx_z])

    v1 = values[idx_z * nodes_per_dim**2 + idx_y * nodes_per_dim + idx_x]
    v2 = values[idx_z * nodes_per_dim**2 + idx_y * nodes_per_dim + (idx_x + 1)]
    v3 = values[idx_z * nodes_per_dim**2 + (idx_y + 1) * nodes_per_dim + (idx_x + 1)]
    v4 = values[idx_z * nodes_per_dim**2 + (idx_y + 1) * nodes_per_dim + idx_x]
    v5 = values[(idx_z + 1) * nodes_per_dim**2 + idx_y * nodes_per_dim + idx_x]
    v6 = values[(idx_z + 1) * nodes_per_dim**2 + idx_y * nodes_per_dim + (idx_x + 1)]
    v7 = values[(idx_z + 1) * nodes_per_dim**2 + (idx_y + 1) * nodes_per_dim + (idx_x + 1)]
    v8 = values[(idx_z + 1) * nodes_per_dim**2 + (idx_y + 1) * nodes_per_dim + idx_x]

    interpolated_value = (
        (1 - z_diff) * ((1 - x_diff) * ((1 - y_diff) * v1 + y_diff * v4) + x_diff * ((1 - y_diff) * v2 + y_diff * v3)) +
        z_diff * ((1 - x_diff) * ((1 - y_diff) * v5 + y_diff * v8) + x_diff * ((1 - y_diff) * v6 + y_diff * v7))
    )

    return interpolated_value

def galfunc(X, Y, Z, x_grid, y_grid, z_grid, values, nodes_per_dim):
    f_fe = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Z.shape[2]):
                f_fe[i, j, k] = trilinear_interpolation(X[i, j, k], Y[i, j, k], Z[i, j, k], x_grid, y_grid, z_grid, values, nodes_per_dim)
    return f_fe

file_path = 'out/galsol.txt'
dim, nodes_per_dim, coefficients = read_coefficients(file_path)

x_grid = np.linspace(0, 1, nodes_per_dim)
y_grid = np.linspace(0, 1, nodes_per_dim)
z_grid = np.linspace(0, 1, nodes_per_dim)
X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

f_fe = galfunc(X, Y, Z, x_grid, y_grid, z_grid, coefficients, nodes_per_dim)
f_anal = f(X, Y, Z)

fig = plt.figure()

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def update(a):
  ax1.clear()
  ax2.clear()
  
  X_slice = X[:, :, a]
  Y_slice = Y[:, :, a]
  f_fe_slice = f_fe[:, :, a]
  f_anal_slice = f_anal[:, :, a]

  ax1.plot_surface(X_slice, Y_slice, f_fe_slice, cmap='viridis')
  ax1.set_title('Finite Element Solution')
  ax1.set_xlabel('X axis')
  ax1.set_ylabel('Y axis')
  ax1.set_zlabel('f')

  ax2.plot_surface(X_slice, Y_slice, f_anal_slice, cmap='viridis')
  ax2.set_title('Analytical Solution')
  ax2.set_xlabel('X axis')
  ax2.set_ylabel('Y axis')
  ax2.set_zlabel('f')

ani = animation.FuncAnimation(fig, update, frames=nodes_per_dim, interval=200)

plt.tight_layout()
plt.show()
