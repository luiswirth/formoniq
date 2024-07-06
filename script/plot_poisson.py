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

file_path = 'out/galsol.txt'
dim, nodes_per_dim, coefficients = read_coefficients(file_path)

x_grid = np.linspace(0, 1, nodes_per_dim)
y_grid = np.linspace(0, 1, nodes_per_dim)
z_grid = np.linspace(0, 1, nodes_per_dim)
X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

f_fe = coefficients.reshape([nodes_per_dim, nodes_per_dim, nodes_per_dim])
f_anal = f(X, Y, Z)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def update(z_idx):
  z = z_idx / nodes_per_dim
  ax1.clear()
  ax2.clear()
  
  X_slice = X[:, :, z_idx]
  Y_slice = Y[:, :, z_idx]
  f_fe_slice = f_fe[:, :, z_idx]
  f_anal_slice = f_anal[:, :, z_idx]

  ax1.plot_surface(X_slice, Y_slice, f_fe_slice, cmap='viridis')
  ax1.set_title(f'Finite Element Solution at z={z}')
  ax1.set_xlabel('X axis')
  ax1.set_ylabel('Y axis')
  ax1.set_zlabel('f')

  ax2.plot_surface(X_slice, Y_slice, f_anal_slice, cmap='viridis')
  ax2.set_title(f'Analytical Solution at z={z}')
  ax2.set_xlabel('X axis')
  ax2.set_ylabel('Y axis')
  ax2.set_zlabel('f')

ani = animation.FuncAnimation(fig, update, frames=nodes_per_dim, interval=200)

plt.tight_layout()
plt.show()
