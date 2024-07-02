import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def f(x, y):
  return np.exp(x * y)

file_path = 'out/galsol.txt'
with open(file_path, 'r') as file:
  header = file.readline().strip()
  (dim, nodes_per_dim) = tuple(map(int, header.split()))
  lines = file.readlines()

coefficients = np.array([float(line.strip()) for line in lines])

x_grid = np.linspace(0, 1, nodes_per_dim)
y_grid = np.linspace(0, 1, nodes_per_dim)
X, Y = np.meshgrid(x_grid, y_grid)

def galfunc(x):
  points = np.column_stack((X.flatten(), Y.flatten()))
  values = coefficients.flatten()
  
  idx_x = np.argmin(np.abs(x[0] - x_grid))
  idx_y = np.argmin(np.abs(x[1] - y_grid))
  
  if idx_x < nodes_per_dim - 1 and idx_y < nodes_per_dim - 1:
    v1 = values[idx_y * nodes_per_dim + idx_x]
    v2 = values[idx_y * nodes_per_dim + (idx_x + 1)]
    v3 = values[(idx_y + 1) * nodes_per_dim + (idx_x + 1)]
    v4 = values[(idx_y + 1) * nodes_per_dim + idx_x]
    
    # Bilinear interpolation weights
    x_diff = (x[0] - x_grid[idx_x]) / (x_grid[idx_x + 1] - x_grid[idx_x])
    y_diff = (x[1] - y_grid[idx_y]) / (y_grid[idx_y + 1] - y_grid[idx_y])
    
    # Perform bilinear interpolation
    interpolated_value = \
      (1 - x_diff) * (1 - y_diff) * v1 + \
      x_diff * (1 - y_diff) * v2 + \
      x_diff * y_diff * v3 + \
      (1 - x_diff) * y_diff * v4
  else:
    interpolated_value = values[idx_y * nodes_per_dim + idx_x]
  
  return interpolated_value

Z_fe = np.zeros_like(X)
Z_anal = np.zeros_like(X)

for i in range(len(x_grid)):
  for j in range(len(y_grid)):
    Z_fe[i,j] = galfunc(np.array([X[i,j], Y[i,j]]))
    Z_anal[i,j] = f(X[i,j], Y[i,j])

fig = plt.figure()

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_fe, cmap='viridis')
ax1.set_title('Finite Element Solution')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_anal, cmap='viridis')
ax2.set_title('Analytical Solution')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

plt.tight_layout()
plt.show()


