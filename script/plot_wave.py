import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

file_path = 'out/wavesol.txt'
with open(file_path, 'r') as file:
    header = file.readline().strip()
    dim, nodes_per_dim, tfinal, nsteps = map(float, header.split())
    dim = int(dim)
    nodes_per_dim = int(nodes_per_dim)
    nsteps = int(nsteps)
    ndofs = nodes_per_dim**dim
    coefficients = np.array([float(line.strip()) for line in file])
nsteps = len(coefficients) // (nodes_per_dim * nodes_per_dim)
coefficients = coefficients.reshape(nsteps, nodes_per_dim, nodes_per_dim).T


x_grid = np.linspace(0, 1, nodes_per_dim)
y_grid = np.linspace(0, 1, nodes_per_dim)
X, Y = np.meshgrid(x_grid, y_grid)

f_fe = coefficients
nsteps = f_fe.shape[2]

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

z_min, z_max = np.min(f_fe), np.max(f_fe)
z_range = z_max - z_min

def update(istep):
    t = istep / (nsteps - 1) * tfinal
    
    ax1.clear()
    Z = f_fe[:, :, istep]
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title(f'Wave Equation - t={t:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)

ani = animation.FuncAnimation(fig, update, frames=nsteps, interval=10)

plt.show()
