import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

fps = 30
duration = 5

file_path = 'out/wavesol.txt'
with open(file_path, 'r') as file:
    header = file.readline().strip()
    ndims, xfinal, nodes_per_dim, tfinal, nsteps = header.split()
    ndims = int(ndims)
    xfinal = float(xfinal)
    nodes_per_dim = int(nodes_per_dim)
    tfinal = float(tfinal)
    nsteps = int(nsteps)

    assert(ndims == 2)

    ndofs = nodes_per_dim**ndims

    coeffs = np.array([float(line.strip()) for line in file])

coeffs = coeffs.reshape(nsteps, nodes_per_dim, nodes_per_dim)

nframes = fps * duration
assert nsteps >= nframes
step_frame_interval = int(np.ceil(nsteps / nframes))
coeffs = coeffs[::step_frame_interval, :, :]
nframes = coeffs.shape[0]

x_grid = np.linspace(0, xfinal, nodes_per_dim)
y_grid = np.linspace(0, xfinal, nodes_per_dim)
X, Y = np.meshgrid(x_grid, y_grid)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

z_min, z_max = np.min(coeffs), np.max(coeffs)
z_range = z_max - z_min

def update(iframe):
    print(f"Plotting wave equation at frame={iframe}/{nframes} ...")
    t = iframe / (nframes - 1) * tfinal
    
    ax1.clear()
    Z = coeffs[iframe, :, :]
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title(f'Wave Equation - t={t:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)

ani = animation.FuncAnimation(fig, update, frames=nframes, interval=1)

ani.save('out/wave.gif', writer='pillow', fps=fps)
plt.close(fig)
