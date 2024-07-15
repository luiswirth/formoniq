import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

fps = 30
duration = 5

file_path = 'out/advection_transient_sol.txt'
with open(file_path, 'r') as file:
    header = file.readline().strip()
    ndims, nodes_per_dim, tfinal, nsteps = header.split()
    ndims = int(ndims)
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

x_grid = np.linspace(0.0, 1.0, nodes_per_dim)
y_grid = np.linspace(0.0, 1.0, nodes_per_dim)
h = 2.0 / nodes_per_dim
X, Y = np.meshgrid(x_grid, y_grid)

fig = plt.figure(figsize=(10, 5), dpi=150)

# Set up 3D subplot
ax3d = fig.add_subplot(121, projection='3d')

# Set up 2D heatmap subplot
ax2d = fig.add_subplot(122)


def update(iframe):
    t = iframe / (nframes - 1) * tfinal
    Z = coeffs[iframe, :, :]

    z_min, z_max = np.min(Z), np.max(Z)
    z_range = z_max - z_min

    ax3d.clear()
    ax2d.clear()

    # 3D surface plot
    ax3d.plot_surface(X, Y, Z, edgecolor='white', linewidth=50/nodes_per_dim, cmap='viridis')
    ax3d.set_title(f'3D Surface Plot\n$t={t:.2f}$')
    ax3d.set_xlabel('$x$')
    ax3d.set_ylabel('$y$')
    ax3d.set_zlabel('$u(x,y,t)$')
    ax3d.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)

    # 2D heatmap
    c = ax2d.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=z_min, vmax=z_max)
    ax2d.set_title(f'2D Heatmap\n$t={t:.2f}$')
    ax2d.set_xlabel('$x$')
    ax2d.set_ylabel('$y$')

    return c,

ani = animation.FuncAnimation(fig, update, frames=nframes, interval=1000/fps)
#plt.show()
ani.save(
    'out/advection.gif',
    fps=fps,
    progress_callback=lambda i, n: print(f"Saving Animation Frame {i}/{n}..."),
    writer='pillow',
)
plt.close(fig)
