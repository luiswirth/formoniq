import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_coefficients(file_path):
    with open(file_path, 'r') as file:
        header = file.readline().strip()
        dim, nodes_per_dim = map(int, header.split())
        coefficients = np.array([float(line.strip()) for line in file])
    nsteps = len(coefficients) // nodes_per_dim
    coefficients = coefficients.reshape(nodes_per_dim, nsteps)
    return dim, nodes_per_dim, coefficients

file_path = 'out/heatsol.txt'
dim, nodes_per_dim, coefficients = read_coefficients(file_path)

x_grid = np.linspace(0, 1, nodes_per_dim)
alpha = 1.0

f_fe = coefficients
nsteps = f_fe.shape[1]

fig, ax1 = plt.subplots()

y_min, y_max = np.min(f_fe), np.max(f_fe)
y_range = y_max - y_min

def update(istep):
    t = istep / (nsteps - 1) * 0.1  # Adjust based on final time
    f_anal = np.sin(np.pi * x_grid) * np.exp(-alpha * np.pi**2 * t)

    ax1.clear()
    ax1.plot(x_grid, f_fe[:, istep], color='blue', label='Finite Element Solution')
    ax1.plot(x_grid, f_anal, color='red', label='Analytical Solution')
    ax1.set_title(f'Heat Equation - t={t:.2f}')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('u')
    ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(True)
    ax1.legend()

ani = animation.FuncAnimation(fig, update, frames=f_fe.shape[1], interval=50)

plt.tight_layout()
plt.show()
