import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
  return np.exp(x * y)

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.title('Contour Plot of e^xy')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('3D Surface Plot of e^xy')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.grid(True)

plt.show()
