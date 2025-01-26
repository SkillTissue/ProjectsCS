import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

width, height = 5, 5  
resolution = 512      
max_iterations = 100  
constant_modulus = 0.77

colors = ["#000056", "#0050A0", "#0063CA", "#1E7BAA", "#E07A5F", "#D34C48", "#EB4034", "#8A1D1A", "#000000"]
colormap = LinearSegmentedColormap.from_list("custom_map", colors, N=max_iterations)

def julia_set(x, y, c):
    z = complex(x, y)
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z**2 + c
        n += 1
    return n

x_vals = np.linspace(-width / 2, width / 2, resolution)
y_vals = np.linspace(-height / 2, height / 2, resolution)
X, Y = np.meshgrid(x_vals, y_vals)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim([-width / 2, width / 2])
ax.set_ylim([-height / 2, height / 2])
ax.set_xticks([])  
ax.set_yticks([])  

image = np.zeros((resolution, resolution))

def update_frame(a):
    c = constant_modulus * np.exp(1j * a)  
    for i in range(resolution):
        for j in range(resolution):
            iteration = julia_set(X[i, j], Y[i, j], c)
            image[i, j] = iteration

    ax.imshow(image, cmap=colormap, extent=(-width / 2, width / 2, -height / 2, height / 2))
    return [ax.imshow(image, cmap=colormap, extent=(-width / 2, width / 2, -height / 2, height / 2))]

a_values = np.linspace(0, 2 * np.pi, 600)
ani = FuncAnimation(fig, update_frame, frames=a_values, blit=False, interval=100)

ani.save("julia3.mp4", writer="ffmpeg", fps=60, dpi=150)