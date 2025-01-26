import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

WIDTH, HEIGHT = 800, 800
MAX_ITER = 512  

def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n + 1 - np.log(np.log2(abs(z)))
        z = z * z + c
    return max_iter

def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x, y = np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height)
    C = x[:, None] + 1j * y[None, :]
    mandelbrot_set = np.frompyfunc(lambda c: mandelbrot(c, max_iter), 1, 1)(C).astype(float)
    return mandelbrot_set

def update_plot(frame, zoom, ax):
    ax.clear()
    zoom_factor = zoom ** frame
    x_center, y_center = -0.7435, 0.1314 
    scale = 1.5 / zoom_factor
    
    xmin, xmax = x_center - scale, x_center + scale
    ymin, ymax = y_center - scale, y_center + scale
    
    mandelbrot_image = generate_mandelbrot(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, MAX_ITER)
    ax.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax), cmap='twilight_shifted', origin='lower')

fig, ax = plt.subplots(figsize=(8, 8))
zoom = 1.0

ani = animation.FuncAnimation(fig, update_plot, frames=240, fargs=(zoom, ax), interval=1000)

ani.save("mandelbrot_zoom2.mp4", writer="ffmpeg", dpi=150, fps=60)


