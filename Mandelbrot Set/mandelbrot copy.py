import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters for Mandelbrot set generation
WIDTH, HEIGHT = 800, 450  # Wider aspect ratio for horizontal orientation
MAX_ITER = 256  # Maximum iterations to check for points inside the Mandelbrot set

# Function to compute continuous iteration count for smooth coloring
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            # Calculate smooth iteration count
            return n + 1 - np.log(np.log2(abs(z)))
        z = z * z + c
    return max_iter

# Generate the Mandelbrot set for a given range of coordinates and resolution
def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x, y = np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height)
    C = x[:, None] + 1j * y[None, :]
    mandelbrot_set = np.frompyfunc(lambda c: mandelbrot(c, max_iter), 1, 1)(C).astype(float)
    return mandelbrot_set

# Create animation function
def update_plot(frame, zoom, ax):
    ax.clear()
    zoom_factor = zoom ** frame
    x_center, y_center = -0.7435, 0.1314  # Coordinates to zoom into (can be adjusted)
    scale = 1.5 / zoom_factor
    
    # Calculate new bounds
    xmin, xmax = x_center - scale, x_center + scale
    ymin, ymax = y_center - scale * HEIGHT / WIDTH, y_center + scale * HEIGHT / WIDTH  # Adjust for aspect ratio
    
    # Generate and display Mandelbrot set for new bounds
    mandelbrot_image = generate_mandelbrot(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, MAX_ITER)
    ax.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax), cmap='twilight_shifted', origin='lower')

# Set up the figure and animation
fig, ax = plt.subplots(figsize=(10, 6))  # Wider aspect ratio for horizontal orientation
zoom = 1.12  # Adjusted zoom factor for a slower zoom

# Generate animation with more frames and 60 FPS
ani = animation.FuncAnimation(fig, update_plot, frames=240, fargs=(zoom, ax), interval=1000/60)  # 60 FPS

# Save as a video or GIF (optional)
ani.save("mandelbrot_zoom2.mp4", writer="ffmpeg", fps=60, dpi=150)
# To save as GIF, use ani.save("mandelbrot_zoom.gif", writer="imagemagick", fps=60, dpi=80)

plt.show()
