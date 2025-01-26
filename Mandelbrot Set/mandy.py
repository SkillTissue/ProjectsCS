import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set parameters
WIDTH, HEIGHT = 800, 800
MAX_ITER = 128
ZOOM = 1.05
x_center, y_center = -1.0, 0.1314

# Vectorized Mandelbrot calculation with D-Bail algorithm
def mandelbrot(c, max_iter):
    z = np.zeros_like(c, dtype=complex)
    escape = np.full(c.shape, max_iter, dtype=float)  # Initialize to max_iter
    mask = np.ones(c.shape, dtype=bool)  # Mask for points still being processed
    
    for i in range(max_iter):
        # Only calculate for points that haven't escaped
        z[mask] = z[mask] * z[mask] + c[mask]
        
        # Find points that have diverged (abs(z) >= 2)
        escaped = np.abs(z) >= 2
        escape[mask & escaped] = i + 1 - np.log(np.log2(np.abs(z[mask & escaped])))
        
        # Update the mask to keep processing non-escaped points
        mask[mask & escaped] = False
        
        # Break early if all points have escaped
        if not mask.any():
            break

    return escape

def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = x[:, None] + 1j * y[None, :]
    mandelbrot_set = mandelbrot(C, max_iter)
    return mandelbrot_set

def update_plot(frame, zoom, ax):
    print(f"Rendering frame {frame}...")
    ax.clear()
    zoom_factor = zoom ** frame
    scale = 1.5 / zoom_factor
    xmin, xmax = x_center - scale, x_center + scale
    ymin, ymax = y_center - scale, y_center + scale
    mandelbrot_image = generate_mandelbrot(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, MAX_ITER)
    ax.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax), cmap='twilight_shifted', origin='lower')
    ax.axis("off")

# Create and save animation 
fig, ax = plt.subplots(figsize=(8, 8))
ani = animation.FuncAnimation(fig, update_plot, frames=360, fargs=(ZOOM, ax), interval=1000, cache_frame_data=False)
ani.save("optimized_mandelbrot_zoom.mp4", writer="ffmpeg", dpi=150, fps=60)
