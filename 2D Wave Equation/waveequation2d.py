import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from wave_eqn2d import WaveEqn2D

# Raindrop probability (with each time tick) and intensity.
drop_probability, max_intensity = 0.005, 10
# Width of the Gaussian profile for each initial drop.
drop_width = 2
# Number of Gaussian widths to calculate to.
ndrop_widths = 3
# Size of the Gaussian template each drop is based on.
NDx = NDy = drop_width * ndrop_widths
Dx, Dy = np.arange(0, NDx, 1, dtype=np.int32), np.arange(0, NDy, 1, dtype=np.int32)
MDx, MDy = np.meshgrid(Dx, Dy)
# Create the 2D template of the initial drop.
cx, cy = NDx // 2, NDy // 2
gauss_template = np.exp(-(((MDx-cx)/drop_width)**2 + ((MDy-cy)/drop_width)**2))
template = np.sin(MDx -cx + MDy - cy)*np.exp(-(((MDx-cx)/drop_width)**2 + ((MDy-cy)/drop_width)**2))*(cx + cy)

dt = 1
nx = ny = 200
sim = WaveEqn2D(nx, ny, dt=dt, use_mur_abc=True)

fig, ax = plt.subplots()
ax.axis("off")
img = ax.imshow(sim.u[0], vmin=0, vmax=max_intensity, cmap='YlGnBu_r')

def update(i):
    """Advance the simulation by one tick."""
    # Random raindrops.
    if np.random.random() < drop_probability:
        x, y = np.random.randint(NDx//2, nx-NDx//2-1), np.random.randint(NDy//2, ny-NDy//2-1)
        sim.u[0, y-NDy//2:y+NDy//2, x-NDx//2:x+NDx//2] = max_intensity * template
    sim.update()

def init():
    """
    Initialization, because we're blitting and need references to the
    animated objects.
    """
    return img,

def animate(i):
    """Draw frame i of the animation."""
    update(i)
    img.set_data(sim.u[0])
    return img,

interval, nframes = 2*sim.dt, 600
ani = animation.FuncAnimation(fig, animate, frames=nframes,
                              repeat=False,
                              init_func=init, interval=interval, blit=True)
ani.save("droplets.mp4", fps=60, bitrate=500)