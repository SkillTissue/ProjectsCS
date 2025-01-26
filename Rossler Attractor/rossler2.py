import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def rossler(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# Parameters
a = 0.2
b = 0.2
c = 5.7

# Initial conditions
initial_state = [1.0, 1.0, 1.0]

# Time span
t_span = (0, 50)
t_eval = np.linspace(*t_span, 50*60)

# Solve the ODE
solution = solve_ivp(rossler, t_span, initial_state, args=(a, b, c), t_eval=t_eval)

# Extract the solution
x, y, z = solution.y

# Create the figure and axis
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# Set the viewing angle
ax.view_init(elev=30, azim=30)

# Set the limits based on the data
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))

# Customize plot to hide axes and ticks
ax.set_axis_off()
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

# Initial plot
line, = ax.plot([], [], [], color='blue')
glow_points = []

# Create layers for glowing effect
for i in range(6, 1, -1):
    gp, = ax.plot([], [], [], 'o', color='deepskyblue', markersize=0.5+0.5*(6-i), alpha=0.025*i, zorder=4)
    glow_points.append(gp)

# Initialize the animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    for gp in glow_points:
        gp.set_data([], [])
        gp.set_3d_properties([])
    return [line] + glow_points

# Update the animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    for i, gp in enumerate(glow_points):
        gp.set_data([x[frame]], [y[frame]])
        gp.set_3d_properties([z[frame]])
    return [line] + glow_points

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval = 5)

# Save the animation with higher quality
writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate = 1000)
ani.save('rossler_attractor_with_glow.mp4', writer=writer, dpi = 300)

