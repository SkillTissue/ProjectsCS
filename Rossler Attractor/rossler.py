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
t_span = (0, 300)
t_eval = np.linspace(*t_span, 300*60)

# Solve the ODE
solution = solve_ivp(rossler, t_span, initial_state, args=(a, b, c), t_eval=t_eval)

# Extract the solution
x, y, z = solution.y

# Create the figure and axis
fig = plt.figure(facecolor='black', dpi = 300)
ax = fig.add_subplot(111, projection='3d', facecolor='black')

azimuth = 30
# Set the viewing angle
ax.view_init(elev=30, azim=azimuth)

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
glowing_point = []
main_point, = ax.plot([], [], [], 'o', color = 'cyan', markersize = 0.7*(3), alpha = 1, zorder=4)

for i in range(11, 1, -1):
    gp, = ax.plot([], [], [], 'o', color = 'cyan', markersize = 0.7*(12-i), alpha = 0.01*i, zorder=4)
    glowing_point.append(gp)

glowing_point.append(main_point)

# Initialize the animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    for gp in glowing_point:
        gp.set_data([], [])
        gp.set_3d_properties([])
    return [line] + glowing_point

# Update the animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    global azimuth
    azimuth += 0.1
    ax.view_init(elev=30, azim=azimuth)
    for i, gp in enumerate(glowing_point):
        gp.set_data([x[frame]], [y[frame]])
        gp.set_3d_properties([z[frame]])
    return [line] + glowing_point 

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True)

# Save the animation
writer = FFMpegWriter(fps=180, metadata=dict(artist='Me'), bitrate=1800)
ani.save('rossler_attractor.mp4', writer=writer)
