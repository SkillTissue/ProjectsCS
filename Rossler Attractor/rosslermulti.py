import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def rossler(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a*y
    dzdt = b + z * (x - c) 
    return [dxdt, dydt, dzdt]

a = 0.2
b = 0.2
c = 5.7

initial_states = []
solutions = []
init_state1 = [-5.0, -5.0, -5.0]
for i in range(10):
    x, y, z  = init_state1
    x, y, z = x+i, y+i, z+i
    initial_states.append([x, y, z])

print ("im done with initial state creation!")

# Time span
t_span = (0, 50)
t_eval = np.linspace(*t_span, 50*60)

for initialstate in initial_states:
    solution = solve_ivp(rossler, t_span, initialstate, args=(a, b, c), t_eval=t_eval)
    solutions.append(solution.y)


fig = plt.figure(facecolor='black', dpi = 300)
ax = fig.add_subplot(111, projection='3d', facecolor='black')

azimuth = 30
ax.view_init(elev=30, azim=azimuth)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Customize plot to hide axes and ticks
ax.set_axis_off()
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

lines = []
for i in range(len(solutions)):
    line, = ax.plot([], [], [], color='blue')
    lines.append(line)

glowing_points = []
for i in range(len(solutions)):
    glowing_point = []
    main_point, = ax.plot([], [], [], 'o', color = 'cyan', markersize = 0.7*(3), alpha = 1, zorder=4)
    glowing_point.append(main_point)
    for i in range(11, 1, -1):
        gp, = ax.plot([], [], [], 'o', color = 'cyan', markersize = 0.7*(12-i), alpha = 0.01*i, zorder=4)
        glowing_point.append(gp)
    glowing_points.append(glowing_point)


def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for glowing_point in glowing_points:
        for gp in glowing_point:
            gp.set_data([], [])
            gp.set_3d_properties([])
    return [lines] + glowing_points

def update(frame):
    for i in range(len(lines)):
        line = lines[i]
        x, y, z = solutions[i]
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
    global azimuth
    azimuth += 0.1
    ax.view_init(elev=30, azim=azimuth)
    for i in range(len(glowing_points)):
        glowing_point = glowing_points[i]
        x, y, z = solutions[i]
        for gp in (glowing_point):
            gp.set_data([x[frame]], [y[frame]])
            gp.set_3d_properties([z[frame]])
    return [lines] + glowing_points

print("prereq done")

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=False)

print("ani initiated")

# Save the animation
writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
ani.save('nigu_attractor_multi.mp4', writer=writer)
print ("ani saved")
