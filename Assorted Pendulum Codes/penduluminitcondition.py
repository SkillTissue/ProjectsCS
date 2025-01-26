import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 9.8
ell = float(input("Enter length desired: "))

def pendulum_ODE(t, y):
    return (y[1], -g *np.sin(y[0]) / ell)

x = np.linspace(-5, 5, 100)

# Preallocate memory for solutions
theta_list = np.zeros((len(x), 60*20))

for i, initial_condition in (enumerate(x)):
    sol = solve_ivp(pendulum_ODE, [0, 20], (initial_condition, 0), t_eval=np.linspace(0, 20, 60*20))
    theta_list[i] = sol.y[0]

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

def animate(i):
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.plot(x, theta_list[:, i], color='white')

ani = animation.FuncAnimation(fig, animate, frames=60*20)

# Save animation
ffmpeg_writer = animation.FFMpegWriter(fps=60)
ani.save('pendulumapprox_vs_init.mp4', writer=ffmpeg_writer, dpi=300)
