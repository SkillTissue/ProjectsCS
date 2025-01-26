import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

g = 9.8
ell = float(input("Enter length desired: "))
damp = float(input("Enter damping term: "))
f = float(input("Enter forcing term: "))

theta0 = np.deg2rad(float(input("Enter initial angle in degrees: ")))
theta_dot0 = float(input("Enter intitial velocity: "))

def pendulum_ODE(t, y):
    return (y[1], -g*np.sin(y[0])/ell - damp*y[1] + f*np.cos(t))

sol = solve_ivp(pendulum_ODE, [0, 20], (theta0, theta_dot0), 
    t_eval=np.linspace(0,20,60*20))

theta, theta_dot = sol.y
t = sol.t

theta_deg = np.rad2deg(sol.y[0])
theta_dot_deg = np.rad2deg(sol.y[1])

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['legend.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['grid.color'] = '#707070'

def pend_pos(theta):
    return (ell*np.sin(theta), -ell*np.cos(theta))

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1.25, 0.25)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])

x0, y0 = pend_pos(theta0)
line, = ax.plot([0, x0], [0, y0], lw=2, c='k')
circle = ax.add_patch(plt.Circle(pend_pos(theta0), 0.05, fc='k', zorder=3))

def animate(i):
    x,y = pend_pos(theta[i])
    line.set_data([0, x], [0, y])
    circle.set_center((x, y))

ani = animation.FuncAnimation(fig, animate, frames=len(t))
ffmpeg_writer = animation.FFMpegWriter(fps=60)
ani.save('pend.mp4', writer=ffmpeg_writer, dpi = 300)

print ("animation done!")







