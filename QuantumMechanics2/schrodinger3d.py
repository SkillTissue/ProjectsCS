import numpy as np
from scipy import integrate
from scipy import sparse

import matplotlib.pyplot as plt
from matplotlib import animation
plt.rc('savefig', dpi=300)


# Set initial conditions
dx = 0.01                  # spatial separation
x = np.arange(0, 15, dx)    # spatial grid points

kx = 50                     # wave number
m = 1                       # mass
sigma = 0.5                 # width of initial gaussian wave-packet
x0 = 3.0                    # center of initial gaussian wave-packet


# Initial Wavefunction
A = 1.0 / (sigma * np.sqrt(np.pi))  # normalization constant
psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)


# Potential V(x)

V = np.zeros(x.shape)
for i, _x in enumerate(x):
    if _x > 7.25 and _x < 7.75:
        V[i] = 1000
    else:
        V[i] = 0


# Make a plot of psi0 and V
fig = plt.figure(figsize=(15, 5))
plt.plot(x, V*0.01, "k--", label=r"$V(x) (x0.01)")
plt.plot(x, np.abs(psi0)**2, "r", label=r"$\vert\psi(t=0,x)\vert^2$")
plt.plot(x, np.real(psi0), "g", label=r"$Re\{\psi(t=0,x)\}$")
plt.legend(loc=1, fontsize=8, fancybox=False)
fig.savefig('expstep_initial@2x.png')

print("Total Probability: ", np.sum(np.abs(psi0)**2)*dx)


# Laplace Operator (Finite Difference)
D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2


# Solve Schrodinger Equation
hbar = 1
# hbar = 1.0545718176461565e-34
# RHS of Schrodinger Equation
def psi_t(t, psi):
    return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)


# Solve the Initial Value Problem
dt = 0.001  # time interval for snapshots
t0 = 0.0    # initial time
tf = 1  # final time
t_eval = np.arange(t0, tf, dt)  # recorded time shots

print("Solving initial value problem")
sol = integrate.solve_ivp(psi_t,
                          t_span=[t0, tf],
                          y0=psi0,
                          t_eval=t_eval,
                          method="RK23")

print ("Solved! ")


# Plotting
fig = plt.figure(figsize=(6, 4))
for i, t in enumerate(sol.t):
    plt.plot(x, np.abs(sol.y[:, i])**2)                  # Plot Wavefunctions
    print("Total Prob. in frame", i, "=", np.sum(np.abs(sol.y[:, i])**2)*dx)   # Print Total Probability (Should = 1)
plt.plot(x, V * 0.001, "k--", label=r"$V(x) (x0.001)")   # Plot Potential
plt.legend(loc=1, fontsize=8, fancybox=False)
fig.savefig('sinestep@2x.png')


# Animation
fig = plt.figure(figsize=(8, 6), facecolor = 'black')
ax = fig.add_subplot(111, projection = '3d', facecolor = 'black')

azimuth = 30
ax.view_init(elev=30, azim=azimuth)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

ax.set_xlim(0, 15)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 3)
title = ax.set_title('')
line11, = ax.plot([], [], [], "w--", label=r"$V(x)$ (x0.001)")
line12, = ax.plot([], [], [], "w", label=r"$psi$")
plt.legend(loc=1, fontsize=8, fancybox=False)

print (np.real(sol.y))
z0 = []
for i in range(len(x)):
    z0.append(1.5)

fig.patch.set_facecolor('black')

def init():
    line11.set_data(x, z0)
    line11.set_3d_properties(V * 0.001)
    return line11,


def animate(i):
    line12.set_data(x, np.real(sol.y[:, i])+1.5)
    line12.set_3d_properties(np.imag(sol.y[:, i]))
    global azimuth
    azimuth += 0.1
    ax.view_init(elev=30, azim=azimuth)
    title.set_text('Time = {0:1.3f}'.format(sol.t[i]))
    print (f"Frame: {i}")
    return line12, 


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(sol.t), interval=200, blit=True)


# Save the animation into a short video
print("Generating mp4")
anim.save('expstep3d.mp4', fps=60, extra_args=['-vcodec', 'libx264'], dpi=300)
# anim.save('step@2x.gif', writer='pillow', fps=15)
# anim.save('step@2x.gif', writer='imagemagick', fps=15, dpi=150, extra_args=['-layers Optimize'])