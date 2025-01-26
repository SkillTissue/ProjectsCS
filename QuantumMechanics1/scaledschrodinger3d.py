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
    if _x > 7.4:
        V[i] = 1300
    else:
        V[i] = 0

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

# Animation
fig = plt.figure(figsize=(12, 9), facecolor='black')  # Larger figure
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# Set the initial view
azimuth = 30
ax.view_init(elev=30, azim=azimuth)

# Customize the appearance of the 3D plot
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

# Hide panes
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

# Axis limits
ax.set_xlim(0, 15)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 3)

# Draw simple coordinate lines for axes
ax.plot([0, 15], [0, 0], [0, 0], 'w-', lw=1.5)  # X-axis
ax.plot([0, 0], [-1, 3], [0, 0], 'w-', lw=1.5)  # Y-axis
ax.plot([0, 0], [0, 0], [-1, 3], 'w-', lw=1.5)  # Z-axis

# Title and legend
title = ax.set_title('', color='white', fontsize=16)
line11, = ax.plot([], [], [], "w--", label=r"$V(x)$ (x0.001)", lw=2)
line12, = ax.plot([], [], [], "w", label=r"$Psi$", lw=2)
plt.legend(loc=1, fontsize=10, fancybox=False, edgecolor='white')

# Initial z0
z0 = [1.5] * len(x)

fig.patch.set_facecolor('black')

def init():
    line11.set_data(x, z0)
    line11.set_3d_properties(V * 0.001)
    return line11,

def animate(i):
    line12.set_data(x, np.real(sol.y[:, i]) + 1.5)
    line12.set_3d_properties(np.imag(sol.y[:, i]))
    global azimuth
    azimuth += 0.3
    ax.view_init(elev=30, azim=azimuth)
    title.set_text('Time = {0:1.3f}'.format(sol.t[i]))
    return line12,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(sol.t), interval=200, blit=True)

# Save the animation into a short video
print("Generating mp4")
anim.save('expstep3d_scaled_potentozone.mp4', fps=60, extra_args=['-vcodec', 'libx264'], dpi=300)
