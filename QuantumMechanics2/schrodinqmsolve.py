import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, femtoseconds, m_e, Å
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d

#=========================================================================================================#
# First, we define the Hamiltonian of a single particle
#=========================================================================================================#

# Interaction potential
def double_slit(particle):
    b = 2.0 * Å  # slits separation
    a = 0.5 * Å  # slits width
    d = 1 * Å  # slits depth

    return np.where(((particle.x < -b / 2 - a) | (particle.x > b / 2 + a) | ((particle.x > -b / 2) & (particle.x < b / 2))) &
                    ((particle.y < d / 2) & (particle.y > -d / 2)), 1e5, 0)

# Build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m=m_e),
                potential=double_slit,
                spatial_ndim=2, N=512, extent=30 * Å)

#=========================================================================================================#
# Define the wavefunction at t = 0 (initial condition)
#=========================================================================================================#

def initial_wavefunction(particle):
    # This wavefunction corresponds to a Gaussian wave packet with a mean Y momentum equal to p_y0
    σ = 1.0 * Å
    v0 = 80 * Å / femtoseconds
    p_y0 = m_e * v0
    return (np.exp(-1 / (4 * σ**2) * ((particle.x - 0)**2 + (particle.y + 8 * Å)**2)) / 
            np.sqrt(2 * np.pi * σ**2) * np.exp(p_y0 * particle.y * 1j))

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#

total_time = 0.7 * femtoseconds
sim = TimeSimulation(hamiltonian=H, method="split-step")
sim.run(initial_wavefunction, total_time=total_time, dt=total_time / 300., store_steps=300)
t_eval = np.arange(0, total_time, total_time / 300)

#=========================================================================================================#
# Visualization of the time-dependent simulation
#=========================================================================================================#

# Define grid for plotting
x = np.linspace(-H.extent / 2, H.extent / 2, H.N)
y = np.linspace(-H.extent / 2, H.extent / 2, H.N)
X, Y = np.meshgrid(x, y)

b = 2.0   # slits separation
a = 0.5   # slits width
d = 0.5   # slits depth

# Extract the wavefunction
psi = sim.Ψ

V = np.zeros_like(X)
barrier_mask = ((X < -b / 2 - a) | (X > b / 2 + a) | ((X > -b / 2) & (X < b / 2))) & ((Y < d / 2) & (Y > -d / 2))
V[barrier_mask] = 10000

plt.figure(figsize=(8, 6))
plt.imshow(V, extent=(-H.extent / 2, H.extent / 2, -H.extent / 2, H.extent / 2), cmap="inferno")
plt.colorbar(label="Potential Barrier (V)")
plt.xlabel("x (Å)")
plt.ylabel("y (Å)")
plt.title("Potential Barrier Visualization")
plt.show()


# Create figure and axis for animation
fig = plt.figure(figsize=(8, 6), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# Set initial plot settings
ax.set_xlim(-H.extent / 2, H.extent / 2)
ax.set_ylim(-H.extent / 2, H.extent / 2)
ax.set_zlim(0, 0.1)
ax.set_xlabel("x (Å)", color="white")
ax.set_ylabel("y (Å)", color="white")
ax.set_zlabel("|ψ(x, y, t)|²", color="white")
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.tick_params(colors="white")
ax.grid(False)

ax.view_init(elev = 10, azim = -65, roll = 0)

print(psi.shape)

# Update function for animation
def update(frame):
    ax.clear()
    psi_abs2 = np.abs(psi[frame, :, :])**2  # Compute |psi|^2
    psi_abs2[V>0] = 0
    """
    ax.plot_surface(X, Y, V*0.00001, color='white', alpha=0.25, edgecolor="none")
    potential_surface = ax.plot_surface(
        #X, Y, V * 1e-6, cmap='inferno', edgecolor='none', alpha=0.25, label="Potential"
    )"""
    rect1 = patches.Rectangle((-30, 0), 30 -b/2 - a, 0.01, linewidth=1, edgecolor='black', facecolor='gray', alpha=1)
    ax.add_patch(rect1)
    art3d.pathpatch_2d_to_3d(rect1, z=0, zdir='y')
    rect2 = patches.Rectangle((-b/2, 0), b, 0.01, linewidth=1, edgecolor='black', facecolor='gray', alpha=1)
    ax.add_patch(rect2)
    art3d.pathpatch_2d_to_3d(rect2, z=0, zdir='y')
    rect3 = patches.Rectangle((b/2 + a, 0), 30 -b/2 - a, 0.01, linewidth=1, edgecolor='black', facecolor='gray', alpha=1)
    ax.add_patch(rect3)
    art3d.pathpatch_2d_to_3d(rect3, z=0, zdir='y')
    ax.plot_surface(X, Y, psi_abs2*1.0, cmap="viridis", edgecolor="none")
    print (f"frame: {frame}")
    ax.set_title(f"Time = {t_eval[frame]:.3f} fs", color="white")
    ax.set_xlim(-H.extent / 2, H.extent / 2)
    ax.set_ylim(-H.extent / 2, H.extent / 2)
    ax.set_zlim(0, 0.1)
    ax.set_xlabel("x (Å)", color="white")
    ax.set_ylabel("y (Å)", color="white")
    ax.set_zlabel("|ψ(x, y, t)|²", color="white")
    ax.tick_params(colors="white")

# Create and save animation
ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=50)
ani.save("schrodinger_2d_ydse_new_qm.mp4", fps=60, writer="ffmpeg")