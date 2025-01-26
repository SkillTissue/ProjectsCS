import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, femtoseconds, m_e, Å

#=========================================================================================================#
# First, define the Hamiltonian of a single particle
#=========================================================================================================#

# Interaction potential
def double_slit(particle):
    b = 2.0 * Å  # slits separation
    a = 0.5 * Å  # slits width
    d = 0.5 * Å  # slits depth

    return np.where(((particle.x < -b/2 - a) | (particle.x > b/2 + a) | ((particle.x > -b/2) 
                     & (particle.x < b/2))) & ((particle.y < d/2) & (particle.y > -d/2)), 1e5, 0)

# Build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m=m_e), 
                potential=double_slit, 
                spatial_ndim=2, N=256, extent=30 * Å)

#=========================================================================================================#
# Define the wavefunction at t = 0 (initial condition)
#=========================================================================================================#

def initial_wavefunction(particle):
    # This wavefunction corresponds to a Gaussian wavepacket with a mean Y momentum equal to p_y0
    σ = 1.0 * Å
    v0 = 80 * Å / femtoseconds
    p_y0 = m_e * v0
    return np.exp(-1/(4*σ**2) * ((particle.x-0)**2 + (particle.y+8*Å)**2)) / np.sqrt(2*np.pi*σ**2) * np.exp(p_y0*particle.y*1j)

#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#

total_time = 0.7 * femtoseconds
sim = TimeSimulation(hamiltonian=H, method="split-step")
sim.run(initial_wavefunction, total_time=total_time, dt=total_time/8000., store_steps=800)

#=========================================================================================================#
# Visualization using Matplotlib
#=========================================================================================================#

# Set up the figure and axis
fig, ax = plt.subplots()
extent = [-15 * Å, 15 * Å, -15 * Å, 15 * Å]
im = ax.imshow(np.abs(sim.states[0].wavefunction)**2, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=0.2)
ax.set_xlim(-15 * Å, 15 * Å)
ax.set_ylim(-15 * Å, 15 * Å)
ax.set_title("Wavefunction Probability Density")
ax.set_xlabel("x (Å)")
ax.set_ylabel("y (Å)")

# Animation function
def update(frame):
    im.set_data(np.abs(sim.simulation_states[frame].wavefunction)**2)
    return [im]

# Create the animation
ani = FuncAnimation(fig, update, frames=len(sim.wavefunction), interval=10, blit=True)

# Show the animation
plt.show()
