import time
import numpy as np
import pyvista as pv
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, femtoseconds, m_e, Å
pv.global_theme.colorbar_orientation = 'vertical'

#=========================================================================================================#
# Define the Hamiltonian and wavefunction
#=========================================================================================================#

# Interaction potential
def double_slit(particle):
    b = 2.0 * Å  # slits separation
    a = 0.5 * Å  # slits width
    d = 1 * Å  # slits depth

    return np.where(((particle.x < -b / 2 - a) | (particle.x > b / 2 + a) | ((particle.x > -b / 2) & (particle.x < b / 2))) &
                    ((particle.y < d / 2) & (particle.y > -d / 2)), 1e5, 0)

def gaussian_well(particle):
    return 100*np.exp(-(particle.x**2 + particle.y**2))

# Build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m=m_e),
                potential=gaussian_well,
                spatial_ndim=2, N=512, extent=30 * Å)

# Initial wavefunction
def initial_wavefunction(particle):
    σ = 1.0 * Å
    v0 = 80 * Å / femtoseconds
    p_y0 = m_e * v0
    return (np.exp(-1 / (4 * σ**2) * ((particle.x - 0)**2 + (particle.y + 8 * Å)**2)) / 
            np.sqrt(2 * np.pi * σ**2) * np.exp(p_y0 * particle.y * 1j))

#=========================================================================================================#
# Run the simulation
#=========================================================================================================#

total_time = 0.7 * femtoseconds
sim = TimeSimulation(hamiltonian=H, method="split-step")
sim.run(initial_wavefunction, total_time=total_time, dt=total_time / 300., store_steps=300)
t_eval = np.arange(0, total_time, total_time / 300)

# Extract the wavefunction
psi = sim.Ψ

# Create grid for 3D visualization
x = np.linspace(-H.extent / 2, H.extent / 2, H.N)
y = np.linspace(-H.extent / 2, H.extent / 2, H.N)
X, Y = np.meshgrid(x, y)

#=========================================================================================================#
# 3D Visualization using pyvista
#=========================================================================================================#

# Parameters for the barrier
b = 2.0   # Slits separation
a = 0.5   # Slits width
d = 1.0   # Slits depth

pointa1, pointb1, pointc1 = [-28, 0, 0], [-b/2-a, 0, -0], [-b/2 -a, 0, 3]
pointa2, pointb2, pointc2 = [-b/2, 0, 0], [b/2, 0, 0], [b/2, 0, 3]
pointa3, pointb3, pointc3 = [b/2+a, 0, 0], [28, 0, 0], [28, 0, 3]

rect1 = pv.Rectangle([pointa1, pointb1, pointc1])
rect2 = pv.Rectangle([pointa2, pointb2, pointc2])
rect3 = pv.Rectangle([pointa3, pointb3, pointc3])

# Initialize the plotter
plotter = pv.Plotter(off_screen=True)

# Add dynamic wavefunction mesh
wave_grid = pv.StructuredGrid(Y, X, np.zeros_like(X))
wave_mesh = plotter.add_mesh(wave_grid, scalars = np.zeros_like(X))
#plotter.add_mesh(rect1, color="white")
#plotter.add_mesh(rect2, color="white")
#plotter.add_mesh(rect3, color="white")

# Function to update each frame
def update_wavefunction_plot(frame):
    psi_abs2 = np.abs(psi[frame, :, :])**2
    psi_phase = np.angle(psi[frame, :, :])  # Calculate phase
    psi_phase *= psi_abs2*100
    psi_phase = psi_phase.reshape(-1)

    Z = psi_abs2 * 500  # Scale height

    # Update the wavefunction mesh
    wave_grid.points[:, 2] = Z.ravel()  # Update z-axis

    plotter.update_scalars(psi_phase, mesh = wave_grid)
    plotter.add_title(f"Time = {t_eval[frame]:.3f} fs", font_size=20, color="black")

# Open movie file
plotter.open_movie("schrodinger_3d_gaussian_well_pyvista_with_phase.mp4", 60)

# Run animation
for frame in range(len(t_eval)):
    update_wavefunction_plot(frame)
    plotter.write_frame()

plotter.close()