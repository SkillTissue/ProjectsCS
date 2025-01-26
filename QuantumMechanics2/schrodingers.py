import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, sph_harm

# Constants
a0 = 1  # Bohr radius

# Radial wavefunction
def laguerre(n, l, r):
    """ Radial part of the hydrogen wavefunction. """
    rho = 2 * r / (n * a0)
    return genlaguerre(n - l - 1, 2 * l + 1)(rho)

# Angular wavefunction
def Ye(m, l, theta, phi):
    """ Angular part (spherical harmonics). """
    return sph_harm(m, l, phi, theta)  # Note: phi first, theta second

# Full wavefunction
def psi(r, theta, phi, n, l, m):
    """ Full hydrogen wavefunction. """
    prefactor = np.sqrt((2 / (n * a0))**3 * np.math.factorial(n - l - 1) /
                        (2 * n * np.math.factorial(n + l)))
    radial_part = np.exp(-r / (n * a0)) * (2 * r / (n * a0))**l * laguerre(n, l, r)
    angular_part = Ye(m, l, theta, phi)
    return prefactor * radial_part * angular_part


N = 100 #points
# Spherical coordinate grid
phi = np.linspace(0, 2 * np.pi, N)  # Azimuthal angle
theta = np.linspace(0, np.pi, N)    # Polar angle
r = np.linspace(0, 20, N)           # Radial distance (positive values only)

# Create 3D spherical meshgrid
phi, theta = np.meshgrid(phi, theta)  # 2D grid for angles
r_fixed = 10  # Fix the radius for surface visualization

# Quantum numbers
n, l, m = 4, 3, 1

# Compute wavefunction on the grid
wavefunction = psi(r_fixed, theta, phi, n, l, m)
prob_density = np.abs(wavefunction)**2  # Probability density

# Scale the radius based on probability density
R = r_fixed * prob_density / prob_density.max()

# Convert spherical coordinates to Cartesian
X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot with color mapping
surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(prob_density / prob_density.max()),
                       rstride=4, cstride=4, antialiased=True, linewidth=0, alpha=1)
ax.set_box_aspect([1, 1, 1])

# Set title
plt.title(f"Hydrogen Wavefunction: n={n}, l={l}, m={m}")
plt.show()
