import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre, factorial
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def hydrogen_wavefunction(n, l, m, r, theta, phi):
    # Constants
    a0 = 1.0  # Bohr radius

    # Radial component
    rho = 2 * r / (n * a0)
    prefactor_radial = np.sqrt((2 / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    radial_part = prefactor_radial * np.exp(-rho / 2) * rho**l * genlaguerre(n - l - 1, 2 * l + 1)(rho)

    # Angular component
    angular_part = sph_harm(m, l, phi, theta)

    # Full wavefunction
    psi = radial_part * angular_part
    return psi

def probability_density(n, l, m, grid):
    x, y, z = grid
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r, where=(r != 0), out=np.zeros_like(r))
    phi = np.arctan2(y, x)

    psi = hydrogen_wavefunction(n, l, m, r, theta, phi)
    return np.abs(psi)**2

def generate_orbital(n, l, m, grid_size=50, isovalue=0.0005):
    # Define grid in Cartesian coordinates
    x = np.linspace(-20, 20, grid_size)
    y = np.linspace(-20, 20, grid_size)
    z = np.linspace(-20, 20, grid_size)
    grid = np.meshgrid(x, y, z, indexing="ij")

    # Calculate probability density
    prob_density = probability_density(n, l, m, grid)
    print (np.max(prob_density))
    print (np.sum(prob_density))

    # Extract isosurface using marching cubes
    verts, faces, _, _ = marching_cubes(prob_density, level=isovalue)

    # Transform vertices to Cartesian coordinates
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    verts[:, 0] = verts[:, 0] * spacing[0] + x[0]
    verts[:, 1] = verts[:, 1] * spacing[1] + y[0]
    verts[:, 2] = verts[:, 2] * spacing[2] + z[0]

    return verts, faces, grid

def plot_orbital(verts, faces):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    mesh = Poly3DCollection(verts[faces], alpha=0.7, edgecolor='k')
    ax.add_collection3d(mesh)

    # Adjust plot limits
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

# Parameters for the orbital
n = 3
l = 2
m = 0

grid_size = 50
isovalue = 0.0000028

# Generate and plot orbital
verts, faces, _ = generate_orbital(n, l, m, grid_size, isovalue)
plot_orbital(verts, faces)
